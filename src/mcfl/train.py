import logging
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mcfl.client import Client, ClientUpdate, sample_clients
from mcfl.config import Config
from mcfl.dataset import NistDataManager
from mcfl.experiment_logger import LOG_FORMAT, ExperimentLogger
from mcfl.init_centers import initialize_centers
from mcfl.model import SimpleCNN
from mcfl.server import ClientEval, Server, macro_accuracy, summarize

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """再現性のためのシード固定"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def log_environment(config: Config):
    logger.info(f"Using device: {config.device}")
    alpha = config.alpha if config.noniid_type == "dirichlet" else "N/A"
    logger.info(f"Partitioning: {config.noniid_type} (alpha={alpha})")
    if "cuda" in config.device:
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")


def log_data_summary(clients: list[Client]):
    """各ユーザーのデータ数分布を出力"""
    counts = [c.num_samples for c in clients]
    logger.info("=== Client Data Distribution Summary ===")
    logger.info(f"Total Clients: {len(clients)}")
    logger.info(f"Total Samples: {sum(counts)}")
    logger.info(
        f"Stats: Mean={np.mean(counts):.1f}, Std={np.std(counts):.1f}, Min={np.min(counts)}, Max={np.max(counts)}"
    )
    # 上位5名の内訳を表示
    logger.info("Sample Clients (Writer ID: Total Samples):")
    for c in clients[:5]:
        logger.info(f"  - {c.writer_id}: {c.num_samples}")
    logger.info("========================================")


def sample_round_clients(
    clients: list[Client], server: Server, config: Config
) -> list[int]:
    """そのラウンドに参加するクライアントのインデックスを選ぶ"""
    if config.sampling == "global":
        return list(sample_clients(len(clients), config.frac))

    # 所属センターごとに frac ずつ抽出し、小さいクラスタのセンターにも毎ラウンド更新が入るようにする
    groups: dict[int | None, list[int]] = {}
    for i, client in enumerate(clients):
        groups.setdefault(server.assignments.get(client.writer_id), []).append(i)
    return [
        members[j]
        for members in groups.values()
        for j in sample_clients(len(members), config.frac)
    ]


def run_round(
    round_num: int,
    clients: list[Client],
    server: Server,
    config: Config,
    exp_logger: ExperimentLogger,
):
    start_time = time.time()

    sampled_indices = sample_round_clients(clients, server, config)

    updates: list[ClientUpdate] = []

    # 選択されたクライアントで学習
    for idx in tqdm(
        sampled_indices,
        desc=f"Round {round_num}/{config.rounds}",
        leave=False,
        ncols=80,
    ):
        client = clients[idx]
        if config.assignment == "distance":
            update = server.participate_by_distance(client)
        else:
            update = client.participate(server.models)
        updates.append(update)

    # 集約
    server.aggregate(updates)
    train_time = time.time() - start_time

    # 評価
    eval_start = time.time()
    eval_results = server.evaluate(clients)
    eval_time = time.time() - eval_start

    log_round(
        round_num, updates, eval_results, train_time, eval_time, config, exp_logger
    )


def run_local_round(
    round_num: int,
    clients: list[Client],
    models: list[torch.nn.Module],
    config: Config,
    exp_logger: ExperimentLogger,
):
    """
    Local only ベースラインの1ラウンド

    通信せずに各クライアントが自分のモデルを local_epochs エポック学習し、自分のテストデータで評価する。
    協調しない場合の下限として、frac によらず毎ラウンド全クライアントが学習する
    """
    start_time = time.time()
    updates = []
    for client, model in tqdm(
        zip(clients, models, strict=True),
        desc=f"Round {round_num}/{config.rounds}",
        total=len(clients),
        leave=False,
        ncols=80,
    ):
        loss = client.train(model, config.local_epochs)
        updates.append(
            ClientUpdate(client.writer_id, 0, 0, {}, model.state_dict(), loss)
        )
    train_time = time.time() - start_time

    eval_start = time.time()
    eval_results = [
        ClientEval(client.writer_id, 0, *client.evaluate(model))
        for client, model in zip(clients, models, strict=True)
        if client.test_loader is not None
    ]
    eval_time = time.time() - eval_start

    log_round(
        round_num, updates, eval_results, train_time, eval_time, config, exp_logger
    )


def log_round(
    round_num: int,
    updates: list[ClientUpdate],
    eval_results: list[ClientEval],
    train_time: float,
    eval_time: float,
    config: Config,
    exp_logger: ExperimentLogger,
):
    """1ラウンドの結果をコンソールとログファイルに出力する"""
    avg_train_loss = float(np.mean([u.train_loss for u in updates]))
    acc, val_loss = summarize(eval_results)
    center_counts = dict.fromkeys(range(config.num_centers), 0)
    for u in updates:
        center_counts[u.center] += 1

    counts_str = ", ".join(f"C{k}: {v}" for k, v in center_counts.items())
    logger.info(
        f"Round {round_num:02d} | Time: {train_time + eval_time:.2f}s | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2f}% | Macro: {macro_accuracy(eval_results):.2f}% | [{counts_str}]"
    )

    exp_logger.save_round_log(
        round_num,
        avg_train_loss,
        val_loss,
        acc,
        center_counts,
        train_time,
        eval_time,
        eval_results,
    )
    exp_logger.save_client_rounds(round_num, updates)
    exp_logger.save_client_evals(round_num, eval_results)


# --- メイン実行 ---
def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    config = Config.from_args()

    exp_logger = ExperimentLogger(config)
    try:
        # tqdm のプログレスバーとログ出力が混ざらないようにする
        with exp_logger, logging_redirect_tqdm():
            train(config, exp_logger)
    except Exception:  # noqa: BLE001
        # 例外の内容は ExperimentLogger が記録済み
        sys.exit(1)


def train(config: Config, exp_logger: ExperimentLogger):
    log_environment(config)
    set_seed(config.seed)

    # 1. データ準備
    datasets = NistDataManager(config).prepare_data()
    if not datasets:
        raise RuntimeError("No datasets created.")

    # 2. Client / Server 初期化
    clients = [Client(ds, config) for ds in datasets]
    log_data_summary(clients)

    if config.method == "local":
        train_local(clients, config, exp_logger)
        return

    server = Server(config)

    # 3. センターの初期化
    init_centers = None
    if config.pretrain_rounds > 0:
        centers, init_centers = initialize_centers(clients, config)
        server.load_centers(centers)
        if config.assignment == "distance":
            # K-Means のクラスタを初期の所属とする
            server.assignments = {
                c.writer_id: k for c, k in zip(clients, init_centers, strict=True)
            }
    exp_logger.save_clients(clients, init_centers)

    # 4. 学習ループ
    logger.info(
        f"Start Training with {len(clients)} writers. (Centers: {config.num_centers})"
    )
    for r in range(1, config.rounds + 1):
        run_round(r, clients, server, config, exp_logger)

    if config.save_models:
        exp_logger.save_models(server.models)
    logger.info("Training Finished.")


def train_local(clients: list[Client], config: Config, exp_logger: ExperimentLogger):
    """Local only ベースライン: 各クライアントが自分のモデルを自分のデータだけで学習する"""
    exp_logger.save_clients(clients)
    device = torch.device(config.device)
    models = [SimpleCNN(config.num_classes).to(device) for _ in clients]

    logger.info(f"Start local-only training with {len(clients)} writers.")
    for r in range(1, config.rounds + 1):
        run_local_round(r, clients, models, config, exp_logger)
    logger.info("Training Finished.")


if __name__ == "__main__":
    main()
