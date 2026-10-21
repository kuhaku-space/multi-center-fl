import logging
import time

import numpy as np
import torch
from tqdm import tqdm

from client import Client
from config import Config
from dataset import NistDataManager
from logger import ExperimentLogger
from server import Server

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


def run_round(
    round_num: int,
    clients: list[Client],
    server: Server,
    config: Config,
    exp_logger: ExperimentLogger,
):
    start_time = time.time()

    # クライアントのランダムサンプリング
    num_sampled = max(int(config.frac * len(clients)), 1)
    sampled_indices = np.random.choice(range(len(clients)), num_sampled, replace=False)

    updates = []
    allocations: dict[str, int] = {}
    center_counts = dict.fromkeys(range(config.num_centers), 0)

    # 選択されたクライアントで学習
    for idx in tqdm(
        sampled_indices,
        desc=f"Round {round_num}/{config.rounds}",
        leave=False,
        ncols=80,
    ):
        client = clients[idx]
        probs, weights, loss = client.participate(server.models)
        updates.append((probs, weights, loss))

        # 最大確率のセンターを割り当てとして記録
        best_center = probs.index(max(probs))
        allocations[client.writer_id] = best_center
        center_counts[best_center] += 1

    # 集約・評価
    server.aggregate(updates)
    acc, val_loss = server.evaluate(clients)
    avg_train_loss = float(np.mean([loss for _, _, loss in updates]))

    elapsed_time = time.time() - start_time

    counts_str = ", ".join(f"C{k}: {v}" for k, v in center_counts.items())
    logger.info(
        f"Round {round_num:02d} | Time: {elapsed_time:.2f}s | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.2f}% | [{counts_str}]"
    )

    exp_logger.save_round_log(
        round_num, avg_train_loss, val_loss, acc, center_counts, elapsed_time
    )
    exp_logger.save_allocations(round_num, allocations)


# --- メイン実行 ---
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    config = Config.from_args()
    log_environment(config)
    set_seed(config.seed)

    # 1. データ準備
    datasets = NistDataManager(config).prepare_data()
    if not datasets:
        logger.error("No datasets created. Exiting.")
        return

    # 2. Client / Server 初期化
    clients = [Client(ds, config) for ds in datasets]
    server = Server(config)
    log_data_summary(clients)

    # 3. ログ管理初期化
    exp_logger = ExperimentLogger(config, [c.writer_id for c in clients])

    # 4. 学習ループ
    logger.info(
        f"Start Training with {len(clients)} writers. (Centers: {config.num_centers})"
    )
    for r in range(1, config.rounds + 1):
        run_round(r, clients, server, config, exp_logger)

    logger.info("Training Finished.")


if __name__ == "__main__":
    main()
