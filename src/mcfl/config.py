import argparse
import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # 基本設定
    # mcfl: マルチセンター連合学習、local: 各クライアントが自分のデータだけで学習する（比較用）、
    # cfl: Clustered Federated Learning（比較用。num_centers はクラスタ数の上限）、
    # oracle: 正解のグループ（cluster_shift）をセンターの所属に使う（比較用の上限）
    method: str = "mcfl"
    num_centers: int = 2
    rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 0.05
    frac: float = 0.1  # クライアントのサンプリング割合
    # サンプリング方法。global: 全体から frac を抽出、
    # center: 所属センターごとに frac ずつ抽出（distance モードのみ）
    sampling: str = "global"
    seed: int = 42  # 再現性のためのシード

    # データ設定
    max_writers: int = 100
    max_imgs_per_writer: int = 100
    data_root: str = "./data/nist/extracted"
    noniid_type: str = "writer"  # "writer" or "dirichlet"
    alpha: float = 0.5  # Dirichlet concentration parameter
    # 正解のあるクラスタを作る変換（none / rotation / label）と、そのグループ数
    cluster_shift: str = "none"
    num_groups: int = 4

    # EM algorithm設定 (Multi-Center FL paper)
    # E-step の割り当て方法。distance: 学習後の重みに最も近いセンター（論文の Eq. 8）、
    # loss: 学習前に損失が最小のセンター
    assignment: str = "distance"
    # ローカル学習の近接項 (μ/2)‖w − w_center‖² の係数（論文の Eq. 7 の λ）。0 で無効
    prox_mu: float = 0.0
    # Temperature for soft assignment (lower = harder assignment)。loss モードのログ用
    em_temperature: float = 1.0
    # センター初期化前の FedAvg 事前学習ラウンド数（0 ならランダム初期化）
    pretrain_rounds: int = 5
    # CFL の分割条件: ラウンド数 > warmup かつ 更新の平均のノルム < eps1 かつ 更新のノルムの最大 > eps2
    cfl_eps1: float = 0.4
    cfl_eps2: float = 1.6
    cfl_warmup_rounds: int = 20
    # 評価前に所属センターのモデルを各クライアントでローカル学習するエポック数（0 で無効）
    finetune_epochs: int = 0

    # 高速化設定
    num_workers: int = 0  # GPUなら0推奨
    pin_memory: bool = False  # データが既にGPUにある場合はFalse

    # システム設定
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    log_base_dir: str = "./logs"
    save_models: bool = False  # 学習完了時に各センターのモデルを保存する
    num_classes: int = 62

    @classmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(
            description="NIST Federated Learning Simulation"
        )
        d = cls()

        parser.add_argument(
            "--method",
            type=str,
            default=d.method,
            choices=["mcfl", "local", "cfl", "oracle"],
            help="Training method (mcfl: multi-center FL, local: each client trains alone, cfl: clustered FL with --num-centers as the max clusters, oracle: centers from the ground-truth groups of --cluster-shift)",
        )
        parser.add_argument(
            "--num-centers", type=int, default=d.num_centers, help="Number of centers"
        )
        parser.add_argument(
            "--rounds", type=int, default=d.rounds, help="Number of rounds"
        )
        parser.add_argument(
            "--local-epochs", type=int, default=d.local_epochs, help="Local epochs"
        )
        parser.add_argument(
            "--batch-size", type=int, default=d.batch_size, help="Batch size"
        )
        parser.add_argument("--lr", type=float, default=d.lr, help="Learning rate")
        parser.add_argument(
            "--frac",
            type=float,
            default=d.frac,
            help="Fraction of clients to sample per round",
        )
        parser.add_argument(
            "--sampling",
            type=str,
            default=d.sampling,
            choices=["global", "center"],
            help="Client sampling (global: frac of all clients, center: frac of each center's clients)",
        )
        parser.add_argument(
            "--seed", type=int, default=d.seed, help="Random seed for reproducibility"
        )
        parser.add_argument(
            "--max-writers", type=int, default=d.max_writers, help="Max writers"
        )
        parser.add_argument(
            "--max-imgs-per-writer",
            type=int,
            default=d.max_imgs_per_writer,
            help="Max images per writer",
        )
        parser.add_argument(
            "--data-root", type=str, default=d.data_root, help="Data root path"
        )
        # Note: Official FEMNIST uses NIST SD19 (by_class and by_write).
        # This implementation uses 'by_class' and extracts writer info from .mit files,
        # which yields the same natural writer partitioning.
        parser.add_argument(
            "--noniid-type",
            type=str,
            default=d.noniid_type,
            choices=["writer", "dirichlet"],
            help="Type of non-IID partitioning (writer: natural, dirichlet: synthetic)",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=d.alpha,
            help="Dirichlet concentration parameter (smaller alpha = more non-IID)",
        )
        parser.add_argument(
            "--cluster-shift",
            type=str,
            default=d.cluster_shift,
            choices=["none", "rotation", "label"],
            help="Split clients into groups with a group-specific shift to create ground-truth clusters",
        )
        parser.add_argument(
            "--num-groups",
            type=int,
            default=d.num_groups,
            help="Number of ground-truth groups for --cluster-shift",
        )
        parser.add_argument(
            "--assignment",
            type=str,
            default=d.assignment,
            choices=["distance", "loss"],
            help="E-step assignment (distance: nearest center in weight space, loss: lowest loss)",
        )
        parser.add_argument(
            "--prox-mu",
            type=float,
            default=d.prox_mu,
            help="Coefficient of the proximal term to the center in local training (0: off)",
        )
        parser.add_argument(
            "--pretrain-rounds",
            type=int,
            default=d.pretrain_rounds,
            help="FedAvg rounds before initializing centers by K-Means (0: random init)",
        )
        parser.add_argument(
            "--cfl-eps1",
            type=float,
            default=d.cfl_eps1,
            help="CFL: split a cluster when the norm of its mean update is below this",
        )
        parser.add_argument(
            "--cfl-eps2",
            type=float,
            default=d.cfl_eps2,
            help="CFL: split a cluster when the max norm of its client updates is above this",
        )
        parser.add_argument(
            "--cfl-warmup-rounds",
            type=int,
            default=d.cfl_warmup_rounds,
            help="CFL: do not split clusters during these first rounds",
        )
        parser.add_argument(
            "--finetune-epochs",
            type=int,
            default=d.finetune_epochs,
            help="Epochs to fine-tune the center model on each client before evaluation (0: off)",
        )
        parser.add_argument(
            "--num-workers", type=int, default=d.num_workers, help="Number of workers"
        )
        pin_group = parser.add_mutually_exclusive_group()
        pin_group.add_argument(
            "--pin-memory",
            action="store_true",
            default=d.pin_memory,
            help="Enable pin_memory (ignored when data is on GPU)",
        )
        pin_group.add_argument(
            "--no-pin-memory",
            action="store_false",
            dest="pin_memory",
            help="Disable pin_memory",
        )
        parser.add_argument(
            "--device", type=str, default=d.device, help="Device (cpu/cuda)"
        )
        parser.add_argument(
            "--log-base-dir", type=str, default=d.log_base_dir, help="Log directory"
        )
        parser.add_argument(
            "--save-models",
            action="store_true",
            default=d.save_models,
            help="Save each center's model to checkpoints/ after training",
        )

        config = cls(**vars(parser.parse_args()))

        if config.num_groups < 1:
            parser.error("--num-groups must be at least 1")
        if config.cluster_shift == "rotation" and config.num_groups > 4:
            parser.error("--cluster-shift rotation supports at most 4 groups")

        # local ではセンターを使わない。ログの集計に合わせて 1 つのセンターとして扱う
        if config.method == "local":
            config.num_centers = 1
            config.pretrain_rounds = 0

        # CFL は1つのクラスタからランダム初期化で始め、分割に全メンバーの更新を使う
        if config.method == "cfl":
            config.pretrain_rounds = 0
            if config.frac != 1.0:
                parser.error("--method cfl requires --frac 1.0")

        # oracle は正解のグループごとに1つのセンターを持ち、所属は固定する
        if config.method == "oracle":
            if config.cluster_shift == "none":
                parser.error("--method oracle requires --cluster-shift")
            config.num_centers = config.num_groups
            config.pretrain_rounds = 0

        # loss モードには所属センターがないため、センターごとに抽出できない
        if config.sampling == "center" and config.assignment != "distance":
            parser.error("--sampling center requires --assignment distance")

        # データはデバイス上に置かれるため、CUDA使用時の pin_memory はエラーになる。
        # また CUDA が使えない環境では pin_memory は意味がない。
        if config.pin_memory and (
            "cuda" in config.device or not torch.cuda.is_available()
        ):
            logger.warning("pin_memory forced to False.")
            config.pin_memory = False

        return config
