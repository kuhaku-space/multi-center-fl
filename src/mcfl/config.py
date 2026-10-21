import argparse
import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # 基本設定
    num_centers: int = 2
    rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 0.05
    frac: float = 0.1  # クライアントのサンプリング割合
    seed: int = 42  # 再現性のためのシード

    # データ設定
    max_writers: int = 100
    max_imgs_per_writer: int = 100
    data_root: str = "./data/nist/extracted"
    noniid_type: str = "writer"  # "writer" or "dirichlet"
    alpha: float = 0.5  # Dirichlet concentration parameter

    # EM algorithm設定 (Multi-Center FL paper)
    # Temperature for soft assignment (lower = harder assignment)
    em_temperature: float = 1.0
    # センター初期化前の FedAvg 事前学習ラウンド数（0 ならランダム初期化）
    pretrain_rounds: int = 5

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
            "--pretrain-rounds",
            type=int,
            default=d.pretrain_rounds,
            help="FedAvg rounds before initializing centers by K-Means (0: random init)",
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

        # データはデバイス上に置かれるため、CUDA使用時の pin_memory はエラーになる。
        # また CUDA が使えない環境では pin_memory は意味がない。
        if config.pin_memory and (
            "cuda" in config.device or not torch.cuda.is_available()
        ):
            logger.warning("pin_memory forced to False.")
            config.pin_memory = False

        return config
