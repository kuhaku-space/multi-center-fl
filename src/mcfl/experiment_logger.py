import json
import logging
from dataclasses import asdict
from datetime import datetime
from itertools import count
from pathlib import Path

import pandas as pd

from mcfl.config import Config

logger = logging.getLogger(__name__)


# --- ロガーヘルパー ---
class ExperimentLogger:
    def __init__(self, config: Config, writer_ids: list[str]):
        run_name = f"{datetime.now():%Y%m%d_%H%M%S}_K{config.num_centers}_s{config.seed}"
        self.save_dir = _create_run_dir(Path(config.log_base_dir), run_name)
        self.run_id = self.save_dir.name

        self.config_file = self.save_dir / "config.json"
        self.log_csv = self.save_dir / "log.csv"
        self.allocation_csv = self.save_dir / "client_allocation.csv"
        # 列は全クライアントで固定（未参加のラウンドは空欄になる）
        self.allocation_columns = ["Round"] + sorted(writer_ids)

        with self.config_file.open("w") as f:
            json.dump(asdict(config), f, indent=4)

        logger.info(f"Experiment initialized. Logs at: {self.save_dir}")

    def save_round_log(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
        center_counts: dict[int, int],
        elapsed_time: float,
    ):
        record = {
            "Round": round_num,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Accuracy": accuracy,
            "Time": elapsed_time,
        }
        for k, v in center_counts.items():
            record[f"Center_{k}_Clients"] = v

        self._append_csv(self.log_csv, pd.DataFrame([record]))

    def save_allocations(self, round_num: int, allocations: dict[str, int]):
        """allocations: {"writerA": 0, ...}（そのラウンドに参加したクライアントの所属センター）"""
        record = {"Round": round_num, **allocations}
        df = pd.DataFrame([record]).reindex(columns=self.allocation_columns)
        self._append_csv(self.allocation_csv, df)

    @staticmethod
    def _append_csv(path: Path, df: pd.DataFrame):
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _create_run_dir(base_dir: Path, name: str) -> Path:
    """新しい実験ディレクトリを作る（同名があれば連番を付け、既存の実験に書き込まない）"""
    base_dir.mkdir(parents=True, exist_ok=True)
    for i in count():
        path = base_dir / (name if i == 0 else f"{name}_{i}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise AssertionError("unreachable")
