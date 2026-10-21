import json
import logging
import math
import platform
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Self

import pandas as pd
import torch
from torch import nn

from mcfl.client import Client, ClientUpdate
from mcfl.config import Config
from mcfl.server import ClientEval, macro_accuracy, summarize

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# --- ロガーヘルパー ---
class ExperimentLogger:
    """
    実験ごとのログディレクトリを管理する

    with 文の間はログを train.log にも保存し、抜けるときに終了状態
    (completed / failed / interrupted) を summary.json に書き出す
    """

    def __init__(self, config: Config):
        self.started_at = datetime.now()
        run_name = (
            f"{self.started_at:%Y%m%d_%H%M%S}_K{config.num_centers}_s{config.seed}"
        )
        if config.method != "mcfl":
            run_name += f"_{config.method}"
        self.save_dir = _create_run_dir(Path(config.log_base_dir), run_name)
        self.run_id = self.save_dir.name

        self.config_file = self.save_dir / "config.json"
        self.log_csv = self.save_dir / "log.csv"
        self.allocation_csv = self.save_dir / "client_allocation.csv"
        self.client_round_csv = self.save_dir / "client_round.csv"
        self.client_eval_csv = self.save_dir / "client_eval.csv"
        self.num_centers = config.num_centers
        self.allocation_columns = ["Round"]
        self.last_record: dict | None = None
        self.best_record: dict | None = None

        _write_json(self.config_file, asdict(config))
        _write_json(self.save_dir / "meta.json", self._collect_meta(config))

        # コンソールと同じ内容を train.log にも書き出す
        self._file_handler = logging.FileHandler(self.save_dir / "train.log")
        self._file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    def __enter__(self) -> Self:
        logging.getLogger().addHandler(self._file_handler)
        logger.info(f"Experiment initialized. Logs at: {self.save_dir}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            status = "completed"
        elif isinstance(exc, KeyboardInterrupt):
            status = "interrupted"
            logger.warning("Training interrupted.")
        else:
            status = "failed"
            logger.error("Training failed.", exc_info=exc)
        self._write_summary(status, exc if status == "failed" else None)

        logging.getLogger().removeHandler(self._file_handler)
        self._file_handler.close()

    def save_clients(
        self, clients: list[Client], init_centers: list[int] | None = None
    ):
        """各クライアントのデータ数（と K-Means による初期クラスタ）を保存する"""
        df = pd.DataFrame(
            {
                "Writer ID": [c.writer_id for c in clients],
                "Train Samples": [len(c.train_dataset) for c in clients],
                "Test Samples": [len(c.test_dataset) for c in clients],
            }
        )
        if init_centers is not None:
            df["Init Center"] = init_centers
        if clients and clients[0].group is not None:
            df["Group"] = [c.group for c in clients]
        df.to_csv(self.save_dir / "clients.csv", index=False)
        # 割り当ての列は全クライアントで固定（未参加のラウンドは空欄になる）
        self.allocation_columns = ["Round"] + sorted(df["Writer ID"])

    def save_round_log(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        accuracy: float,
        center_counts: dict[int, int],
        train_time: float,
        eval_time: float,
        eval_results: list[ClientEval],
    ):
        record = {
            "Round": round_num,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Accuracy": accuracy,
            "Macro Accuracy": macro_accuracy(eval_results),
            "Time": train_time + eval_time,
            "Train Time": train_time,
            "Eval Time": eval_time,
        }
        for k, v in center_counts.items():
            record[f"Center_{k}_Clients"] = v
        # 評価時にそのセンターが選ばれたクライアントだけで集計した指標
        for k in range(self.num_centers):
            acc_k, loss_k = summarize([r for r in eval_results if r.center == k])
            record[f"Center_{k}_Accuracy"] = acc_k
            record[f"Center_{k}_Val Loss"] = loss_k

        self.last_record = record
        if not math.isnan(accuracy) and (
            self.best_record is None or accuracy > self.best_record["Accuracy"]
        ):
            self.best_record = record

        self._append_csv(self.log_csv, pd.DataFrame([record]))

    def save_client_rounds(self, round_num: int, updates: list[ClientUpdate]):
        """そのラウンドに参加したクライアントの割り当て（とその根拠）と学習損失を保存する"""
        rows = [
            {
                "Round": round_num,
                "Writer ID": u.writer_id,
                "Start Center": u.start_center,
                "Center": u.center,
                **u.scores,
                "Train Loss": u.train_loss,
            }
            for u in updates
        ]
        self._append_csv(self.client_round_csv, pd.DataFrame(rows))

        # ダッシュボードのヒートマップ用（行: ラウンド、列: クライアント）
        allocations = {row["Writer ID"]: row["Center"] for row in rows}
        record = {"Round": round_num, **allocations}
        df = pd.DataFrame([record]).reindex(columns=self.allocation_columns)
        self._append_csv(self.allocation_csv, df)

    def save_client_evals(self, round_num: int, eval_results: list[ClientEval]):
        """クライアントごとの評価結果（評価に使ったセンター・精度・損失）を保存する"""
        if not eval_results:
            return
        df = pd.DataFrame(
            [
                {
                    "Round": round_num,
                    "Writer ID": r.writer_id,
                    "Center": r.center,
                    "Accuracy": 100 * r.correct / r.samples,
                    "Val Loss": r.loss_sum / r.samples,
                    "Test Samples": r.samples,
                }
                for r in eval_results
            ]
        )
        self._append_csv(self.client_eval_csv, df)

    def save_models(self, models: list[nn.Module]):
        """各センターのモデルの重みを checkpoints/center_<k>.pt に保存する"""
        checkpoint_dir = self.save_dir / "checkpoints"
        checkpoint_dir.mkdir()
        for k, model in enumerate(models):
            torch.save(model.state_dict(), checkpoint_dir / f"center_{k}.pt")
        logger.info(f"Saved {len(models)} center models to {checkpoint_dir}")

    def _write_summary(self, status: str, exc: BaseException | None):
        finished_at = datetime.now()
        summary = {
            "status": status,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - self.started_at).total_seconds(), 1),
            "last_round": self.last_record["Round"] if self.last_record else 0,
            "final": self.last_record,
            "best": self.best_record,
            "error": "".join(traceback.format_exception(exc)) if exc else None,
        }
        _write_json(self.save_dir / "summary.json", summary)

    def _collect_meta(self, config: Config) -> dict:
        """再現に必要な実行環境の情報を集める（未コミットの変更は git_diff.patch に保存）"""
        git_status = _git("status", "--porcelain")
        git_dirty = None if git_status is None else git_status != ""
        if git_dirty:
            (self.save_dir / "git_diff.patch").write_text(_git("diff", "HEAD") or "")

        return {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "command": sys.argv,
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(config.device)
                if "cuda" in config.device
                else None
            ),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": git_dirty,
        }

    @staticmethod
    def _append_csv(path: Path, df: pd.DataFrame):
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_json(path: Path, data: dict):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def _git(*args: str) -> str | None:
    """パッケージのあるリポジトリで git を実行する（git がない・リポジトリ外なら None）"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


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
