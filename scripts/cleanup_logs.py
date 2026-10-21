import json
import shutil
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ログ保存場所
LOG_DIR = Path("./logs")


class ExperimentStatus(NamedTuple):
    is_complete: bool
    message: str
    current_round: int = 0
    total_rounds: int = 0


def get_experiment_status(exp_dir: Path) -> ExperimentStatus:
    """実験フォルダの状態を判定する"""
    config_path = exp_dir / "config.json"
    log_path = exp_dir / "log.csv"

    # 1. Configがない = 起動直後に落ちたゴミデータ
    if not config_path.exists():
        return ExperimentStatus(False, "Missing config.json")

    try:
        target_rounds = json.loads(config_path.read_text()).get("rounds", 0)
    except (OSError, json.JSONDecodeError) as e:
        return ExperimentStatus(False, f"Broken config: {e}")

    # 2. Logがない = 学習が始まる前に落ちた
    if not log_path.exists():
        return ExperimentStatus(False, "Missing log.csv", 0, target_rounds)

    try:
        df = pd.read_csv(log_path)
    except (OSError, pd.errors.ParserError) as e:
        return ExperimentStatus(False, f"Error reading log: {e}", 0, target_rounds)

    if df.empty or "Round" not in df.columns:
        return ExperimentStatus(False, "Empty or invalid log.csv", 0, target_rounds)

    # 3. 完了判定 (最終ラウンドが設定値に達しているか。Roundは1始まり)
    last_round = int(df["Round"].max())
    is_complete = last_round >= target_rounds
    message = "Complete" if is_complete else "Incomplete"
    return ExperimentStatus(is_complete, message, last_round, target_rounds)


def main():
    if not LOG_DIR.exists():
        print(f"Directory {LOG_DIR} does not exist.")
        return

    print(f"Checking experiments in {LOG_DIR}...\n")

    experiments = sorted(d for d in LOG_DIR.iterdir() if d.is_dir())
    deletion_candidates = []

    print(f"{'Experiment ID':<25} | {'Status':<25} | {'Progress':<10}")
    print("-" * 65)

    for exp_dir in experiments:
        status = get_experiment_status(exp_dir)
        progress = (
            f"{status.current_round}/{status.total_rounds}"
            if status.total_rounds > 0
            else "-"
        )
        mark = "✅" if status.is_complete else "❌"
        print(
            f"{exp_dir.name:<25} | {mark + ' ' + status.message:<25} | {progress:<10}"
        )

        if not status.is_complete:
            deletion_candidates.append(exp_dir)

    print("-" * 65)

    if not deletion_candidates:
        print("\n✨ No incomplete experiments found. Everything is clean!")
        return

    # --- 削除確認 ---
    print(f"\nFound {len(deletion_candidates)} incomplete experiments.")
    print("WARNING: These directories will be permanently deleted.")

    choice = input("Do you want to delete them? [y/N]: ").strip().lower()
    if choice != "y":
        print("\nOperation cancelled. No files were deleted.")
        return

    print("\nDeleting...")
    for d in deletion_candidates:
        try:
            shutil.rmtree(d)
            print(f"Deleted: {d.name}")
        except OSError as e:
            print(f"Failed to delete {d.name}: {e}")
    print("\nCleanup finished!")


if __name__ == "__main__":
    main()
