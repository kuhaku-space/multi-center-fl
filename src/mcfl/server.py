import math
from dataclasses import dataclass

import torch

from mcfl.client import Client, ClientUpdate
from mcfl.config import Config
from mcfl.model import (
    SimpleCNN,
    StateDict,
    average_weights,
    flatten_weights,
)


@dataclass
class ClientEval:
    """1クライアントのテストデータに対する評価結果"""

    writer_id: str
    center: int  # 評価に使ったセンター（損失が最小のもの）
    correct: int
    samples: int
    loss_sum: float


def summarize(results: list[ClientEval]) -> tuple[float, float]:
    """評価結果をサンプル数で重み付けして集計し、(精度[%], 平均損失) を返す（対象がなければ NaN）"""
    total_samples = sum(r.samples for r in results)
    if total_samples == 0:
        return math.nan, math.nan
    total_correct = sum(r.correct for r in results)
    total_loss = sum(r.loss_sum for r in results)
    return 100 * total_correct / total_samples, total_loss / total_samples


def macro_accuracy(results: list[ClientEval]) -> float:
    """クライアントごとの精度[%]の単純平均（論文の macro accuracy）。対象がなければ NaN"""
    if not results:
        return math.nan
    return sum(100 * r.correct / r.samples for r in results) / len(results)


# --- Server 定義 ---
class Server:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.models = [
            SimpleCNN(config.num_classes).to(self.device)
            for _ in range(config.num_centers)
        ]
        # クライアントの所属センター（distance モードで更新する）。評価もこの所属で行う
        self.assignments: dict[str, int] = {}

    def load_centers(self, centers: list[StateDict]):
        for model, weights in zip(self.models, centers, strict=True):
            model.load_state_dict(weights)

    def participate_by_distance(self, client: Client) -> ClientUpdate:
        """
        距離ベースの E-step（論文の Eq. 8）

        クライアントは所属センターからローカル学習し、学習後の重みに最も近いセンターに割り当て直す。
        所属がまだないクライアント（ランダム初期化時）は、損失が最小のセンターから始める。
        """
        start = self.assignments.get(client.writer_id)
        if start is None:
            probs = client.compute_center_probabilities(self.models)
            start = probs.index(max(probs))
        weights, avg_loss = client.train_from(self.models[start])

        flat = flatten_weights(weights)
        distances = [
            torch.linalg.vector_norm(flat - flatten_weights(m.state_dict())).item()
            for m in self.models
        ]
        center = distances.index(min(distances))
        self.assignments[client.writer_id] = center

        scores = {f"D_{k}": d for k, d in enumerate(distances)}
        return ClientUpdate(client.writer_id, start, center, scores, weights, avg_loss)

    def aggregate(self, updates: list[ClientUpdate]):
        """
        M-step: 各センターを、そのセンターに割り当てられたクライアントの重みの平均で更新する

        割り当てられたクライアントがいないセンターは前のモデルを維持する。

        Note:
            確率で重み付けして全センターに混ぜると、確率が一様に近いときに全センターが
            同じ平均を受け取り、数ラウンドで同一のモデルに収束してしまう。
        """
        for center_idx, model in enumerate(self.models):
            assigned = [u.weights for u in updates if u.center == center_idx]
            if assigned:
                model.load_state_dict(average_weights(assigned))

    def evaluate(self, clients: list[Client]) -> list[ClientEval]:
        """
        各クライアントのテストデータを、所属センターのモデルで評価する

        所属がないクライアント（loss モードでは全員）は、E-step と同じく学習データの
        先頭バッチで損失が最小のセンターを選ぶ（テストデータは選択に使わない）
        """
        results = []
        for client in clients:
            if client.test_loader is None:
                continue

            center = self.assignments.get(client.writer_id)
            if center is None:
                probs = client.compute_center_probabilities(self.models)
                center = probs.index(max(probs))

            correct, samples, loss_sum = client.evaluate(self.models[center])
            results.append(
                ClientEval(client.writer_id, center, correct, samples, loss_sum)
            )

        return results
