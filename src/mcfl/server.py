import math
from dataclasses import dataclass

import torch
from torch import nn

from mcfl.client import Client
from mcfl.config import Config
from mcfl.model import SimpleCNN, compute_losses


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


# --- Server 定義 ---
class Server:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.models = [
            SimpleCNN(config.num_classes).to(self.device)
            for _ in range(config.num_centers)
        ]
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(
        self, updates: list[tuple[list[float], dict[str, torch.Tensor], float]]
    ):
        """
        M-step: 各センターを、そのセンターに割り当てられたクライアントの重みの平均で更新する

        割り当ては確率が最大のセンター（= クライアントが学習に使ったセンター）とする hard な割り当て。
        割り当てられたクライアントがいないセンターは前のモデルを維持する。

        Note:
            確率で重み付けして全センターに混ぜると、確率が一様に近いときに全センターが
            同じ平均を受け取り、数ラウンドで同一のモデルに収束してしまう。
        """
        for center_idx, model in enumerate(self.models):
            assigned = [
                weights
                for probs, weights, _ in updates
                if probs.index(max(probs)) == center_idx
            ]
            if not assigned:
                continue

            averaged = {
                key: sum(weights[key] for weights in assigned) / len(assigned)
                for key in assigned[0]
            }
            model.load_state_dict(averaged)

    def evaluate(self, clients: list[Client]) -> list[ClientEval]:
        """各クライアントのテストデータで、損失が最小のセンターを選んで評価する"""
        results = []
        for client in clients:
            if client.test_loader is None:
                continue

            center = self._select_best_center(client.test_loader)
            if center is None:
                continue

            model = self.models[center]
            model.eval()
            correct, samples, loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for x, y in client.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = model(x)
                    loss_sum += self.criterion(out, y).item() * x.size(0)
                    correct += (out.argmax(dim=1) == y).sum().item()
                    samples += y.size(0)
            results.append(
                ClientEval(client.writer_id, center, correct, samples, loss_sum)
            )

        return results

    def _select_best_center(self, loader) -> int | None:
        """1バッチの損失が最小のセンターを選ぶ（本来はValidation setを使うべきだが簡易的に）"""
        try:
            x, y = next(iter(loader))
        except StopIteration:
            return None

        x, y = x.to(self.device), y.to(self.device)
        losses = compute_losses(self.models, x, y, self.criterion)
        return losses.index(min(losses))
