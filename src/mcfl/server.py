import torch
from torch import nn

from mcfl.client import Client
from mcfl.config import Config
from mcfl.model import SimpleCNN, compute_losses

# 無視できる寄与とみなす割り当て確率の閾値
MIN_WEIGHT = 1e-6


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
        M-step: 確率に基づく重み付き集約

        各クライアントの寄与は、そのクライアントのセンターへの割り当て確率で重み付けされる
        """
        for center_idx, model in enumerate(self.models):
            contributions = [
                (probs[center_idx], weights)
                for probs, weights, _ in updates
                if probs[center_idx] > MIN_WEIGHT
            ]
            if not contributions:
                continue

            total_weight = sum(w for w, _ in contributions)
            averaged = {
                key: sum(w * weights[key] for w, weights in contributions)
                / total_weight
                for key in contributions[0][1]
            }
            model.load_state_dict(averaged)

    def evaluate(self, clients: list[Client]) -> tuple[float, float]:
        """各クライアントのテストデータで、損失が最小のセンターを選んで評価する"""
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for client in clients:
            if client.test_loader is None:
                continue

            best_model = self._select_best_model(client.test_loader)
            if best_model is None:
                continue

            best_model.eval()
            with torch.no_grad():
                for x, y in client.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = best_model(x)
                    total_loss += self.criterion(out, y).item() * x.size(0)
                    total_correct += (out.argmax(dim=1) == y).sum().item()
                    total_samples += y.size(0)

        if total_samples == 0:
            return 0.0, 0.0
        return 100 * total_correct / total_samples, total_loss / total_samples

    def _select_best_model(self, loader) -> nn.Module | None:
        """1バッチの損失が最小のモデルを選ぶ（本来はValidation setを使うべきだが簡易的に）"""
        try:
            x, y = next(iter(loader))
        except StopIteration:
            return None

        x, y = x.to(self.device), y.to(self.device)
        losses = compute_losses(self.models, x, y, self.criterion)
        return self.models[losses.index(min(losses))]
