import torch
from torch import nn
from torch.utils.data import DataLoader

StateDict = dict[str, torch.Tensor]


# --- モデル定義 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: filters=32, kernel_size=[5, 5], padding="same"
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            # Pool1: pool_size=[2, 2], strides=2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv2: filters=64, kernel_size=[5, 5], padding="same"
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            # Pool2: pool_size=[2, 2], strides=2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dense層
        # 入力次元: 7 * 7 * 64 = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dense1: units=248, activation=relu
            nn.Linear(3136, 248),
            nn.ReLU(),
            # Logits: units=num_classes
            nn.Linear(248, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@torch.no_grad()
def compute_losses(
    models: list[nn.Module],
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> list[float]:
    """loader の全データに対する各モデルの平均損失を計算する"""
    loss_sums = [0.0] * len(models)
    samples = 0
    for model in models:
        model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        for k, model in enumerate(models):
            loss_sums[k] += criterion(model(x), y).item() * x.size(0)
        samples += x.size(0)
    return [s / samples for s in loss_sums]


def flatten_weights(weights: StateDict) -> torch.Tensor:
    """重みを1本のベクトル (CPU, float32) にする"""
    return torch.cat([w.detach().flatten().float().cpu() for w in weights.values()])


def average_weights(weights: list[StateDict]) -> StateDict:
    """複数モデルの重みの単純平均"""
    return {key: sum(w[key] for w in weights) / len(weights) for key in weights[0]}
