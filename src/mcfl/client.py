import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from mcfl.config import Config
from mcfl.dataset import NistWriterDataset
from mcfl.model import StateDict, compute_losses

TRAIN_RATIO = 0.8


@dataclass
class ClientUpdate:
    """1クライアントの1ラウンド分の学習結果"""

    writer_id: str
    start_center: int  # ローカル学習を始めたセンター
    center: int  # M-step で割り当てるセンター
    # 割り当ての根拠（ログ用）。loss: 割り当て確率 P_k と Entropy、distance: 距離 D_k
    scores: dict[str, float]
    weights: StateDict
    train_loss: float


def sample_clients(num_clients: int, frac: float) -> np.ndarray:
    """そのラウンドに参加するクライアントのインデックスをランダムに選ぶ"""
    num_sampled = max(int(frac * num_clients), 1)
    return np.random.choice(num_clients, num_sampled, replace=False)


# --- Client 定義 ---
class Client:
    def __init__(self, dataset: NistWriterDataset, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.writer_id = dataset.writer_id
        self.group = dataset.group

        # --- データの分割 (80% Train, 20% Test) ---
        total_len = len(dataset)
        train_len = int(total_len * TRAIN_RATIO) or total_len
        test_len = total_len - train_len

        self.train_dataset, self.test_dataset = random_split(
            dataset,
            [train_len, test_len],
            generator=torch.Generator().manual_seed(config.seed),
        )

        loader_kwargs = {
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
        }
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=min(config.batch_size, train_len),
            shuffle=True,
            **loader_kwargs,
        )
        self.test_loader = (
            DataLoader(
                self.test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                **loader_kwargs,
            )
            if test_len > 0
            else None
        )

    @property
    def num_samples(self) -> int:
        return len(self.train_dataset) + len(self.test_dataset)

    def participate(self, global_models: list[nn.Module]) -> ClientUpdate:
        """損失ベースの E-step: 割り当て確率を計算し、確率が最大のセンターから学習する"""
        probs = self.compute_center_probabilities(global_models)
        center = probs.index(max(probs))
        weights, avg_loss = self.train_from(global_models[center])

        scores = {f"P_{k}": p for k, p in enumerate(probs)}
        scores["Entropy"] = -sum(p * math.log(p) for p in probs if p > 0)
        return ClientUpdate(self.writer_id, center, center, scores, weights, avg_loss)

    def train_from(self, model: nn.Module) -> tuple[StateDict, float]:
        """model のコピーをローカル学習し、(更新後の重み, 平均損失) を返す"""
        local_model = copy.deepcopy(model)
        avg_loss = self.train(local_model, self.cfg.local_epochs)
        return local_model.state_dict(), avg_loss

    def train(
        self,
        model: nn.Module,
        epochs: int,
        prox_mu: float | None = None,
        anchor_model: nn.Module | None = None,
    ) -> float:
        """
        model をその場で epochs エポックローカル学習し、平均損失（近接項を除く）を返す

        prox_mu（省略時は設定の prox_mu）> 0 なら、anchor_model（省略時は学習開始時の重み
        = センター）からの距離の近接項 (μ/2)‖w − w_anchor‖² を損失に加える（論文の Eq. 7）
        """
        if prox_mu is None:
            prox_mu = self.cfg.prox_mu
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr)
        anchor = [p.detach().clone() for p in (anchor_model or model).parameters()]

        total_loss = 0.0
        batch_count = 0
        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(model(x), y)
                total_loss += loss.item()
                if prox_mu > 0:
                    prox = sum(
                        ((p - a) ** 2).sum()
                        for p, a in zip(model.parameters(), anchor, strict=True)
                    )
                    loss = loss + prox_mu / 2 * prox
                loss.backward()
                optimizer.step()

                batch_count += 1

        return total_loss / batch_count if batch_count > 0 else 0.0

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> tuple[int, int, float]:
        """テストデータで model を評価し、(正解数, サンプル数, 損失の合計) を返す"""
        model.eval()
        correct, samples, loss_sum = 0, 0, 0.0
        if self.test_loader is None:
            return correct, samples, loss_sum
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = model(x)
            loss_sum += self.criterion(out, y).item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            samples += y.size(0)
        return correct, samples, loss_sum

    def compute_center_probabilities(
        self, global_models: list[nn.Module]
    ) -> list[float]:
        """
        E-step: 全センターに対するソフト割り当て確率を計算

        学習データ全体に対する各センターの平均損失から softmax(-loss/τ) で確率を計算する。
        損失が最小のセンターを選ぶのは、論文が比較手法として挙げている HypoCluster / IFCA
        （IFCA と同じくローカルの学習データ全体で損失を測る）と同様の方法で、
        論文の手法（重み空間での距離）とは異なる

        Note:
            損失ベースの割り当てでは、多くのクライアントから更新を受けて学習が進んだセンターが
            どのクライアントに対しても損失が小さくなり、全クライアントを取り込みやすい。
            論文と同様の重み空間での距離による割り当ては Server.participate_by_distance を参照。
        """
        if len(self.train_dataset) == 0:
            # データがない場合は均等確率
            return [1.0 / len(global_models)] * len(global_models)

        losses = compute_losses(
            global_models, self.train_loader, self.criterion, self.device
        )

        # p_k = exp(-loss_k / τ) / Σ_j exp(-loss_j / τ)
        neg_losses = (
            torch.tensor(losses, dtype=torch.float64) / -self.cfg.em_temperature
        )
        return torch.softmax(neg_losses, dim=0).tolist()
