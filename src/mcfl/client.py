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
        avg_loss = self._train(local_model)
        return local_model.state_dict(), avg_loss

    def _train(self, model: nn.Module) -> float:
        """ローカル学習を行い、平均損失を返す"""
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr)

        total_loss = 0.0
        batch_count = 0
        for _ in range(self.cfg.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(model(x), y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

        return total_loss / batch_count if batch_count > 0 else 0.0

    def compute_center_probabilities(
        self, global_models: list[nn.Module]
    ) -> list[float]:
        """
        E-step: 全センターに対するソフト割り当て確率を計算

        softmax(-loss/τ) で確率を計算する。損失が最小のセンターを選ぶのは、論文が比較手法として
        挙げている HypoCluster / IFCA と同様の方法で、論文の手法（重み空間での距離）とは異なる

        Note:
            損失ベースの割り当てでは、多くのクライアントから更新を受けて学習が進んだセンターが
            どのクライアントに対しても損失が小さくなり、全クライアントを取り込みやすい。
            論文と同様の重み空間での距離による割り当ては Server.participate_by_distance を参照。
        """
        # 学習データの最初のバッチを使って判断する
        try:
            x, y = next(iter(self.train_loader))
        except StopIteration:
            # データがない場合は均等確率
            return [1.0 / len(global_models)] * len(global_models)

        x, y = x.to(self.device), y.to(self.device)
        losses = compute_losses(global_models, x, y, self.criterion)

        # p_k = exp(-loss_k / τ) / Σ_j exp(-loss_j / τ)
        neg_losses = (
            torch.tensor(losses, dtype=torch.float64) / -self.cfg.em_temperature
        )
        return torch.softmax(neg_losses, dim=0).tolist()
