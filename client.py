import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from config import Config
from dataset import NistWriterDataset
from model import compute_losses

TRAIN_RATIO = 0.8


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

    def participate(
        self, global_models: list[nn.Module]
    ) -> tuple[list[float], dict[str, torch.Tensor], float]:
        """
        E-step: ソフト割り当て確率を計算し、最良モデルで学習

        Returns:
            tuple: (確率リスト, 更新後の重み, 平均損失)
        """
        probs = self._compute_center_probabilities(global_models)
        best_idx = probs.index(max(probs))
        model = copy.deepcopy(global_models[best_idx])
        avg_loss = self._train(model)
        return probs, model.state_dict(), avg_loss

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

    def _compute_center_probabilities(
        self, global_models: list[nn.Module]
    ) -> list[float]:
        """
        E-step: 全センターに対するソフト割り当て確率を計算

        論文のアルゴリズムに基づき、softmax(-loss/τ) で確率を計算

        Note:
            公式実装 (https://github.com/caifederated/multi-center-fed-learning) では
            クライアントのモデル重みに対するK-Meansクラスタリングが使用されていますが、
            本実装では論文(ArXiv:2005.01026)の記述通り、損失値に基づくEMアルゴリズムを採用しています。
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
