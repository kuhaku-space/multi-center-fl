import copy
import math

import torch
import torch.nn.functional as F
from torch import nn, optim

from mcfl.client import Client, ClientUpdate
from mcfl.config import Config
from mcfl.model import StateDict, average_weights
from mcfl.server import ClientEval, Server


class FedEMServer(Server):
    """
    FedEM (Marfoq et al., 2021) のサーバー

    各クライアントのデータ分布を num_centers 個の共有コンポーネント（モデル）の混合とみなし、
    クライアントごとの混合比 π を EM で推定する。公式実装と同様に、参加したクライアントは

    1. E-step: 学習データの各サンプルについて、コンポーネント m の負担率
       q_m(x, y) ∝ π_m exp(−loss_m(x, y)) を計算し、π を q のサンプル平均に更新する
    2. M-step: 各コンポーネントを、q_m で重み付けした損失で local_epochs エポック学習する

    を行い、サーバーはコンポーネントごとに参加クライアントの重みを平均する。
    予測は混合 Σ_m π_m p_m(y|x) で行う。ローカル学習に近接項 (prox_mu) は使わない
    """

    def __init__(self, config: Config, clients: list[Client]):
        super().__init__(config)
        num_components = config.num_centers
        self.mixtures = {
            c.writer_id: torch.full((num_components,), 1 / num_components)
            for c in clients
        }
        # 集約待ちの各コンポーネントの重み（クライアントごと）
        self._pending: dict[str, list[StateDict]] = {}

    def participate(self, client: Client) -> ClientUpdate:
        x, y = self._train_tensors(client)

        # E-step: サンプルごとの負担率と混合比
        with torch.no_grad():
            losses = torch.stack(
                [
                    F.cross_entropy(model.eval()(x), y, reduction="none")
                    for model in self.models
                ]
            ).cpu()
        log_pi = self.mixtures[client.writer_id].log()
        q = torch.softmax(log_pi[:, None] - losses, dim=0)
        pi = q.mean(dim=1)
        self.mixtures[client.writer_id] = pi

        # M-step: 負担率で重み付けした損失で各コンポーネントを学習
        weights, train_loss = [], 0.0
        for model, q_m in zip(self.models, q, strict=True):
            local_model = copy.deepcopy(model)
            train_loss += self._train_weighted(local_model, x, y, q_m.to(x.device))
            weights.append(local_model.state_dict())
        self._pending[client.writer_id] = weights

        probs = pi.tolist()
        center = probs.index(max(probs))
        scores = {f"P_{k}": p for k, p in enumerate(probs)}
        scores["Entropy"] = -sum(p * math.log(p) for p in probs if p > 0)
        return ClientUpdate(client.writer_id, center, center, scores, {}, train_loss)

    def aggregate(self, updates: list[ClientUpdate]):
        """各コンポーネントを、参加クライアントが学習したそのコンポーネントの重みの平均で更新する"""
        for m, model in enumerate(self.models):
            model.load_state_dict(
                average_weights([self._pending[u.writer_id][m] for u in updates])
            )
        self._pending.clear()

    @torch.no_grad()
    def evaluate(self, clients: list[Client]) -> list[ClientEval]:
        """
        各クライアントのテストデータを、混合比 π による混合モデルで評価する

        評価のセンターには π が最大のコンポーネントを記録する
        """
        for model in self.models:
            model.eval()

        results = []
        for client in clients:
            if client.test_loader is None:
                continue
            pi = self.mixtures[client.writer_id].to(self.device)
            correct, samples, loss_sum = 0, 0, 0.0
            for x, y in client.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                probs = sum(
                    p * torch.softmax(model(x), dim=1)
                    for p, model in zip(pi, self.models, strict=True)
                )
                log_probs = probs.clamp_min(1e-12).log()
                loss_sum += F.nll_loss(log_probs, y, reduction="sum").item()
                correct += (probs.argmax(dim=1) == y).sum().item()
                samples += y.size(0)
            center = int(pi.argmax())
            results.append(
                ClientEval(client.writer_id, center, correct, samples, loss_sum)
            )
        return results

    def _train_tensors(self, client: Client) -> tuple[torch.Tensor, torch.Tensor]:
        """負担率をサンプルに対応させるため、学習データを順序を固定したテンソルにする"""
        data = [client.train_dataset[i] for i in range(len(client.train_dataset))]
        x = torch.stack([d[0] for d in data]).to(self.device)
        y = torch.stack([d[1] for d in data]).to(self.device)
        return x, y

    def _train_weighted(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, q: torch.Tensor
    ) -> float:
        """サンプルを q で重み付けした損失で model をローカル学習し、平均損失を返す"""
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr)
        total_loss, batch_count = 0.0, 0
        for _ in range(self.cfg.local_epochs):
            for idx in torch.randperm(len(x)).split(self.cfg.batch_size):
                optimizer.zero_grad()
                loss_vec = F.cross_entropy(model(x[idx]), y[idx], reduction="none")
                # 公式実装と同様、重み付き損失の和をバッチサイズで割る
                loss = (loss_vec * q[idx]).sum() / len(idx)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
        return total_loss / batch_count if batch_count > 0 else 0.0
