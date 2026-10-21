import copy
import logging

import torch

from mcfl.client import Client, ClientUpdate
from mcfl.config import Config
from mcfl.model import flatten_weights
from mcfl.server import Server

logger = logging.getLogger(__name__)


class CFLServer(Server):
    """
    Clustered Federated Learning (Sattler et al., 2020) のサーバー

    1つのクラスタ（モデル）から始め、各クラスタで FedAvg を行う。クラスタの更新の平均が小さい
    （mean_norm < eps1: FedAvg としては収束している）のに個々の更新が大きい（max_norm > eps2:
    一部のクライアントに合っていない）とき、更新のコサイン類似度でクラスタを2つに分割する。
    分割の判定と手順は公式実装に合わせ、num_centers をクラスタ数の上限とする。

    論文と同様、全クライアントが毎ラウンド参加する前提（分割に全メンバーの更新が必要）
    """

    def __init__(self, config: Config, clients: list[Client]):
        super().__init__(config)
        self.models = self.models[:1]
        self.assignments = {c.writer_id: 0 for c in clients}
        self.round_num = 0

    def aggregate(self, updates: list[ClientUpdate]):
        """各クラスタを FedAvg で更新し、分割の条件を満たすクラスタを2つに分ける"""
        self.round_num += 1
        old_models = [flatten_weights(m.state_dict()) for m in self.models]
        deltas = [flatten_weights(u.weights) - old_models[u.center] for u in updates]
        for u, d in zip(updates, deltas, strict=True):
            u.scores["Update Norm"] = torch.linalg.vector_norm(d).item()

        super().aggregate(updates)

        for center in range(len(self.models)):
            members = [i for i, u in enumerate(updates) if u.center == center]
            if not members:
                continue
            cluster_deltas = torch.stack([deltas[i] for i in members])
            mean_norm = torch.linalg.vector_norm(cluster_deltas.mean(dim=0)).item()
            max_norm = max(updates[i].scores["Update Norm"] for i in members)
            logger.info(
                f"CFL cluster {center} | Clients: {len(members)} | Mean norm: {mean_norm:.4f} | Max norm: {max_norm:.4f}"
            )

            if (
                self.round_num > self.cfg.cfl_warmup_rounds
                and mean_norm < self.cfg.cfl_eps1
                and max_norm > self.cfg.cfl_eps2
                and len(members) > 2
                and len(self.models) < self.cfg.num_centers
            ):
                self._split(center, [updates[i] for i in members], cluster_deltas)

    def _split(self, center: int, updates: list[ClientUpdate], deltas: torch.Tensor):
        """更新のコサイン類似度で complete linkage の階層クラスタリングを行い、2つに分ける"""
        unit = deltas / torch.linalg.vector_norm(deltas, dim=1, keepdim=True).clamp(
            min=1e-12
        )
        _, second = bipartition(unit @ unit.T)

        new_center = len(self.models)
        self.models.append(copy.deepcopy(self.models[center]))
        for i in second:
            self.assignments[updates[i].writer_id] = new_center
        logger.info(
            f"CFL split cluster {center} -> {center} ({len(updates) - len(second)} clients), {new_center} ({len(second)} clients)"
        )


def bipartition(similarity: torch.Tensor) -> tuple[list[int], list[int]]:
    """
    類似度行列から complete linkage の凝集型クラスタリングで2クラスタに分ける

    クラスタ間の類似度は、メンバー間の類似度の最小値とする（距離 = −類似度の complete linkage）
    """
    clusters = [[i] for i in range(len(similarity))]
    linkage = similarity.clone().double()
    linkage.fill_diagonal_(-torch.inf)
    while len(clusters) > 2:
        flat = int(linkage.argmax())
        i, j = divmod(flat, len(clusters))
        i, j = min(i, j), max(i, j)
        # 統合したクラスタと他のクラスタの類似度は、両者の類似度の小さいほう
        merged = torch.minimum(linkage[i], linkage[j])
        linkage[i], linkage[:, i] = merged, merged
        linkage[i, i] = -torch.inf
        keep = [k for k in range(len(clusters)) if k != j]
        linkage = linkage[keep][:, keep]
        clusters[i] += clusters.pop(j)
    return clusters[0], clusters[1]
