import logging

import numpy as np
import torch
from tqdm import tqdm

from mcfl.client import Client, sample_clients
from mcfl.config import Config
from mcfl.model import SimpleCNN, StateDict, average_weights, flatten_weights

logger = logging.getLogger(__name__)

KMEANS_MAX_ITERS = 100


def initialize_centers(
    clients: list[Client], config: Config
) -> tuple[list[StateDict], list[int]]:
    """
    センターの初期値を作る（公式実装と同様の手順）

    1. 共通モデルを FedAvg で pretrain_rounds ラウンド事前学習する
    2. 全クライアントが事前学習モデルからローカル学習する
    3. 学習後の重み（事前学習モデルからの差分）を K-Means でクラスタリングし、
       各クラスタに属するクライアントの重みの平均をセンターの初期値とする

    ランダム初期化のままだと、損失ベースの割り当てでは少し学習が進んだセンターが
    全クライアントを取り込み、他のセンターが使われなくなる。

    Returns:
        tuple: (各センターの初期重み, 各クライアントの初期クラスタ番号)
    """
    model = SimpleCNN(config.num_classes).to(torch.device(config.device))

    # 1. FedAvg による事前学習
    for r in range(1, config.pretrain_rounds + 1):
        results = [
            clients[idx].train_from(model)
            for idx in tqdm(
                sample_clients(len(clients), config.frac),
                desc=f"Pretrain {r}/{config.pretrain_rounds}",
                leave=False,
                ncols=80,
            )
        ]
        model.load_state_dict(average_weights([w for w, _ in results]))
        avg_loss = np.mean([loss for _, loss in results])
        logger.info(f"Pretrain {r:02d} | Train: {avg_loss:.4f}")

    # 2. 全クライアントのローカル学習
    weights = [
        c.train_from(model)[0]
        for c in tqdm(clients, desc="Local training", leave=False, ncols=80)
    ]

    # 3. 重みの差分のクラスタリング
    base = flatten_weights(model.state_dict())
    deltas = torch.stack([flatten_weights(w) - base for w in weights])
    labels = kmeans(deltas, config.num_centers, config.seed)

    centers = []
    for k in range(config.num_centers):
        members = [w for w, label in zip(weights, labels) if label == k]
        # クライアント数がセンター数より少ないなどで空のクラスタは事前学習モデルのまま
        centers.append(average_weights(members) if members else model.state_dict())

    sizes = ", ".join(f"C{k}: {labels.count(k)}" for k in range(config.num_centers))
    logger.info(f"Initialized centers by K-Means. [{sizes}]")
    return centers, labels


def kmeans(x: torch.Tensor, k: int, seed: int) -> list[int]:
    """k-means++ で初期化した K-Means。x: (N, D) の各行のクラスタ番号を返す"""
    x = x.cpu()
    gen = torch.Generator().manual_seed(seed)

    # k-means++: 既存の中心から遠い点ほど選ばれやすくする
    centroids = x[torch.randint(len(x), (1,), generator=gen)]
    for _ in range(1, k):
        dist2 = _cdist(x, centroids).min(dim=1).values ** 2
        if dist2.sum() > 0:
            idx = torch.multinomial(dist2, 1, generator=gen)
        else:
            idx = torch.randint(len(x), (1,), generator=gen)
        centroids = torch.cat([centroids, x[idx]])

    labels = _cdist(x, centroids).argmin(dim=1)
    for _ in range(KMEANS_MAX_ITERS):
        # 空のクラスタは中心を動かさない
        centroids = torch.stack(
            [
                x[labels == j].mean(dim=0) if (labels == j).any() else centroids[j]
                for j in range(k)
            ]
        )
        new_labels = _cdist(x, centroids).argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    return labels.tolist()


def _cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 高次元では行列積による高速計算の誤差が大きいため、直接計算する
    return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
