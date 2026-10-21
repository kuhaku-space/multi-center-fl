import argparse
import copy
import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

# RTX 3080 (Ampere) 以降で有効な行列演算の高速化
torch.set_float32_matmul_precision("high")

StateDict = dict[str, torch.Tensor]

# ==========================================
# 1. モデル定義 (論文 Source: 192, 193)
# ==========================================


class MNIST_2NN(nn.Module):
    """
    単純な多層パーセプトロン (2NN)
    2 hidden layers with 200 units each using ReLu activations
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class MNIST_CNN(nn.Module):
    """
    CNN
    Two 5x5 convolution layers (32, 64 channels), 2x2 max pooling,
    fully connected layer with 512 units and ReLu, softmax output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


MODELS: dict[str, type[nn.Module]] = {"mlp": MNIST_2NN, "cnn": MNIST_CNN}

# ==========================================
# 2. データ分割ヘルパー (論文 Source: 195)
# ==========================================


def get_mnist() -> tuple[datasets.MNIST, datasets.MNIST]:
    """MNISTデータのダウンロードと変換"""
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset_train = datasets.MNIST(
        "./data/", train=True, download=True, transform=trans
    )
    dataset_test = datasets.MNIST(
        "./data/", train=False, download=True, transform=trans
    )
    return dataset_train, dataset_test


def dataset_to_device(dataset: datasets.MNIST, device: torch.device) -> TensorDataset:
    """データセット全体をGPUメモリに転送する"""
    # DataLoaderを使って変換済みの全データを取得する
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, targets = next(iter(loader))
    return TensorDataset(data.to(device), targets.to(device))


def mnist_iid(dataset: datasets.MNIST, num_users: int) -> dict[int, set[int]]:
    """
    IID: データをシャッフルして各ユーザーに均等に分配
    """
    num_items = len(dataset) // num_users
    dict_users: dict[int, set[int]] = {}
    all_idxs = list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset: datasets.MNIST, num_users: int) -> dict[int, np.ndarray]:
    """
    Non-IID: ラベルでソートし、200個のシャードに分割。
    各ユーザーに2つのシャードを割り当てる（結果として各ユーザーは少数の数字クラスしか持たない）。
    Source: 195
    """
    num_shards, num_imgs = 200, 300
    idx_shard = list(range(num_shards))
    dict_users: dict[int, np.ndarray] = {
        i: np.array([], dtype="int64") for i in range(num_users)
    }
    idxs = np.arange(num_shards * num_imgs)
    labels: np.ndarray = dataset.targets.numpy()

    # ラベルでソート
    idxs_labels = np.vstack((idxs, labels))
    idxs = idxs_labels[:, idxs_labels[1, :].argsort()][0, :]

    # 各ユーザーに2つのシャードを割り当て
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


# ==========================================
# 3. Local Update (Client Side) (論文 Source: 184)
# ==========================================


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument("--epochs", type=int, default=50, help="rounds of training")
    parser.add_argument("--num-users", type=int, default=100, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local-ep", type=int, default=5, help="the number of local epochs: E"
    )
    parser.add_argument("--local-bs", type=int, default=10, help="local batch size: B")
    parser.add_argument("--bs", type=int, default=128, help="test batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="exponential",
        choices=["constant", "exponential", "step", "cosine"],
        help="learning rate scheduler type",
    )
    parser.add_argument(
        "--lr-step-size", type=int, default=20, help="step size for step scheduler"
    )
    parser.add_argument(
        "--lr-decay", type=float, default=0.99, help="learning rate decay per round"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )

    # model arguments
    parser.add_argument(
        "--model", type=str, default="mlp", choices=list(MODELS), help="model name"
    )
    parser.add_argument("--dataset", type=str, default="mnist", help="name of dataset")
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0 if torch.cuda.is_available() else -1,
        help="GPU ID, -1 for CPU",
    )
    return parser.parse_args()


def learning_rate_at(args: argparse.Namespace, round_idx: int) -> float:
    """ラウンド round_idx (0始まり) で使う学習率"""
    base_lr = args.lr
    if args.lr_scheduler == "exponential":
        return base_lr * (args.lr_decay**round_idx)
    if args.lr_scheduler == "step":
        return base_lr * (args.lr_decay ** (round_idx // args.lr_step_size))
    if args.lr_scheduler == "cosine":
        return 0.5 * base_lr * (1 + np.cos(np.pi * round_idx / args.epochs))
    return base_lr


class LocalUpdate:
    def __init__(
        self,
        args: argparse.Namespace,
        dataset: Dataset,
        idxs: set[int] | np.ndarray,
    ) -> None:
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # Subsetはindicesとしてsequenceを受け取るためlist化
        # データセットが既にGPUにある場合、num_workers=0 (デフォルト) が最も高速です
        self.ldr_train: DataLoader = DataLoader(
            Subset(dataset, list(idxs)),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=0,
        )

    def train(self, net: nn.Module, lr: float) -> tuple[StateDict, float]:
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=lr, momentum=self.args.momentum
        )

        epoch_loss: list[float] = []
        for _ in range(self.args.local_ep):
            batch_loss: list[float] = []
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self.loss_func(net(images), labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# ==========================================
# 4. Federated Averaging (Server Side) (論文 Source: 184)
# ==========================================


def fed_avg(w: list[StateDict]) -> StateDict:
    """
    重みの平均化 (Aggregation)
    w: list of client state_dicts
    """
    return {k: torch.stack([client_w[k] for client_w in w], 0).mean(0) for k in w[0]}


def test_img(
    net_g: nn.Module, datatest: Dataset, args: argparse.Namespace
) -> tuple[float, float]:
    """グローバルモデルの評価"""
    net_g.eval()
    test_loss = 0.0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)

    with torch.no_grad():  # 推論時は勾配計算不要
        for data, target in data_loader:
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target, reduction="sum").item()
            correct += (log_probs.argmax(dim=1) == target).sum().item()

    num_samples = len(data_loader.dataset)  # type: ignore[arg-type]
    return 100.0 * correct / num_samples, test_loss / num_samples


# ==========================================
# 5. メイン実行スクリプト
# ==========================================


def main() -> None:
    args = args_parser()
    use_cuda = torch.cuda.is_available() and args.gpu != -1
    args.device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {args.device}")
    print(f"Dataset: {args.dataset}")
    print(f"IID: {args.iid}")
    if use_cuda:
        print(f"GPU Name: {torch.cuda.get_device_name(args.device)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # データのロード
    mnist_train, mnist_test = get_mnist()

    # データの分割
    split = mnist_iid if args.iid else mnist_noniid
    dict_users = split(mnist_train, args.num_users)

    # データセットをGPUへ転送
    dataset_train: Dataset = mnist_train
    dataset_test: Dataset = mnist_test
    if args.gpu != -1:
        dataset_train = dataset_to_device(mnist_train, args.device)
        dataset_test = dataset_to_device(mnist_test, args.device)

    # モデルの初期化
    net_glob: nn.Module = MODELS[args.model]().to(args.device)
    print(net_glob)

    # ローカル学習用のモデルを1つ作成して使い回す
    net_local = copy.deepcopy(net_glob)

    # モデルのコンパイル (PyTorch 2.0+)
    # castを使用して型チェッカーのエラーを回避し、学習用・評価用両方を最適化する
    net_glob = cast(nn.Module, torch.compile(net_glob))
    net_local = cast(nn.Module, torch.compile(net_local))

    # グローバルモデルの重みをコピー (コンパイル後に行うことでキーを一致させる)
    w_glob = net_glob.state_dict()

    acc_test_history: list[float] = []

    # クライアントオブジェクトを事前に作成してDataLoaderの生成コストを抑える
    local_updates = [
        LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[i])
        for i in range(args.num_users)
    ]
    # クライアントの選択数 (m = max(C * K, 1))
    m = max(int(args.frac * args.num_users), 1)

    # 通信ラウンドのループ
    for round_idx in range(args.epochs):
        start_time = time.time()
        lr = learning_rate_at(args, round_idx)

        w_locals: list[StateDict] = []
        loss_locals: list[float] = []

        # 選択された各クライアントでローカル学習
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            net_local.load_state_dict(w_glob)
            w, loss = local_updates[idx].train(net=net_local, lr=lr)
            # 重みのコピーを高速化
            w_locals.append({k: v.clone() for k, v in w.items()})
            loss_locals.append(loss)

        # 重みの集約 (FedAvg) とグローバルモデルの更新
        w_glob = fed_avg(w_locals)
        net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_test, _ = test_img(net_glob, dataset_test, args)
        acc_test_history.append(acc_test)

        round_time = time.time() - start_time
        print(
            f"Round {round_idx:3d}, LR: {lr:.6f}, Average Loss: {loss_avg:.3f}, Test Accuracy: {acc_test:.2f}%, Time: {round_time:.2f}s"
        )

    # 結果のプロット
    plt.figure()
    plt.plot(range(len(acc_test_history)), acc_test_history)
    plt.ylabel("Test Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig("fedavg_result.png")
    print("Saved plot to fedavg_result.png")


if __name__ == "__main__":
    main()
