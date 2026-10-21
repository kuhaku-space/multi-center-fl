import logging
from pathlib import Path
from typing import override

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)

# 書き手ごとの (画像パス, ラベル)
WriterData = dict[str, tuple[list[str], list[int]]]

# writer 分割で採用する書き手の最低枚数（これより多い必要がある）
MIN_IMGS_WRITER = 10
# dirichlet 分割で採用する書き手の最低枚数
MIN_IMGS_DIRICHLET = 5


# --- Dataset 定義 ---
class NistWriterDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        writer_id: str,
        images: list[str],
        labels: list[int],
        transform: transforms.Compose | None = None,
        device: str = "cpu",
    ):
        self.writer_id = writer_id

        loaded_images = []
        loaded_labels = []
        for path, label in zip(images, labels):
            try:
                with Image.open(path) as img:
                    img = img.convert("L")
                    tensor = transform(img) if transform else img
                    if not isinstance(tensor, torch.Tensor):
                        tensor = transforms.ToTensor()(tensor)
            except (UnidentifiedImageError, OSError) as e:
                logger.warning(f"Error loading image {path}: {e}")
                continue
            loaded_images.append(tensor)
            loaded_labels.append(label)

        self.data: torch.Tensor
        self.targets: torch.Tensor
        if not loaded_images:
            logger.warning(f"Writer {writer_id} has no valid images.")
            self.data = torch.empty(0)
            self.targets = torch.empty(0)
            return

        # データをGPUメモリへ転送
        self.data = torch.stack(loaded_images).to(device)
        self.targets = torch.tensor(loaded_labels, dtype=torch.long).to(device)

    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


def _generate_hex_label_map() -> dict[str, int]:
    """by_class のディレクトリ名 (ASCIIコードの16進) → ラベル (0-9, A-Z, a-z の順)"""
    chars = (
        [chr(c) for c in range(ord("0"), ord("9") + 1)]
        + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    )
    return {f"{ord(ch):02x}": label for label, ch in enumerate(chars)}


def _extract_writer_id(orig_path: str) -> str | None:
    for comp in orig_path.split("/"):
        # "f"で始まり、数字が含まれるディレクトリ名をWriterIDとみなす
        if comp.startswith("f") and any(c.isdigit() for c in comp):
            return comp
    return None


# --- データ管理クラス ---
class NistDataManager:
    def __init__(self, config: Config):
        self.cfg = config
        self.transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def prepare_data(self) -> list[NistWriterDataset]:
        class_dir = Path(self.cfg.data_root) / "by_class"

        logger.info("--- Data Preparation Start ---")
        if not class_dir.exists():
            logger.error(f"Directory not found: {class_dir}")
            return []

        writers_data: WriterData = {}
        hex_to_label = _generate_hex_label_map()

        # .mit ファイルの解析
        for hex_code, label in tqdm(
            hex_to_label.items(), desc="Parsing classes", ncols=80
        ):
            # ディレクトリ検索（大文字小文字対応）
            for name in (hex_code, hex_code.upper()):
                target_dir = class_dir / name
                if target_dir.exists():
                    self._parse_mit_files_in_dir(target_dir, label, writers_data)
                    break

        logger.info(f"Found {len(writers_data)} unique writers.")

        # データ数が多い順にソート
        sorted_writers = sorted(
            writers_data.items(), key=lambda x: len(x[1][0]), reverse=True
        )
        if self.cfg.noniid_type == "dirichlet":
            return self._create_dirichlet_datasets(sorted_writers)
        return self._create_datasets(sorted_writers)

    @staticmethod
    def _parse_mit_files_in_dir(target_dir: Path, label: int, writers_data: WriterData):
        for mit_file in target_dir.rglob("*.mit"):
            try:
                lines = mit_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                # ファイル破損等はスキップ
                continue

            # ヘッダー行(0行目)スキップ、各行解析
            for line in lines.splitlines()[1:]:
                parts = line.split()
                if len(parts) < 2:
                    continue

                filename, orig_path = parts[0], parts[1]
                writer_id = _extract_writer_id(orig_path)
                if not writer_id:
                    continue

                # 画像パスの解決
                candidates = (
                    mit_file.parent / mit_file.stem / filename,
                    mit_file.parent / filename,
                )
                target_path = next((p for p in candidates if p.exists()), None)
                if target_path is None:
                    continue

                images, labels = writers_data.setdefault(writer_id, ([], []))
                images.append(str(target_path))
                labels.append(label)

    def _make_dataset(
        self, writer_id: str, images: list[str], labels: list[int]
    ) -> NistWriterDataset:
        return NistWriterDataset(
            writer_id, images, labels, self.transform, self.cfg.device
        )

    def _create_datasets(
        self, sorted_writers: list[tuple[str, tuple[list[str], list[int]]]]
    ) -> list[NistWriterDataset]:
        max_imgs = self.cfg.max_imgs_per_writer
        datasets = []

        for w_id, (images, labels) in sorted_writers:
            # データ数間引き
            if len(images) > max_imgs:
                indices = np.random.permutation(len(images))[:max_imgs]
                images = [images[i] for i in indices]
                labels = [labels[i] for i in indices]

            if len(images) > MIN_IMGS_WRITER:
                datasets.append(self._make_dataset(w_id, images, labels))
                if len(datasets) >= self.cfg.max_writers:
                    break

        logger.info(f"Prepared {len(datasets)} writer datasets.")
        return datasets

    def _create_dirichlet_datasets(
        self, sorted_writers: list[tuple[str, tuple[list[str], list[int]]]]
    ) -> list[NistWriterDataset]:
        """
        Dirichlet分布を用いたNon-IID分割。
        各クライアントには特定の1人のライターのデータのみを割り当て、
        その中でのクラス分布をDirichlet分布に基づいてサンプリング（間引き）することで不均衡を作ります。
        """
        selected_writers = sorted_writers[: self.cfg.max_writers]
        num_clients = len(selected_writers)
        num_classes = self.cfg.num_classes

        # クラスごとに、全クライアント（ライター）への配分比率をサンプリング
        # proportions[k][i] は クラスk を クライアントi にどれだけ割り当てるかの比率
        proportions = [
            np.random.dirichlet([self.cfg.alpha] * num_clients)
            for _ in range(num_classes)
        ]

        datasets = []
        for i, (w_id, (images, labels)) in enumerate(selected_writers):
            images_arr = np.array(images)
            labels_arr = np.array(labels)

            final_indices = []
            for k in range(num_classes):
                idx_k = np.where(labels_arr == k)[0]
                if len(idx_k) == 0:
                    continue

                # このライターが持つクラスkのうち、Dirichlet比率に基づいて採用する数を決定
                # p * num_clients の期待値は1。alphaが小さいと特定のクライアントに集中する。
                p = proportions[k][i]
                num_to_keep = min(len(idx_k), int(len(idx_k) * p * num_clients))

                # 完全に0にならないよう、比率がある程度あれば最低1つは残す
                if num_to_keep == 0 and p > 1.0 / (num_clients * 5):
                    num_to_keep = 1

                if num_to_keep > 0:
                    final_indices.extend(
                        np.random.choice(idx_k, num_to_keep, replace=False)
                    )

            if len(final_indices) < MIN_IMGS_DIRICHLET:
                continue

            # 最大枚数制限
            if len(final_indices) > self.cfg.max_imgs_per_writer:
                final_indices = np.random.choice(
                    final_indices, self.cfg.max_imgs_per_writer, replace=False
                )

            datasets.append(
                self._make_dataset(
                    w_id,
                    images_arr[final_indices].tolist(),
                    labels_arr[final_indices].tolist(),
                )
            )

        logger.info(
            f"Prepared {len(datasets)} Dirichlet-partitioned datasets (One writer per client)."
        )
        return datasets
