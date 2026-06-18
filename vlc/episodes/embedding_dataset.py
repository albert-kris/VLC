"""Contrastive learning dataset for criterion-conditioned visual encoder.

Each __getitem__ returns one batch dict:
    images   : list[PIL.Image]   (n_classes * images_per_class images, shuffled)
    criterion: str               the instruction text
    labels   : list[int]         0-indexed super-class per image
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):

    def __init__(
        self,
        dataset_configs: list[dict],
        batches_per_epoch: int = 400,
        images_per_class: int = 8,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.images_per_class = images_per_class
        self.rng = random.Random(seed)

        self._sources: list[_DataSource] = []
        for cfg in dataset_configs:
            self._sources.append(_build_source(cfg, split, seed))

        self._cache: list[dict] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self._cache = [
            self.rng.choice(self._sources).sample_batch(self.images_per_class, self.rng)
            for _ in range(self.batches_per_epoch)
        ]

    def on_epoch_end(self) -> None:
        self._rebuild()

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx]


class _DataSource:
    def __init__(
        self,
        images_np: np.ndarray,      # (N, H, W, 3) uint8
        super_labels: np.ndarray,   # (N,) int
        criteria: dict[str, str],   # {key: instruction_text}
        image_size: int = 64,
    ) -> None:
        self._images_np = images_np
        self._super_labels = super_labels
        self._criteria_list = list(criteria.items())
        self._image_size = image_size

        k = int(super_labels.max()) + 1
        self._k = k
        self._cluster_idx = {
            sc: np.where(super_labels == sc)[0]
            for sc in range(k)
        }

    def sample_batch(self, images_per_class: int, rng: random.Random) -> dict:
        crit_key, crit_text = rng.choice(self._criteria_list)
        images, labels = [], []
        for sc in range(self._k):
            pool = self._cluster_idx[sc].tolist()
            n = min(images_per_class, len(pool))
            if n == 0:
                continue
            chosen = rng.sample(pool, n)
            for idx in chosen:
                arr = self._images_np[idx]
                img = Image.fromarray(arr).resize((self._image_size, self._image_size), Image.BILINEAR)
                images.append(img)
                labels.append(sc)

        combined = list(zip(images, labels))
        rng.shuffle(combined)
        if combined:
            images, labels = zip(*combined)
        return {"images": list(images), "criterion": crit_text, "labels": list(labels), "criterion_key": crit_key}


def _build_source(cfg: dict, split: str, seed: int) -> _DataSource:
    name = cfg["name"]
    image_size = cfg.get("image_size", 64)
    train_ratio = cfg.get("train_ratio", 0.9)
    rng = np.random.default_rng(seed)

    if name == "cifar10":
        return _cifar10_source(cfg, split, train_ratio, image_size, rng)
    if name == "cifar100":
        return _cifar100_source(cfg, split, train_ratio, image_size, rng)
    if name == "3dshapes":
        return _shapes3d_source(cfg, split, train_ratio, image_size, rng)
    raise ValueError(f"Unknown dataset: {name}")


def _cifar10_source(cfg, split, train_ratio, image_size, rng) -> _DataSource:
    from vlc.episodes.cifar10 import CRITERIA, CIFAR_TO_SUPER

    data_dir = Path(cfg["data_dir"])
    if split == "train":
        batch_files = [f"data_batch_{i}" for i in range(1, 6)]
    else:
        batch_files = ["test_batch"]

    imgs_list, labels_list = [], []
    for fname in batch_files:
        with open(data_dir / fname, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs_list.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels_list.extend(d[b"labels"])

    raw_imgs = np.concatenate(imgs_list, axis=0)
    cifar_labels = np.array(labels_list)

    if split == "train":
        n = len(raw_imgs)
        perm = rng.permutation(n)
        cut = int(n * train_ratio)
        idx = np.sort(perm[:cut])
        raw_imgs = raw_imgs[idx]
        cifar_labels = cifar_labels[idx]

    super_labels = np.array([CIFAR_TO_SUPER[int(l)] for l in cifar_labels])
    criteria = {k: v["instruction"] for k, v in CRITERIA.items()}
    return _DataSource(raw_imgs, super_labels, criteria, image_size)


def _cifar100_source(cfg, split, train_ratio, image_size, rng) -> _DataSource:
    from vlc.episodes.cifar100 import CRITERIA

    data_dir = Path(cfg["data_dir"])
    fname = "train" if split == "train" else "test"
    with open(data_dir / fname, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    raw_imgs = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    coarse_labels = np.array(d[b"coarse_labels"])

    if split == "train":
        n = len(raw_imgs)
        perm = rng.permutation(n)
        cut = int(n * train_ratio)
        idx = np.sort(perm[:cut])
        raw_imgs = raw_imgs[idx]
        coarse_labels = coarse_labels[idx]

    c2s = CRITERIA["by_kingdom"]["coarse_to_super"]
    super_labels = np.array([c2s[int(c)] for c in coarse_labels])
    criteria = {k: v["instruction"] for k, v in CRITERIA.items()}
    return _DataSource(raw_imgs, super_labels, criteria, image_size)


def _shapes3d_source(cfg, split, train_ratio, image_size, rng) -> _DataSource:
    import h5py

    h5_path = cfg["h5_path"]
    with h5py.File(h5_path, "r") as f:
        all_labels = f["labels"][:]
        n_total = len(all_labels)

    perm = rng.permutation(n_total)
    cut = int(n_total * train_ratio)
    idx = np.sort(perm[:cut] if split == "train" else perm[cut:])

    # Use floor_hue factor (index 0, 10 values) binned into 4 groups as super-class
    hue_vals = all_labels[idx, 0]
    bins = np.linspace(hue_vals.min(), hue_vals.max() + 1e-6, 5)
    super_labels = np.digitize(hue_vals, bins[1:]).astype(np.int64)

    with h5py.File(h5_path, "r") as f:
        images_np = f["images"][np.sort(idx)]

    # Reorder to match idx order
    sort_order = np.argsort(np.argsort(idx))
    images_np = images_np[sort_order]

    criteria = {
        "by_hue": "Group these images by the dominant color hue of the scene.",
        "by_shape": "Group these images by the shape of the 3D object shown.",
        "by_size": "Group these images by the relative size of the object in the scene.",
    }
    return _DataSource(images_np, super_labels, criteria, image_size)
