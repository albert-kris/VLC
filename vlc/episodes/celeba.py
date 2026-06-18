"""CelebA episode synthesizer.

CelebA has 40 binary attributes per image. We select a subset for multi-criteria
clustering training:
  - hair_color: {black, blond, brown, gray} → K=4
  - eyeglasses: {with, without} → K=2
  - smiling:    {smiling, not_smiling} → K=2
  - young:      {young, old} → K=2
  - gender:     {male, female} → K=2

CelebA data layout:
  data/celeba/img_align_celeba/*.jpg
  data/celeba/list_attr_celeba.txt   (image_name + 40 binary attributes, -1/+1)

Attribute indices (0-indexed in list_attr_celeba.txt, column order):
  5 = Black_Hair, 9 = Blond_Hair, 11 = Brown_Hair, 17 = Gray_Hair
  15 = Eyeglasses, 31 = Smiling, 39 = Young, 20 = Male
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from vlc.episodes.base import (
    ClusterCard,
    ClusterEpisode,
    EpisodeStep,
    build_initial_cards,
    split_into_batches,
    update_cards_from_assignments,
)


# Attribute column names from CelebA (space-separated header row)
CELEBA_ATTR_NAMES = [
    "5o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]

CRITERIA: dict[str, dict] = {
    "hair_color": {
        "instruction": "cluster these images by hair color",
        "type": "multi",
        "attr_indices": [8, 9, 11, 17],  # Black, Blond, Brown, Gray
        "attr_names": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"],
        "k": 4,
        "card_names": {
            1: ("black hair", "People with black or dark hair"),
            2: ("blond hair", "People with blond or light hair"),
            3: ("brown hair", "People with brown or dark brown hair"),
            4: ("gray hair", "People with gray or silver hair"),
        },
    },
    "eyeglasses": {
        "instruction": "cluster these images by whether the person wears glasses",
        "type": "binary",
        "attr_index": 15,
        "k": 2,
        "card_names": {
            1: ("with glasses", "People wearing eyeglasses or sunglasses"),
            2: ("without glasses", "People not wearing glasses"),
        },
    },
    "smiling": {
        "instruction": "cluster these images by facial expression",
        "type": "binary",
        "attr_index": 31,
        "k": 2,
        "card_names": {
            1: ("smiling", "People with a smile or happy expression"),
            2: ("not smiling", "People with a neutral or serious expression"),
        },
    },
    "young": {
        "instruction": "cluster these images by apparent age",
        "type": "binary",
        "attr_index": 39,
        "k": 2,
        "card_names": {
            1: ("younger appearance", "People who appear young"),
            2: ("older appearance", "People who appear older"),
        },
    },
    "gender": {
        "instruction": "cluster these images by gender presentation",
        "type": "binary",
        "attr_index": 20,
        "k": 2,
        "card_names": {
            1: ("masculine presentation", "People presenting as male"),
            2: ("feminine presentation", "People presenting as female"),
        },
    },
}


def _load_celeba_attrs(root: Path) -> tuple[list[str], np.ndarray]:
    """Load CelebA attribute file. Returns (image_names, attrs array [-1/+1])."""
    attr_path = root / "list_attr_celeba.txt"
    if not attr_path.exists():
        raise FileNotFoundError(f"CelebA attributes not found: {attr_path}")

    with open(attr_path, encoding="utf-8") as f:
        n = int(f.readline().strip())
        f.readline()  # skip header
        names = []
        rows = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            names.append(parts[0])
            rows.append([int(x) for x in parts[1:]])

    return names, np.array(rows, dtype=np.int8)


class CelebAEpisodeBuilder:
    """Builds streaming clustering episodes from CelebA face images."""

    def __init__(
        self,
        root: str | Path,
        n_images: int = 32,
        batch_size: int = 8,
        seed: int = 42,
        split: str = "train",
        train_ratio: float = 0.9,
        use_paraphrase: bool = True,
        image_size: int = 128,
    ) -> None:
        self.root = Path(root)
        self.n_images = n_images
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.use_paraphrase = use_paraphrase
        self.image_size = image_size

        img_dir = self.root / "img_align_celeba"
        if not img_dir.exists():
            raise FileNotFoundError(f"CelebA images not found: {img_dir}")

        all_names, all_attrs = _load_celeba_attrs(self.root)

        n_total = len(all_names)
        perm = self.np_rng.permutation(n_total)
        split_point = int(n_total * train_ratio)
        if split == "train":
            idxs = np.sort(perm[:split_point])
        else:
            idxs = np.sort(perm[split_point:])

        self._img_paths = [img_dir / all_names[i] for i in idxs]
        self._attrs = all_attrs[idxs]  # (N, 40)

        # Build cluster index per criterion
        self._cluster_index: dict[str, dict[int, list[int]]] = {}
        for cname, meta in CRITERIA.items():
            if meta["type"] == "binary":
                col = meta["attr_index"]
                vals = self._attrs[:, col]
                # +1 → cluster 1, -1 → cluster 2
                c1 = np.where(vals == 1)[0].tolist()
                c2 = np.where(vals == -1)[0].tolist()
                self._cluster_index[cname] = {1: c1, 2: c2}
            else:
                # multi: use first matching attribute column
                cols = meta["attr_indices"]
                cluster_map: dict[int, list[int]] = {i + 1: [] for i in range(len(cols))}
                for idx in range(len(self._attrs)):
                    matched = False
                    for ci, col in enumerate(cols):
                        if self._attrs[idx, col] == 1:
                            cluster_map[ci + 1].append(idx)
                            matched = True
                            break
                self._cluster_index[cname] = cluster_map

    def build_episode(self, criterion: str | None = None) -> ClusterEpisode:
        if criterion is None:
            criterion = self.rng.choice(list(CRITERIA.keys()))

        meta = CRITERIA[criterion]
        k = meta["k"]
        cluster_map = self._cluster_index[criterion]
        templates = meta["card_names"]

        per_cluster = self.n_images // k
        sampled: list[tuple[int, int]] = []
        for cid in range(1, k + 1):
            pool = cluster_map.get(cid, [])
            n = min(per_cluster, len(pool))
            chosen = self.rng.sample(pool, n)
            sampled.extend((idx, cid) for idx in chosen)

        self.rng.shuffle(sampled)
        if not sampled:
            from vlc.episodes.shapes3d import Shapes3DEpisodeBuilder
            raise RuntimeError(f"No samples for CelebA criterion '{criterion}'")

        indices, one_labels = zip(*sampled)
        indices, one_labels = list(indices), list(one_labels)

        images = []
        for i in indices:
            img = Image.open(self._img_paths[i]).convert("RGB")
            if self.image_size:
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            images.append(img)

        instruction = meta["instruction"]
        if self.use_paraphrase:
            from vlc.episodes.paraphrase import sample_criterion
            instruction = sample_criterion(instruction, self.rng)

        batches = split_into_batches(list(range(len(images))), self.batch_size, rng=self.rng)
        cards = build_initial_cards(k)
        steps: list[EpisodeStep] = []

        for step_idx, batch_pos in enumerate(batches):
            batch_images = [images[p] for p in batch_pos]
            batch_labels = [one_labels[p] for p in batch_pos]
            cards_before = [ClusterCard(c.cluster_id, c.name, c.description, c.count) for c in cards]
            cards = update_cards_from_assignments(cards, batch_labels, templates)
            steps.append(EpisodeStep(
                step_idx=step_idx,
                images=batch_images,
                gt_assignments=batch_labels,
                cards_before=cards_before,
                cards_after=[ClusterCard(c.cluster_id, c.name, c.description, c.count) for c in cards],
                image_ids=[indices[p] for p in batch_pos],
            ))

        return ClusterEpisode(
            dataset="celeba",
            criterion=instruction,
            k=k,
            steps=steps,
            total_images=len(images),
            global_labels=one_labels,
            metadata={"criterion_key": criterion},
        )

    def iter_episodes(self, n_episodes: int | None = None) -> Iterator[ClusterEpisode]:
        count = 0
        criteria_cycle = list(CRITERIA.keys())
        while n_episodes is None or count < n_episodes:
            criterion = criteria_cycle[count % len(criteria_cycle)]
            yield self.build_episode(criterion)
            count += 1
