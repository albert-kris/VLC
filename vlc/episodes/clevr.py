"""CLEVR episode synthesizer.

CLEVR_v1.0 layout (from vlc/legacy):
  CLEVR_v1.0/images/train/CLEVR_train_XXXXXX.png
  CLEVR_v1.0/scenes/CLEVR_train_scenes.json

Scene object attributes used for clustering:
  color:    {red, blue, green, yellow, cyan, purple, brown, gray} → K=8 or K=4 (top-4)
  shape:    {cube, sphere, cylinder} → K=3
  size:     {small, large} → K=2
  material: {rubber, metal} → K=2

For same-K=4 design we merge smaller-K attributes or restrict to 4 most-common colors.
We expose two modes:
  - full: use all color values (K=8) with instruction "cluster by color"
  - same_k4: use top-4 colors OR shape with K dynamically set
"""

from __future__ import annotations

import json
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


# Criteria that produce K clusters in CLEVR
CRITERIA: dict[str, dict] = {
    "color_4": {
        "instruction": "cluster these images by the dominant object color",
        "attribute": "color",
        "values": ["red", "blue", "green", "yellow"],  # top-4 colors
        "k": 4,
        "card_names": {
            1: ("red objects", "Images where the main object is red"),
            2: ("blue objects", "Images where the main object is blue"),
            3: ("green objects", "Images where the main object is green"),
            4: ("yellow objects", "Images where the main object is yellow"),
        },
    },
    "shape": {
        "instruction": "cluster these images by object shape",
        "attribute": "shape",
        "values": ["cube", "sphere", "cylinder"],
        "k": 3,
        "card_names": {
            1: ("cube-shaped objects", "Rectangular box-shaped objects"),
            2: ("sphere-shaped objects", "Round ball-shaped objects"),
            3: ("cylinder-shaped objects", "Cylindrical tube-shaped objects"),
        },
    },
    "material": {
        "instruction": "cluster these images by surface material",
        "attribute": "material",
        "values": ["rubber", "metal"],
        "k": 2,
        "card_names": {
            1: ("rubber objects", "Objects with matte rubber finish"),
            2: ("metal objects", "Objects with shiny metallic finish"),
        },
    },
}


def _load_clevr_index(clevr_root: Path, split: str = "train") -> list[dict]:
    """Load CLEVR scene JSON → list of {image_path, attributes} dicts."""
    scenes_path = clevr_root / "scenes" / f"CLEVR_{split}_scenes.json"
    if not scenes_path.exists():
        raise FileNotFoundError(f"CLEVR scenes not found: {scenes_path}")
    with open(scenes_path, encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for scene in data["scenes"]:
        img_name = scene["image_filename"]
        img_path = clevr_root / "images" / split / img_name
        if not img_path.exists():
            continue
        # Use first object's attributes as scene label (simplified for clustering)
        if not scene["objects"]:
            continue
        obj = scene["objects"][0]
        records.append({
            "image_path": img_path,
            "color": obj.get("color", ""),
            "shape": obj.get("shape", ""),
            "material": obj.get("material", ""),
            "size": obj.get("size", ""),
        })
    return records


class CLEVREpisodeBuilder:
    """Builds streaming clustering episodes from CLEVR dataset."""

    def __init__(
        self,
        clevr_root: str | Path,
        n_images: int = 24,
        batch_size: int = 8,
        seed: int = 42,
        split: str = "train",
    ) -> None:
        self.clevr_root = Path(clevr_root)
        self.n_images = n_images
        self.batch_size = batch_size
        self.rng = random.Random(seed)

        records = _load_clevr_index(self.clevr_root, split)
        self._cluster_index: dict[str, dict[int, list[int]]] = {}

        for cname, meta in CRITERIA.items():
            attr = meta["attribute"]
            values = meta["values"]
            val_to_id = {v: i + 1 for i, v in enumerate(values)}
            cluster_map: dict[int, list[int]] = {i + 1: [] for i in range(len(values))}
            for idx, rec in enumerate(records):
                v = rec.get(attr, "")
                if v in val_to_id:
                    cluster_map[val_to_id[v]].append(idx)
            self._cluster_index[cname] = cluster_map

        self._records = records

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
            sampled.extend((rec_idx, cid) for rec_idx in chosen)

        self.rng.shuffle(sampled)
        indices, one_labels = zip(*sampled) if sampled else ([], [])
        indices, one_labels = list(indices), list(one_labels)

        images = [Image.open(self._records[i]["image_path"]).convert("RGB") for i in indices]
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
            dataset="clevr",
            criterion=meta["instruction"],
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
