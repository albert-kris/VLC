"""Vector-based clustering inference using a trained CriterionEncoder."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class EmbedClusterResult:
    assignments: list[int]    # 1-indexed
    embeddings: np.ndarray    # (N, d)
    centroids: np.ndarray     # (K, d)
    n_images: int
    k: int
    criterion: str


class VectorClusterer:
    def __init__(self, encoder: Any, batch_size: int = 8) -> None:
        self.encoder = encoder
        self.batch_size = batch_size

    def encode_all(self, criterion: str, images: list[Image.Image]) -> np.ndarray:
        import torch
        all_z = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i: i + self.batch_size]
            with torch.no_grad():
                z = self.encoder.encode_batch(criterion, batch)
            all_z.append(z.cpu().float().numpy())
        return np.concatenate(all_z, axis=0)

    def cluster(self, images: list[Image.Image], criterion: str, k: int, seed: int = 42) -> EmbedClusterResult:
        from sklearn.cluster import KMeans

        embeddings = self.encode_all(criterion, images)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(embeddings)
        assignments = [int(l) + 1 for l in km.labels_.tolist()]

        return EmbedClusterResult(
            assignments=assignments,
            embeddings=embeddings,
            centroids=km.cluster_centers_.astype(np.float32),
            n_images=len(images),
            k=k,
            criterion=criterion,
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="VLM Vector Clustering")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--lora", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--criterion", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="artifacts/cluster_result.json")
    p.add_argument("--load-4bit", action="store_true")
    args = p.parse_args(argv)

    from vlc.model.encoder import load_encoder_checkpoint

    lora_dir = Path(args.lora)
    encoder = load_encoder_checkpoint(
        model_id=args.model,
        lora_path=str(lora_dir / "lora"),
        proj_path=str(lora_dir / "proj_head.pt"),
        load_in_4bit=args.load_4bit,
    )

    img_dir = Path(args.images)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [Image.open(p).convert("RGB") for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    print(f"Loaded {len(images)} images")

    result = VectorClusterer(encoder, args.batch_size).cluster(images, args.criterion, args.k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"criterion": result.criterion, "k": result.k,
                   "n_images": result.n_images, "assignments": result.assignments}, f, indent=2)
    print(f"Saved to {out}")
