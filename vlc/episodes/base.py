"""Core data structures for streaming clustering episodes.

An Episode represents one complete clustering task:
  - N images split into B batches
  - K cluster slots
  - Natural-language criterion
  - Ground-truth assignments and card supervision texts

Each EpisodeStep is one batch: images + current cards → assignments + updated cards.
The training format is a multi-turn Qwen-VL conversation where every assistant turn
contributes to the SFT loss.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class ClusterCard:
    """One cluster slot's text description (the model's running memory)."""

    cluster_id: int          # 1-indexed
    name: str                # short label, e.g. "sphere-shaped objects"
    description: str         # one-line elaboration
    count: int = 0           # number of images assigned so far

    def to_text(self) -> str:
        if self.count == 0:
            return f"C{self.cluster_id} | (empty) | 0 images"
        return f"C{self.cluster_id} | {self.name} | {self.count} images"

    def to_dict(self) -> dict:
        return {
            "id": self.cluster_id,
            "name": self.name,
            "description": self.description,
            "count": self.count,
        }

    @classmethod
    def empty(cls, cluster_id: int) -> "ClusterCard":
        return cls(cluster_id=cluster_id, name="", description="", count=0)

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterCard":
        return cls(
            cluster_id=int(d["id"]),
            name=str(d.get("name", "")),
            description=str(d.get("description", "")),
            count=int(d.get("count", 0)),
        )


def format_cards_text(cards: list[ClusterCard]) -> str:
    """Render cluster cards as a compact text block for the model prompt."""
    return "\n".join(c.to_text() for c in cards)


@dataclass
class EpisodeStep:
    """One batch step in a streaming clustering episode."""

    step_idx: int                  # 0-indexed
    images: list[Image.Image]      # raw PIL images for this batch
    gt_assignments: list[int]      # 1-indexed cluster IDs for each image
    cards_before: list[ClusterCard]  # cards state before this step
    cards_after: list[ClusterCard]   # ground-truth cards after this step
    image_ids: list[int]           # global indices within the full episode


@dataclass
class ClusterEpisode:
    """A complete streaming clustering episode for one training example."""

    dataset: str
    criterion: str             # natural-language clustering instruction
    k: int                     # number of clusters
    steps: list[EpisodeStep]   # ordered batch steps
    total_images: int
    global_labels: list[int]   # 1-indexed gt cluster assignment for every image (in episode order)
    metadata: dict = field(default_factory=dict)

    def to_qwen_messages(self) -> list[dict[str, Any]]:
        """Convert episode to Qwen2.5-VL multi-turn chat message list."""
        system_msg = {
            "role": "system",
            "content": (
                "You are a visual clustering assistant. "
                "Given a batch of images, the current cluster cards, and a clustering criterion, "
                "assign each image to one of the K clusters and update the cluster cards with "
                "concise names and descriptions. "
                "Always respond with valid JSON matching the schema exactly."
            ),
        }
        messages = [system_msg]
        for step in self.steps:
            user_content = _build_user_content(
                criterion=self.criterion,
                k=self.k,
                cards=step.cards_before,
                images=step.images,
                step_idx=step.step_idx,
                total_steps=len(self.steps),
            )
            assistant_content = _build_assistant_content(
                assignments=step.gt_assignments,
                cards=step.cards_after,
            )
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})
        return messages


def _build_user_content(
    criterion: str,
    k: int,
    cards: list[ClusterCard],
    images: list[Image.Image],
    step_idx: int,
    total_steps: int,
) -> list[dict]:
    """Build Qwen-VL user content block: text + interleaved images."""
    n = len(images)
    cards_text = format_cards_text(cards)
    header = (
        f"Clustering criterion: {criterion}\n"
        f"Number of clusters: K={k}\n\n"
        f"Current cluster cards:\n{cards_text}\n\n"
        f"Batch {step_idx + 1}/{total_steps} — assign each image below to a cluster "
        f"(C1..C{k}) and update the cluster cards.\n\n"
        f"Images ({n} total):\n"
    )
    content: list[dict] = [{"type": "text", "text": header}]
    for i, img in enumerate(images):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[img_{i}]\n"})
    content.append({
        "type": "text",
        "text": (
            '\nRespond with JSON only:\n'
            '{"assignments": [<C-id for img_0>, ...], '
            '"cards": [{"id": <1..K>, "name": "...", "description": "..."}, ...]}'
        ),
    })
    return content


def _build_assistant_content(
    assignments: list[int],
    cards: list[ClusterCard],
) -> str:
    """Serialize the ground-truth assistant response as JSON string."""
    return json.dumps(
        {
            "assignments": assignments,
            "cards": [
                {"id": c.cluster_id, "name": c.name, "description": c.description}
                for c in cards
            ],
        },
        ensure_ascii=False,
    )


def build_initial_cards(k: int) -> list[ClusterCard]:
    """Return K empty cluster cards for step 0."""
    return [ClusterCard.empty(i + 1) for i in range(k)]


def update_cards_from_assignments(
    cards: list[ClusterCard],
    assignments: list[int],
    card_templates: dict[int, tuple[str, str]],
) -> list[ClusterCard]:
    """Return updated cards: increment counts and overwrite name/description from templates."""
    import copy
    new_cards = copy.deepcopy(cards)
    for cid in assignments:
        idx = cid - 1
        new_cards[idx].count += 1
        if cid in card_templates:
            name, desc = card_templates[cid]
            new_cards[idx].name = name
            new_cards[idx].description = desc
    return new_cards


def split_into_batches(
    items: list,
    batch_size: int,
    rng: random.Random | None = None,
    shuffle: bool = True,
) -> list[list]:
    """Split a list into batches, optionally shuffling first."""
    if shuffle and rng is not None:
        items = list(items)
        rng.shuffle(items)
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]
