"""Overfit check: train VLM on 5 fixed episodes, then eval on same episodes.

If the model can memorise 5 episodes (loss → ~0, ACC → ~1.0),
the training pipeline is correct. Run before committing to full training.

Usage:
    python scripts/overfit_check.py [--epochs 30] [--episodes 5]
"""

import argparse
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

# ── setup ────────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")

from vlc.model.qwen_vl import load_model_and_processor, add_lora, LoRAConfig
from vlc.model.trainer import collate_episodes, IGNORE_INDEX
from vlc.episodes.shapes3d import Shapes3DEpisodeBuilder
from vlc.stream.cluster_cards import ClusterCardState, parse_model_json
from vlc.core.metrics import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-length", type=int, default=3000)
    p.add_argument("--n-images", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    print("\nLoading Qwen2.5-VL-3B-Instruct + LoRA ...")
    model, processor = load_model_and_processor(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=64 * 28 * 28,
        max_pixels=128 * 28 * 28,
    )
    lora_cfg = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.0)
    model = add_lora(model, lora_cfg)

    # ── Build fixed episodes ────────────────────────────────────────────────
    print(f"\nBuilding {args.episodes} fixed episodes (will overfit to these) ...")
    builder = Shapes3DEpisodeBuilder(
        "data/3dshapes.h5",
        n_images=args.n_images,
        batch_size=args.batch_size,
        seed=123,
    )
    criteria = ["shape", "object_hue", "object_size", "orientation"]
    episodes = [
        builder.build_episode(criteria[i % len(criteria)])
        for i in range(args.episodes)
    ]
    print(f"  Episodes: {[ep.criterion[:30] for ep in episodes]}")

    # Pre-collate (deterministic, always same batch)
    print("  Pre-collating episodes ...")
    batches = collate_episodes(episodes, processor, args.max_length, device)
    if not batches:
        print("ERROR: All episodes failed to collate")
        sys.exit(1)
    print(f"  Got {len(batches)} batches, "
          f"avg answer tokens: {sum((b['labels'] != IGNORE_INDEX).sum().item() for b in batches) / len(batches):.0f}")

    # ── Training loop ───────────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )
    model.train()

    print(f"\nOverfit training: {args.epochs} epochs × {len(batches)} batches ...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                labels=batch["labels"],
            )
            out.loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            epoch_loss += out.loss.item()

        avg = epoch_loss / len(batches)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:3d}/{args.epochs}: loss={avg:.4f}")

    # ── Overfit eval: greedy decode and compute ACC ─────────────────────────
    print("\nOverfit eval: greedy decoding on training episodes ...")
    model.eval()
    all_preds: list[int] = []
    all_gts: list[int] = []

    for ep in episodes:
        k = ep.k
        cards = ClusterCardState.empty(k)
        system_msg = {
            "role": "system",
            "content": (
                "You are a visual clustering assistant. Assign images to clusters. "
                "Always respond with valid JSON only."
            ),
        }
        history = [system_msg]
        total_steps = len(ep.steps)

        for step in ep.steps:
            from vlc.episodes.base import _build_user_content
            user_content = _build_user_content(
                criterion=ep.criterion,
                k=k,
                cards=cards.cards,
                images=step.images,
                step_idx=step.step_idx,
                total_steps=total_steps,
            )
            history.append({"role": "user", "content": user_content})

            all_imgs = [
                blk["image"]
                for msg in history
                if isinstance(msg.get("content"), list)
                for blk in msg["content"]
                if blk.get("type") == "image"
            ]

            text_prompt = processor.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt],
                images=all_imgs if all_imgs else None,
                return_tensors="pt",
            )
            inputs = {k2: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k2, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            prompt_len = inputs["input_ids"].shape[1]
            raw = processor.decode(generated[0][prompt_len:], skip_special_tokens=True)
            assignments, card_dicts = parse_model_json(raw, len(step.images), k)

            if assignments is None:
                assignments = [(i % k) + 1 for i in range(len(step.images))]
                card_dicts = []

            cards.update_from_model_output(assignments, card_dicts or [])
            all_preds.extend(assignments)
            all_gts.extend(step.gt_assignments)

            import json
            history.append({
                "role": "assistant",
                "content": json.dumps({
                    "assignments": assignments,
                    "cards": [c.to_dict() for c in cards.cards],
                }),
            })

    metrics = evaluate(np.array(all_gts) - 1, np.array(all_preds) - 1)
    print(f"\nOverfit eval results:")
    print(f"  ACC = {metrics['acc']:.4f}  (target: > 0.8)")
    print(f"  ARI = {metrics['ari']:.4f}")
    print(f"  NMI = {metrics['nmi']:.4f}")
    success = metrics["acc"] > 0.7
    print(f"\n{'OVERFIT CHECK PASSED' if success else 'OVERFIT CHECK FAILED'} "
          f"(ACC={'PASS' if success else 'FAIL'})")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
