"""SupCon training pipeline for criterion-conditioned visual encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class EncoderTrainer:

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.get("output_dir", "artifacts/vlm/encoder"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _build_encoder(self) -> Any:
        from vlc.model.encoder import build_encoder, load_encoder_checkpoint

        model_cfg = self.cfg.get("model", {})
        model_id = model_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        lora_path = model_cfg.get("lora_path", None)
        proj_cfg = model_cfg.get("projection", {})
        in_dim = proj_cfg.get("in_dim", 2048)
        mid_dim = proj_cfg.get("mid_dim", 512)
        out_dim = proj_cfg.get("out_dim", 256)

        if lora_path and (Path(lora_path).exists()):
            proj_pt = str(Path(lora_path).parent / "proj_head.pt")
            if Path(proj_pt).exists():
                print(f"[trainer] Loading full checkpoint from {lora_path}")
                encoder = load_encoder_checkpoint(model_id, lora_path, proj_pt, in_dim, mid_dim, out_dim)
            else:
                print(f"[trainer] Warm-starting LoRA from {lora_path}, fresh projection head")
                encoder = build_encoder(model_id, lora_path, in_dim, mid_dim, out_dim)
        else:
            print("[trainer] Building encoder from scratch")
            encoder = build_encoder(model_id, None, in_dim, mid_dim, out_dim)

        if model_cfg.get("gradient_checkpointing", True):
            encoder.backbone.gradient_checkpointing_enable()

        return encoder

    def _build_datasets(self) -> tuple[Any, Any]:
        from vlc.episodes.embedding_dataset import EmbeddingDataset

        data_cfg = self.cfg.get("data", {})
        train_ds = EmbeddingDataset(
            data_cfg["datasets"],
            data_cfg.get("batches_per_epoch", 400),
            data_cfg.get("images_per_class", 8),
            split="train",
            seed=self.cfg.get("seed", 42),
        )
        val_ds = EmbeddingDataset(
            data_cfg["datasets"],
            data_cfg.get("val_batches", 50),
            data_cfg.get("images_per_class", 8),
            split="val",
            seed=self.cfg.get("seed", 42) + 1,
        )
        return train_ds, val_ds

    def _compute_loss(self, encoder: Any, batch: dict) -> torch.Tensor:
        from vlc.core.losses import supcon_loss

        z = encoder.encode_batch(batch["criterion"], batch["images"])
        labels = torch.tensor(batch["labels"], dtype=torch.long, device=z.device)
        return supcon_loss(z, labels, self.cfg.get("training", {}).get("temperature", 0.1))

    def _val_loss(self, encoder: Any, val_ds: Any) -> float:
        encoder.backbone.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(len(val_ds)):
                try:
                    loss = self._compute_loss(encoder, val_ds[i])
                    total += float(loss.item())
                    count += 1
                except Exception:
                    pass
        encoder.backbone.train()
        return total / max(count, 1)

    def train(self) -> None:
        from vlc.model.encoder import save_encoder_checkpoint

        train_cfg = self.cfg.get("training", {})
        n_epochs = train_cfg.get("epochs", 10)
        lr = float(train_cfg.get("lr", 1e-4))
        grad_accum = train_cfg.get("grad_accum_steps", 4)
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        log_every = train_cfg.get("log_every", 20)

        encoder = self._build_encoder()
        train_ds, val_ds = self._build_datasets()

        trainable = (
            [p for p in encoder.backbone.parameters() if p.requires_grad]
            + list(encoder.proj_head.parameters())
        )
        optimizer = AdamW(trainable, lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
        total_steps = max(n_epochs * len(train_ds) // grad_accum, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

        best_val = float("inf")
        for epoch in range(n_epochs):
            train_ds.on_epoch_end()
            encoder.backbone.train()
            epoch_loss, n_batches = 0.0, 0
            optimizer.zero_grad()

            pbar = tqdm(range(len(train_ds)), desc=f"Epoch {epoch+1}/{n_epochs}")
            for step_i in pbar:
                try:
                    loss = self._compute_loss(encoder, train_ds[step_i]) / grad_accum
                    loss.backward()
                    epoch_loss += float(loss.item()) * grad_accum
                    n_batches += 1
                except Exception as e:
                    print(f"  [skip] step {step_i}: {e}")
                    optimizer.zero_grad()
                    continue

                if (step_i + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (step_i + 1) % log_every == 0:
                    avg = epoch_loss / max(n_batches, 1)
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss = epoch_loss / max(n_batches, 1)
            val_loss = self._val_loss(encoder, val_ds)
            print(f"Epoch {epoch+1}/{n_epochs}: train={train_loss:.4f}  val={val_loss:.4f}")

            save_encoder_checkpoint(encoder, str(self.out_dir / f"epoch_{epoch+1:03d}"))
            if val_loss < best_val:
                best_val = val_loss
                save_encoder_checkpoint(encoder, str(self.out_dir / "best"))
                print(f"  -> new best val={best_val:.4f}")

            with open(self.out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch+1, "train": train_loss, "val": val_loss}) + "\n")

        print(f"Done. Best val={best_val:.4f}, checkpoints at {self.out_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    EncoderTrainer(cfg).train()
