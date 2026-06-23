"""SwAV-RL Trainer: two-phase training for last-hidden-state clustering.

Phase 1 — SwAV warmup (direct backprop through LoRA):
    同一张图的两个增广 -> 大模型最后隐状态 -> 投射层 -> SwAV loss.

Phase 2 — GRPO 强化学习:
    同一张图的两个增广 -> 大模型最后隐状态 -> 投射层 -> z_a, z_b.
    奖励 = 两个投射向量是否相同（余弦相似度）.
    只更新 LoRA + 投射层；SwAV 头仅用于聚类评测.

Usage:
    python -m vlc.train.swav_rl_trainer --config configs/vlm/swav_rl_cifar10.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Image augmentation
# ---------------------------------------------------------------------------

class TwoViewAug:
    def __init__(self, size: int = 64) -> None:
        self.size = size

    def _aug(self, img: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = img.size
        scale = random.uniform(0.6, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        x0 = random.randint(0, max(0, w - nw))
        y0 = random.randint(0, max(0, h - nh))
        img = img.crop((x0, y0, x0 + nw, y0 + nh)).resize((self.size, self.size), Image.BILINEAR)
        if random.random() > 0.8:
            img = img.convert("L").convert("RGB")
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        return img

    def __call__(self, img: Image.Image):
        return self._aug(img), self._aug(img)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_cifar10(data_dir: str, n_per_class: int | None, image_size: int = 64):
    data_dir = Path(data_dir)
    imgs_list, labels_list = [], []
    for i in range(1, 6):
        with open(data_dir / f"data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs_list.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels_list.extend(d[b"labels"])
    imgs = np.concatenate(imgs_list, axis=0)
    labels = np.array(labels_list)
    if n_per_class is not None:
        keep = []
        for c in range(10):
            keep.append(np.where(labels == c)[0][:n_per_class])
        keep = np.sort(np.concatenate(keep))
        imgs, labels = imgs[keep], labels[keep]
    images = [Image.fromarray(a).resize((image_size, image_size), Image.BILINEAR) for a in imgs]
    return images, labels


# ---------------------------------------------------------------------------
# Cluster evaluation
# ---------------------------------------------------------------------------

def cluster_acc(y_true, y_pred):
    k = max(y_true.max(), y_pred.max()) + 1
    W = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        W[p, t] += 1
    ri, ci = linear_sum_assignment(W.max() - W)
    return W[ri, ci].sum() / len(y_true)


@torch.no_grad()
def eval_clustering(encoder, swav_head, images, labels, instruction, device, encode_bs=8):
    encoder.backbone.eval()
    all_z = []
    for i in range(0, len(images), encode_bs):
        z = encoder.encode_batch(instruction, images[i:i + encode_bs])
        all_z.append(z.detach().cpu().float())
    z_all = torch.cat(all_z, dim=0)

    swav_head.normalize_prototypes()
    scores = swav_head(z_all.to(device))
    pred = scores.argmax(1).cpu().numpy()

    acc = cluster_acc(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    encoder.backbone.train()
    return acc, nmi, ari


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class SwavRLTrainer:

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.get("output_dir", "artifacts/vlm/swav_rl"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_encoder(self):
        from vlc.model.encoder import build_encoder, load_encoder_checkpoint
        model_cfg = self.cfg["model"]
        model_id = model_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        lora_path = model_cfg.get("lora_path", None)
        p = model_cfg.get("projection", {})
        in_dim, mid_dim, out_dim = p.get("in_dim", 2048), p.get("mid_dim", 512), p.get("out_dim", 256)
        if lora_path and Path(lora_path).exists():
            proj_pt = str(Path(lora_path).parent / "proj_head.pt")
            if Path(proj_pt).exists():
                encoder = load_encoder_checkpoint(model_id, lora_path, proj_pt, in_dim, mid_dim, out_dim)
            else:
                encoder = build_encoder(model_id, lora_path, in_dim, mid_dim, out_dim)
        else:
            encoder = build_encoder(model_id, None, in_dim, mid_dim, out_dim)
        if model_cfg.get("gradient_checkpointing", True):
            encoder.backbone.gradient_checkpointing_enable()
        return encoder

    def _build_swav_head(self, out_dim: int, k: int):
        from vlc.model.swav_head import SwAVHead
        return SwAVHead(dim=out_dim, n_prototypes=k).to(self.device)

    def _save(self, encoder, swav_head, tag: str) -> None:
        from vlc.model.encoder import save_encoder_checkpoint
        save_encoder_checkpoint(encoder, str(self.out_dir / tag))
        torch.save(swav_head.state_dict(), str(self.out_dir / tag / "swav_head.pt"))

    def _load_checkpoint(self, encoder, swav_head, ckpt_dir: str) -> None:
        """Load encoder LoRA + proj_head + swav_head from a saved directory."""
        from peft import PeftModel

        ckpt = Path(ckpt_dir)
        proj_path = ckpt / "proj_head.pt"
        lora_path = ckpt / "lora"
        if proj_path.exists():
            encoder.proj_head.load_state_dict(
                torch.load(proj_path, map_location=encoder.device, weights_only=True)
            )
        if lora_path.exists():
            if isinstance(encoder.backbone, PeftModel):
                encoder.backbone.load_adapter(str(lora_path), adapter_name="default")
            else:
                encoder.backbone = PeftModel.from_pretrained(
                    encoder.backbone, str(lora_path), is_trainable=True,
                )
        swav_pt = ckpt / "swav_head.pt"
        if swav_pt.exists():
            swav_head.load_state_dict(torch.load(swav_pt, map_location=self.device, weights_only=True))
        print(f"[checkpoint] Loaded from {ckpt_dir}")

    def _init_prototypes(self, encoder, swav_head, images, instruction, k, encode_bs=8):
        """KMeans-init SwAV prototypes from current encoder embeddings."""
        n = len(images)
        print("[init] KMeans prototype init ...")
        with torch.no_grad():
            encoder.backbone.eval()
            z0_chunks = []
            for i in range(0, min(n, 200), encode_bs):
                z0_chunks.append(encoder.encode_batch(instruction, images[i:i + encode_bs]).detach().cpu())
            z0 = torch.cat(z0_chunks, dim=0).float().numpy()
        km = KMeans(n_clusters=k, n_init=10, random_state=self.cfg.get("seed", 42))
        km.fit(z0)
        centroids = torch.tensor(km.cluster_centers_, dtype=swav_head.prototypes.weight.dtype,
                                 device=self.device)
        swav_head.prototypes.weight.data.copy_(F.normalize(centroids, dim=1))
        encoder.backbone.train()
        print(f"[init] Prototypes from KMeans on {z0.shape[0]} embeddings.")

    # ------------------------------------------------------------------
    # Phase 1: SwAV warmup — direct backprop
    # ------------------------------------------------------------------

    def _train_warmup(self, encoder, swav_head, images, labels, instruction):
        from vlc.core.losses import swav_loss

        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]
        warmup_epochs = tcfg.get("warmup_epochs", 5)
        lr = float(tcfg.get("lr", 1e-4))  # used for scheduler eta_min scale
        acc_floor = float(tcfg.get("warmup_acc_floor", 0.60))
        early_stop_on_drop = tcfg.get("warmup_early_stop_on_drop", True)
        batch_size = dcfg.get("images_per_batch", 16)
        steps_per_epoch = dcfg.get("steps_per_epoch", 20)
        grad_accum = tcfg.get("grad_accum_steps", 4)
        max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        temperature = float(tcfg.get("temperature", 0.1))
        epsilon = float(tcfg.get("sinkhorn_epsilon", 0.05))
        sink_iters = int(tcfg.get("sinkhorn_iters", 3))
        encode_bs = dcfg.get("encode_batch_size", 8)
        k = tcfg["k"]
        n = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))
        rng = random.Random(self.cfg.get("seed", 42))

        # Init SwAV prototypes from KMeans on initial embeddings
        print("[warmup] KMeans prototype init ...")
        with torch.no_grad():
            encoder.backbone.eval()
            z0_chunks = []
            for i in range(0, min(n, 200), encode_bs):
                z0_chunks.append(encoder.encode_batch(instruction, images[i:i + encode_bs]).detach().cpu())
            z0 = torch.cat(z0_chunks, dim=0).float().numpy()
        km = KMeans(n_clusters=k, n_init=10, random_state=self.cfg.get("seed", 42))
        km.fit(z0)
        centroids = torch.tensor(km.cluster_centers_, dtype=swav_head.prototypes.weight.dtype,
                                 device=self.device)
        swav_head.prototypes.weight.data.copy_(F.normalize(centroids, dim=1))
        encoder.backbone.train()
        print(f"[warmup] Prototypes initialised from KMeans on {z0.shape[0]} embeddings.")

        freeze_backbone = tcfg.get("warmup_freeze_backbone", False)
        if freeze_backbone:
            for p in encoder.backbone.parameters():
                p.requires_grad_(False)
            print("[warmup] Backbone FROZEN — only proj_head + prototypes trainable.")

        lr_proj = float(tcfg.get("lr_proj", tcfg.get("lr", 1e-4)))
        lr_proto = float(tcfg.get("lr_proto", tcfg.get("lr", 1e-4)))
        param_groups = []
        if not freeze_backbone:
            param_groups.append({"params": [p for p in encoder.backbone.parameters() if p.requires_grad],
                                 "lr": float(tcfg.get("lr_enc", tcfg.get("lr", 1e-5)))})
        param_groups.append({"params": list(encoder.proj_head.parameters()), "lr": lr_proj})
        param_groups.append({"params": list(swav_head.parameters()), "lr": lr_proto})

        trainable = [p for g in param_groups for p in g["params"]]
        optimizer = AdamW(param_groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))
        total_steps = max(warmup_epochs * steps_per_epoch // grad_accum, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

        best_acc = -1.0
        prev_acc = None
        for epoch in range(warmup_epochs):
            acc, nmi, ari = eval_clustering(encoder, swav_head, images, labels,
                                            instruction, self.device, encode_bs)
            print(f"[warmup {epoch}/{warmup_epochs}] ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
            with open(self.out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps({"phase": "warmup", "epoch": epoch,
                                    "acc": acc, "nmi": nmi, "ari": ari}) + "\n")

            if epoch > 0 and acc < acc_floor:
                print(f"[warmup] ACC {acc:.4f} < floor {acc_floor:.2f} — stopping warmup early.")
                break
            if early_stop_on_drop and prev_acc is not None and acc < prev_acc:
                print(f"[warmup] ACC dropped {prev_acc:.4f} -> {acc:.4f} — stopping warmup early.")
                break
            prev_acc = acc

            if acc > best_acc:
                best_acc = acc
                self._save(encoder, swav_head, "warmup_best")
                print(f"  -> warmup best ACC={best_acc:.4f}")

            encoder.backbone.train()
            epoch_loss, nb = 0.0, 0
            optimizer.zero_grad()
            pbar = tqdm(range(steps_per_epoch), desc=f"Warmup {epoch+1}/{warmup_epochs}")
            for step_i in pbar:
                idx = rng.sample(range(n), min(batch_size, n))
                views_a, views_b = zip(*[aug(images[i]) for i in idx])
                try:
                    swav_head.normalize_prototypes()
                    z_a = encoder.encode_batch(instruction, list(views_a))
                    z_b = encoder.encode_batch(instruction, list(views_b))
                    scores_a = swav_head(z_a)
                    scores_b = swav_head(z_b)
                    loss = swav_loss(scores_a, scores_b, temperature, epsilon, sink_iters)
                    (loss / grad_accum).backward()
                    epoch_loss += loss.item()
                    nb += 1
                except Exception as e:
                    print(f"  [skip] {e}")
                    optimizer.zero_grad()
                    continue
                if (step_i + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                pbar.set_postfix(loss=f"{epoch_loss/max(nb,1):.4f}")
            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        self._save(encoder, swav_head, "warmup_final")
        print(f"[warmup] Done. Best ACC={best_acc:.4f}")

    # ------------------------------------------------------------------
    # Phase 2: GRPO — 同图两增广，SwAV 一致性为奖励，只训 LoRA + 投射层
    # ------------------------------------------------------------------

    def _swav_consistency(self, encoder, swav_head, instruction, va, vb,
                          temperature, epsilon, sink_iters, freeze_backbone=False):
        """最后隐状态 -> 投射层 -> SwAV 原型分配 -> swapped prediction 一致性。

        奖励 = 0.5 * (q_b·log p(z_a) + q_a·log p(z_b))，越大表示两个增广
        被分配到越一致的原型。原型每步归一化后参与训练。
        返回每个样本的奖励 (B,)。
        """
        from vlc.core.losses import sinkhorn
        if freeze_backbone:
            with torch.no_grad():
                h_a = encoder.encode_hidden_batch(instruction, va)
                h_b = encoder.encode_hidden_batch(instruction, vb)
        else:
            h_a = encoder.encode_hidden_batch(instruction, va)
            h_b = encoder.encode_hidden_batch(instruction, vb)
        z_a = encoder.proj_head(h_a)
        z_b = encoder.proj_head(h_b)
        swav_head.normalize_prototypes()
        sa = swav_head(z_a)
        sb = swav_head(z_b)
        q_a = sinkhorn(sa, epsilon, sink_iters)              # (B, K)
        q_b = sinkhorn(sb, epsilon, sink_iters)
        log_pa = F.log_softmax(sa / temperature, dim=1)
        log_pb = F.log_softmax(sb / temperature, dim=1)
        return 0.5 * ((q_b * log_pa).sum(1) + (q_a * log_pb).sum(1))   # (B,)

    def _train_grpo(self, encoder, swav_head, images, labels, instruction):
        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]
        rl_iters    = tcfg.get("rl_iterations", 100)
        batch_size  = dcfg.get("images_per_batch", 8)
        G           = tcfg.get("group_size", 4)
        lr          = float(tcfg.get("rl_lr", 5e-5))
        lr_enc      = float(tcfg.get("rl_lr_enc", tcfg.get("lr_enc", lr)))
        lr_proj     = float(tcfg.get("rl_lr_proj", tcfg.get("lr_proj", lr)))
        lr_proto    = float(tcfg.get("rl_lr_proto", tcfg.get("lr_proto", lr)))
        max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        eval_every  = int(tcfg.get("rl_eval_every", 1))
        encode_bs   = dcfg.get("encode_batch_size", 8)
        temperature = float(tcfg.get("temperature", 0.1))
        epsilon     = float(tcfg.get("sinkhorn_epsilon", 0.05))
        sink_iters  = int(tcfg.get("sinkhorn_iters", 3))
        n = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))
        rng = random.Random(self.cfg.get("seed", 42) + 1)

        # LoRA lr=0 时彻底冻结 backbone，只训投射层 + SwAV 原型
        freeze_lora = lr_enc <= 0
        if freeze_lora:
            for p in encoder.backbone.parameters():
                p.requires_grad_(False)
            encoder.backbone.eval()
            print("[GRPO] LoRA 已冻结，只更新投射层 + 原型")
        for p in swav_head.parameters():
            p.requires_grad_(True)
        param_groups = []
        if not freeze_lora:
            param_groups.append({"params": [p for p in encoder.backbone.parameters() if p.requires_grad], "lr": lr_enc})
        param_groups.extend([
            {"params": list(encoder.proj_head.parameters()), "lr": lr_proj},
            {"params": list(swav_head.parameters()), "lr": lr_proto},
        ])
        trainable = [p for g in param_groups for p in g["params"]]
        optimizer = AdamW(param_groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))
        patience   = int(tcfg.get("early_stop_patience", 0))   # 0 = 不早停
        print(f"[GRPO] 奖励=SwAV 一致性, G={G}, lr_enc={lr_enc} lr_proj={lr_proj} lr_proto={lr_proto}")
        if patience:
            print(f"[GRPO] 早停: 连续 {patience} 个评测点 ACC 不涨则停")

        best_acc = -1.0
        no_improve = 0
        for it in tqdm(range(1, rl_iters + 1), desc="GRPO-RL"):
            batch_imgs = [images[i] for i in rng.sample(range(n), min(batch_size, n))]

            # 采样 G 组增广：每组里每张图只增广一次，得到 (view_a, view_b)
            aug_pairs = []
            rewards = []
            with torch.no_grad():
                encoder.backbone.eval()
                for _ in range(G):
                    va, vb = zip(*[aug(img) for img in batch_imgs])
                    va, vb = list(va), list(vb)
                    aug_pairs.append((va, vb))
                    rewards.append(self._swav_consistency(
                        encoder, swav_head, instruction, va, vb,
                        temperature, epsilon, sink_iters, freeze_lora))

            rewards_t = torch.stack(rewards, dim=0)                         # (G, B)
            mean_r = rewards_t.mean(dim=0, keepdim=True)
            std_r = rewards_t.std(dim=0, keepdim=True).clamp(min=1e-6)
            advantages = (rewards_t - mean_r) / std_r                        # (G, B)

            # 策略梯度：同一组增广，用当前策略重新算 SwAV 一致性，加权 advantage
            if not freeze_lora:
                encoder.backbone.train()
            optimizer.zero_grad()
            loss = torch.zeros(1, device=self.device)
            for g, (va, vb) in enumerate(aug_pairs):
                sim = self._swav_consistency(
                    encoder, swav_head, instruction, va, vb,
                    temperature, epsilon, sink_iters, freeze_lora)
                loss = loss - (advantages[g].detach() * sim).mean()
            (loss / G).backward()
            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            if it % eval_every == 0:
                acc, nmi, ari = eval_clustering(encoder, swav_head, images, labels,
                                                instruction, self.device, encode_bs)
                mean_rew = rewards_t.mean().item()
                print(f"[GRPO it={it:4d}] reward={mean_rew:.4f}  "
                      f"ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
                with open(self.out_dir / "history.jsonl", "a") as f:
                    f.write(json.dumps({"phase": "grpo", "iter": it,
                                        "reward": mean_rew,
                                        "acc": acc, "nmi": nmi, "ari": ari}) + "\n")
                if acc > best_acc:
                    best_acc = acc
                    no_improve = 0
                    self._save(encoder, swav_head, "rl_best")
                    print(f"  -> RL best ACC={best_acc:.4f}")
                else:
                    no_improve += 1
                    if patience and no_improve >= patience:
                        print(f"[GRPO] 早停: {patience} 个评测点无改善，best ACC={best_acc:.4f}")
                        break

        print(f"[GRPO] Done. Best ACC={best_acc:.4f}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]
        k       = tcfg["k"]
        out_dim = self.cfg["model"]["projection"].get("out_dim", 256)

        instruction = (
            "What is the main object in this image? "
            "Choose from: airplane, automobile, bird, cat, deer, dog, "
            "frog, horse, ship, truck."
        )

        print("[trainer] Loading CIFAR-10 ...")
        images, labels = load_cifar10(
            dcfg["data_dir"],
            dcfg.get("max_per_class", 50),
            dcfg.get("image_size", 64),
        )
        print(f"[trainer] {len(images)} images.")

        encoder   = self._build_encoder()
        swav_head = self._build_swav_head(out_dim, k)

        # Optional: load a pre-built checkpoint (e.g. zero-shot warmup_best ACC=0.644)
        init_ckpt = tcfg.get("init_checkpoint") or self.cfg["model"].get("lora_path")
        if init_ckpt and Path(init_ckpt).exists():
            self._load_checkpoint(encoder, swav_head, init_ckpt)
        elif tcfg.get("warmup_epochs", 0) == 0:
            # No warmup and no checkpoint — init prototypes from KMeans on fresh encoder
            self._init_prototypes(encoder, swav_head, images, instruction,
                                  tcfg["k"], dcfg.get("encode_batch_size", 8))

        if tcfg.get("warmup_epochs", 0) > 0:
            print("\n=== Phase 1: SwAV warmup ===")
            self._train_warmup(encoder, swav_head, images, labels, instruction)
            # Reload best warmup checkpoint before RL (not the degraded final weights)
            best_path = self.out_dir / "warmup_best"
            if best_path.exists() and tcfg.get("rl_from_warmup_best", True):
                self._load_checkpoint(encoder, swav_head, str(best_path))
                print(f"[trainer] Reloaded warmup_best before RL.")

        if tcfg.get("rl_iterations", 0) > 0:
            print("\n=== Phase 2: GRPO-RL ===")
            self._train_grpo(encoder, swav_head, images, labels, instruction)

        # Final eval
        acc, nmi, ari = eval_clustering(encoder, swav_head, images, labels,
                                        instruction, self.device,
                                        dcfg.get("encode_batch_size", 8))
        print(f"\n[FINAL] ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
        self._save(encoder, swav_head, "final")


def main(argv=None):
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    SwavRLTrainer(cfg).train()


if __name__ == "__main__":
    main()
