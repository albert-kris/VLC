"""SwAV-RL Trainer: two-phase training for last-hidden-state clustering.

Phase 1 — SwAV warmup (direct backprop through LoRA):
    Two augmented views of each image -> Qwen (add_generation_prompt=True)
    -> last hidden state -> proj_head -> SwAV swapped-prediction loss.
    Trains: LoRA + proj_head + SwAV prototypes.

Phase 2 — GRPO-RL fine-tuning:
    Same two-view setup, but gradient flows via policy gradient (GRPO).
    Reward = SwAV consistency between two views' representations.
    Advantage = reward standardized within the group.
    Policy gradient = PPO-clip style ratio * advantage.
    KL anchor = keep a frozen reference LoRA, penalize KL divergence.

Usage:
    python -m vlc.train.swav_rl_trainer --config configs/vlm/swav_rl_cifar10.yaml
"""

from __future__ import annotations

import argparse
import copy
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

    # ------------------------------------------------------------------
    # Phase 1: SwAV warmup — direct backprop
    # ------------------------------------------------------------------

    def _train_warmup(self, encoder, swav_head, images, labels, instruction):
        from vlc.core.losses import swav_loss

        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]
        warmup_epochs = tcfg.get("warmup_epochs", 5)
        lr = float(tcfg.get("lr", 1e-4))
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

        trainable = (
            [p for p in encoder.backbone.parameters() if p.requires_grad]
            + list(encoder.proj_head.parameters())
            + list(swav_head.parameters())
        )
        optimizer = AdamW(trainable, lr=lr, weight_decay=float(tcfg.get("weight_decay", 0.01)))
        total_steps = max(warmup_epochs * steps_per_epoch // grad_accum, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

        best_acc = -1.0
        for epoch in range(warmup_epochs):
            acc, nmi, ari = eval_clustering(encoder, swav_head, images, labels,
                                            instruction, self.device, encode_bs)
            print(f"[warmup {epoch}/{warmup_epochs}] ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
            with open(self.out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps({"phase": "warmup", "epoch": epoch,
                                    "acc": acc, "nmi": nmi, "ari": ari}) + "\n")
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

        print(f"[warmup] Done. Best ACC={best_acc:.4f}")

    # ------------------------------------------------------------------
    # Phase 2: GRPO-RL
    # ------------------------------------------------------------------

    def _train_grpo(self, encoder, swav_head, images, labels, instruction):
        """GRPO-style RL fine-tuning.

        For each mini-batch of B images, sample G augmented view-pairs per
        image. Reward = per-sample SwAV swapped-prediction score (higher =
        two views agree on cluster assignment). Advantage = reward
        standardised within the G-sample group for each image.

        Policy log-probability is modelled as Gaussian over the hidden state
        vector, enabling a continuous-action PPO-clip update without discrete
        token sampling.

        KL penalty anchors the policy to a frozen reference LoRA snapshot
        taken at the start of RL training.
        """
        from vlc.core.losses import sinkhorn

        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]
        rl_iters    = tcfg.get("rl_iterations", 200)
        batch_size  = dcfg.get("images_per_batch", 8)
        G           = tcfg.get("group_size", 4)
        lr          = float(tcfg.get("rl_lr", 5e-5))
        clip_eps    = float(tcfg.get("clip_eps", 0.2))
        kl_beta     = float(tcfg.get("kl_beta", 0.01))
        sigma       = float(tcfg.get("policy_sigma", 1.0))   # std of Gaussian policy
        n_reuse     = int(tcfg.get("n_reuse", 2))
        temperature = float(tcfg.get("temperature", 0.1))
        epsilon     = float(tcfg.get("sinkhorn_epsilon", 0.05))
        sink_iters  = int(tcfg.get("sinkhorn_iters", 3))
        max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        eval_every  = int(tcfg.get("rl_eval_every", 20))
        encode_bs   = dcfg.get("encode_batch_size", 8)
        n = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))
        rng = random.Random(self.cfg.get("seed", 42) + 1)
        sigma2 = sigma ** 2

        # Frozen reference policy
        ref_encoder = copy.deepcopy(encoder)
        for p in ref_encoder.backbone.parameters():
            p.requires_grad_(False)
        for p in ref_encoder.proj_head.parameters():
            p.requires_grad_(False)
        ref_encoder.eval()
        print(f"[GRPO] Reference policy frozen. sigma={sigma}, G={G}, clip_eps={clip_eps}")

        trainable = (
            [p for p in encoder.backbone.parameters() if p.requires_grad]
            + list(encoder.proj_head.parameters())
            + list(swav_head.parameters())
        )
        optimizer = AdamW(trainable, lr=lr, weight_decay=float(tcfg.get("weight_decay", 0.01)))
        best_acc = -1.0

        for it in tqdm(range(1, rl_iters + 1), desc="GRPO-RL"):

            # ── Rollout: collect hidden states and rewards (no grad) ──────
            encoder.backbone.eval()
            batch_idx  = rng.sample(range(n), min(batch_size, n))
            batch_imgs = [images[i] for i in batch_idx]

            # h_a_traj[g] = (B, H) hidden states for view-a, group g
            h_a_traj, h_b_traj = [], []
            h_a_ref,  h_b_ref  = [], []

            with torch.no_grad():
                for g in range(G):
                    va, vb = zip(*[aug(img) for img in batch_imgs])
                    h_a = encoder.encode_hidden_batch(instruction, list(va))      # (B, H)
                    h_b = encoder.encode_hidden_batch(instruction, list(vb))
                    h_a_traj.append(h_a)
                    h_b_traj.append(h_b)
                    # Reference policy hidden states for same views
                    h_a_traj[-1]  # already stored
                    r_a = ref_encoder.encode_hidden_batch(instruction, list(va))
                    r_b = ref_encoder.encode_hidden_batch(instruction, list(vb))
                    h_a_ref.append(r_a)
                    h_b_ref.append(r_b)

            # Compute per-sample SwAV consistency reward for each group
            rewards = []  # list of (B,) tensors, length G
            with torch.no_grad():
                for g in range(G):
                    z_a = encoder.proj_head(h_a_traj[g])   # (B, out_dim)
                    z_b = encoder.proj_head(h_b_traj[g])
                    swav_head.normalize_prototypes()
                    sa = swav_head(z_a)                     # (B, K)
                    sb = swav_head(z_b)
                    q_a = sinkhorn(sa, epsilon, sink_iters)  # (B, K)
                    q_b = sinkhorn(sb, epsilon, sink_iters)
                    log_pa = F.log_softmax(sa / temperature, dim=1)
                    log_pb = F.log_softmax(sb / temperature, dim=1)
                    # Per-sample reward: mean log-likelihood under swapped codes
                    r = 0.5 * ((q_b * log_pa).sum(1) + (q_a * log_pb).sum(1))  # (B,)
                    rewards.append(r)

            # Group-relative advantage: standardise over G for each sample
            rewards_t  = torch.stack(rewards, dim=0)          # (G, B)
            mean_r     = rewards_t.mean(dim=0, keepdim=True)
            std_r      = rewards_t.std(dim=0, keepdim=True).clamp(min=1e-6)
            advantages = (rewards_t - mean_r) / std_r          # (G, B)

            # ── Policy-gradient update (with grad) ───────────────────────
            encoder.backbone.train()
            for _ in range(n_reuse):
                optimizer.zero_grad()
                total_pg  = torch.zeros(1, device=self.device)
                total_kl  = torch.zeros(1, device=self.device)

                for g in range(G):
                    va, vb = zip(*[aug(img) for img in batch_imgs])
                    # Current-policy hidden states (with grad)
                    h_a_cur = encoder.encode_hidden_batch(instruction, list(va))   # (B, H)
                    h_b_cur = encoder.encode_hidden_batch(instruction, list(vb))

                    h_a_old = h_a_traj[g].detach()
                    h_b_old = h_b_traj[g].detach()

                    # Gaussian log-ratio: log π_cur(h_old) − log π_old(h_old)
                    # Under Gaussian(μ_cur, σ²I) evaluated at h_old:
                    #   log π_cur ∝ −||h_old − h_a_cur||² / (2σ²)
                    # Under Gaussian(μ_old≈h_old, σ²I): log π_old ≈ 0 (self-eval)
                    # So log ratio ≈ −||h_a_cur − h_a_old||² / (2σ²)
                    log_ratio_a = -((h_a_cur - h_a_old).pow(2).sum(-1)) / (2 * sigma2)  # (B,)
                    log_ratio_b = -((h_b_cur - h_b_old).pow(2).sum(-1)) / (2 * sigma2)

                    ratio_a = log_ratio_a.exp()
                    ratio_b = log_ratio_b.exp()
                    adv_g   = advantages[g].detach()            # (B,)

                    def ppo_clip(ratio, adv):
                        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                        return -torch.min(ratio * adv, clipped * adv).mean()

                    total_pg = total_pg + 0.5 * (ppo_clip(ratio_a, adv_g) +
                                                  ppo_clip(ratio_b, adv_g))

                    # KL penalty: squared distance between current and reference hidden states
                    kl_a = ((h_a_cur - h_a_ref[g].detach()).pow(2).sum(-1)).mean() / (2 * sigma2)
                    kl_b = ((h_b_cur - h_b_ref[g].detach()).pow(2).sum(-1)).mean() / (2 * sigma2)
                    total_kl = total_kl + 0.5 * (kl_a + kl_b)

                loss = (total_pg + kl_beta * total_kl) / G
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()

            # ── Logging and checkpointing ────────────────────────────────
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
                    self._save(encoder, swav_head, "rl_best")
                    print(f"  -> RL best ACC={best_acc:.4f}")

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

        if tcfg.get("warmup_epochs", 0) > 0:
            print("\n=== Phase 1: SwAV warmup ===")
            self._train_warmup(encoder, swav_head, images, labels, instruction)

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
