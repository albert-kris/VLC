"""SwAV-REINFORCE 单卡整合训练入口。

两阶段
------
Phase 1  SwAV warmup —— 同图两增广 → Qwen 最后隐状态 → 投射层 → SwAV 交换预测
         损失，直接对 LoRA + 投射层 + 原型反向传播。
Phase 2  REINFORCE / PPO —— 把 softmax(原型相似度 / T) 当作 K 个簇上的随机策略，
         采样离散簇分配，奖励=两视图分配一致性（hard / soft / sinkhorn 码），
         梯度只走 log π(采样动作)，配 PPO 比率裁剪 + KL-to-init + 边际熵抗塌缩。

运行
----
    python main.py --config configs/vlm/swav_reinforce_ddp.yaml
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
import yaml
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from torch.distributions import Categorical
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from vlc.model.encoder import (
    build_encoder,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from vlc.model.swav_head import SwAVHead

INSTRUCTION = (
    "What is the main object in this image? "
    "Choose from: airplane, automobile, bird, cat, deer, dog, "
    "frog, horse, ship, truck."
)


# ---------------------------------------------------------------------------
# 数据 & 两视图增广
# ---------------------------------------------------------------------------

class TwoViewAug:
    """轻量两视图增广（翻转 + 随机裁剪 + 可选灰度/模糊）。"""

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def _aug(self, img: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = img.size
        scale = random.uniform(0.6, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        x0, y0 = random.randint(0, max(0, w - nw)), random.randint(0, max(0, h - nh))
        img = img.crop((x0, y0, x0 + nw, y0 + nh)).resize((self.size, self.size), Image.BILINEAR)
        if random.random() > 0.8:
            img = img.convert("L").convert("RGB")
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        return img

    def __call__(self, img: Image.Image):
        return self._aug(img), self._aug(img)


def load_cifar10(data_dir: str, n_per_class: int | None, image_size: int = 64):
    """读取 CIFAR-10 训练集（pickle 批文件），可按类下采样。返回 PIL 图列表 + 标签。"""
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
        keep = np.sort(np.concatenate(
            [np.where(labels == c)[0][:n_per_class] for c in range(10)]))
        imgs, labels = imgs[keep], labels[keep]
    images = [Image.fromarray(a).resize((image_size, image_size), Image.BILINEAR) for a in imgs]
    return images, labels


def build_eval_subset(labels: np.ndarray, n_per_class: int, seed: int = 42) -> np.ndarray:
    """固定每类子集索引，用于快速、可复现的评测。"""
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(int(labels.max()) + 1):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        n = min(n_per_class, len(idx))
        keep.append(rng.choice(idx, size=n, replace=False) if len(idx) > n else idx)
    return np.sort(np.concatenate(keep))


# ---------------------------------------------------------------------------
# 损失（SwAV / Sinkhorn）& 聚类评测
# ---------------------------------------------------------------------------

@torch.no_grad()
def sinkhorn(scores: torch.Tensor, epsilon: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp：把原型相似度转成等分软分配码（抗塌缩的等分约束）。"""
    Q = torch.exp(scores / epsilon).t()         # (K, B)
    Q = Q / Q.sum().clamp(min=1e-8)
    K, B = Q.shape
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-8) / K   # 行归一：每原型等质量
        Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-8) / B   # 列归一：每样本码和为 1
    return (Q * B).t()                           # (B, K)


def swav_loss(scores_a, scores_b, temperature=0.1, epsilon=0.05, n_iters=3):
    """交换预测损失：每个视图的 softmax 预测去匹配对面视图的 Sinkhorn 码。"""
    q_a = sinkhorn(scores_a, epsilon, n_iters)
    q_b = sinkhorn(scores_b, epsilon, n_iters)
    log_p_a = F.log_softmax(scores_a / temperature, dim=1)
    log_p_b = F.log_softmax(scores_b / temperature, dim=1)
    return -0.5 * ((q_b * log_p_a).sum(1).mean() + (q_a * log_p_b).sum(1).mean())


def cluster_acc(y_true, y_pred):
    """匈牙利最优匹配下的无监督聚类准确率。"""
    k = max(y_true.max(), y_pred.max()) + 1
    W = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        W[p, t] += 1
    ri, ci = linear_sum_assignment(W.max() - W)
    return W[ri, ci].sum() / len(y_true)


@torch.no_grad()
def eval_clustering(encoder, swav_head, images, labels, instruction, device,
                    encode_bs=8, show_progress=False):
    """用当前编码器 + 原型对图像做 argmax 分配，返回 (ACC, NMI, ARI)。"""
    encoder.backbone.eval()
    swav_head.normalize_prototypes()
    starts = range(0, len(images), encode_bs)
    if show_progress and len(images) > 200:
        starts = tqdm(starts, desc=f"eval {len(images)}", leave=False)
    pred = []
    for i in starts:
        z = encoder.encode_batch(instruction, images[i:i + encode_bs])
        pred.append(swav_head(z.to(device)).argmax(1).cpu())
    pred = torch.cat(pred).numpy()
    encoder.backbone.train()
    return (cluster_acc(labels, pred),
            normalized_mutual_info_score(labels, pred),
            adjusted_rand_score(labels, pred))


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------

class Trainer:
    """单卡 SwAV-REINFORCE 训练器（warmup → REINFORCE/PPO → 评测保存）。"""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.get("output_dir", "artifacts/vlm/swav_reinforce"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.eval_images = None
        self.eval_labels = None

    # ---- 构建 / 保存 / 加载 -------------------------------------------------

    def _build_encoder(self):
        mcfg = self.cfg["model"]
        model_id = mcfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        lora_path = mcfg.get("lora_path")
        p = mcfg.get("projection", {})
        in_dim, mid_dim, out_dim = p.get("in_dim", 2048), p.get("mid_dim", 512), p.get("out_dim", 256)
        if lora_path and Path(lora_path).exists():
            proj_pt = Path(lora_path).parent / "proj_head.pt"
            if proj_pt.exists():
                encoder = load_encoder_checkpoint(model_id, lora_path, str(proj_pt),
                                                  in_dim, mid_dim, out_dim)
            else:
                encoder = build_encoder(model_id, lora_path, in_dim, mid_dim, out_dim)
        else:
            encoder = build_encoder(model_id, None, in_dim, mid_dim, out_dim)
        if mcfg.get("gradient_checkpointing", True):
            encoder.backbone.gradient_checkpointing_enable()
        return encoder

    def _build_swav_head(self, out_dim, k):
        return SwAVHead(dim=out_dim, n_prototypes=k).to(self.device)

    def _save(self, encoder, swav_head, tag):
        save_encoder_checkpoint(encoder, str(self.out_dir / tag))
        torch.save(swav_head.state_dict(), str(self.out_dir / tag / "swav_head.pt"))

    def _load_checkpoint(self, encoder, swav_head, ckpt_dir):
        from peft import PeftModel
        ckpt = Path(ckpt_dir)
        if (ckpt / "proj_head.pt").exists():
            encoder.proj_head.load_state_dict(torch.load(
                ckpt / "proj_head.pt", map_location=encoder.device, weights_only=True))
        if (ckpt / "lora").exists():
            if isinstance(encoder.backbone, PeftModel):
                encoder.backbone.load_adapter(str(ckpt / "lora"), adapter_name="default")
            else:
                encoder.backbone = PeftModel.from_pretrained(
                    encoder.backbone, str(ckpt / "lora"), is_trainable=True)
        if (ckpt / "swav_head.pt").exists():
            swav_head.load_state_dict(torch.load(
                ckpt / "swav_head.pt", map_location=self.device, weights_only=True))
        print(f"[checkpoint] Loaded from {ckpt_dir}")

    def _kmeans_init_prototypes_chain(self, encoder, swav_head, images, instruction, k, max_new_tokens):
        """chain 模式：generate 后取 im_end 前一 token hidden → proj → KMeans 初始化原型。"""
        with torch.no_grad():
            encoder.backbone.eval()
            zs = []
            n = min(len(images), 200)
            for i in range(n):
                h = encoder.encode_chain_hidden_one(instruction, images[i], max_new_tokens)
                zs.append(encoder.proj_head(h.unsqueeze(0)).detach().cpu())
                if (i + 1) % 50 == 0:
                    print(f"[init] chain hidden {i + 1}/{n}", flush=True)
            z0 = torch.cat(zs, dim=0).float().numpy()
        km = KMeans(n_clusters=k, n_init=10, random_state=self.cfg.get("seed", 42)).fit(z0)
        centroids = torch.tensor(km.cluster_centers_, dtype=swav_head.prototypes.weight.dtype,
                                 device=self.device)
        swav_head.prototypes.weight.data.copy_(F.normalize(centroids, dim=1))
        encoder.backbone.train()
        print(f"[init] Prototypes from chain-hidden KMeans on {z0.shape[0]} embeddings.")

    def _kmeans_init_prototypes(self, encoder, swav_head, images, instruction, k, encode_bs):
        """用当前编码器的前若干张图嵌入做 KMeans，初始化 SwAV 原型。"""
        with torch.no_grad():
            encoder.backbone.eval()
            z0 = torch.cat([
                encoder.encode_batch(instruction, images[i:i + encode_bs]).detach().cpu()
                for i in range(0, min(len(images), 200), encode_bs)
            ], dim=0).float().numpy()
        km = KMeans(n_clusters=k, n_init=10, random_state=self.cfg.get("seed", 42)).fit(z0)
        centroids = torch.tensor(km.cluster_centers_, dtype=swav_head.prototypes.weight.dtype,
                                 device=self.device)
        swav_head.prototypes.weight.data.copy_(F.normalize(centroids, dim=1))
        encoder.backbone.train()
        print(f"[init] Prototypes from KMeans on {z0.shape[0]} embeddings.")

    def _log(self, rec):
        with open(self.out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- 策略 logits 辅助 ---------------------------------------------------

    @staticmethod
    def _standardize_scores(scores):
        # 余弦相似度挤在窄区间里时直接 /T 会让 softmax 近似均匀、温度失效；
        # per-row z-score 把每行拉到统一尺度，温度重新成为有效旋钮。仿射变换不改
        # argmax，故 ACC 评测口径不受影响。
        return (scores - scores.mean(1, keepdim=True)) / (scores.std(1, keepdim=True) + 1e-6)

    def _logits_from_hidden(self, encoder, swav_head, h, temperature):
        """h:(B,hidden) → 策略 logits (B,K)，梯度经过 proj + 原型。"""
        scores = swav_head(encoder.proj_head(h))
        if self.cfg["training"].get("policy_standardize", True):
            scores = self._standardize_scores(scores)
        return scores / temperature

    @torch.no_grad()
    def _ref_logits_from_hidden(self, ref_proj, ref_proto, h, temperature):
        """warmup 初始策略（参考策略）的 logits，用于 KL-to-init。"""
        scores = ref_proj(h) @ F.normalize(ref_proto, dim=1).t()
        if self.cfg["training"].get("policy_standardize", True):
            scores = self._standardize_scores(scores)
        return scores / temperature

    # ---- Phase 1: SwAV warmup（直接反向传播）-------------------------------

    def _train_warmup(self, encoder, swav_head, images, instruction):
        tcfg, dcfg = self.cfg["training"], self.cfg["data"]
        warmup_epochs = tcfg.get("warmup_epochs", 5)
        lr = float(tcfg.get("lr", 1e-4))
        acc_floor = float(tcfg.get("warmup_acc_floor", 0.60))
        early_stop_on_drop = tcfg.get("warmup_early_stop_on_drop", True)
        batch_size = dcfg.get("images_per_batch", 16)
        steps_per_epoch = dcfg.get("steps_per_epoch", 20)
        grad_accum = tcfg.get("grad_accum_steps", 4)
        max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        temperature = float(tcfg.get("warmup_temperature", tcfg.get("temperature", 0.1)))
        epsilon = float(tcfg.get("sinkhorn_epsilon", 0.05))
        sink_iters = int(tcfg.get("sinkhorn_iters", 3))
        encode_bs = dcfg.get("encode_batch_size", 8)
        k = tcfg["k"]
        n = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))
        rng = random.Random(self.cfg.get("seed", 42))

        self._kmeans_init_prototypes(encoder, swav_head, images, instruction, k, encode_bs)

        if tcfg.get("warmup_freeze_backbone", False):
            for p in encoder.backbone.parameters():
                p.requires_grad_(False)
            print("[warmup] Backbone FROZEN — only proj_head + prototypes trainable.")

        lr_proj = float(tcfg.get("lr_proj", tcfg.get("lr", 1e-4)))
        lr_proto = float(tcfg.get("lr_proto", tcfg.get("lr", 1e-4)))
        param_groups = []
        if not tcfg.get("warmup_freeze_backbone", False):
            param_groups.append({
                "params": [p for p in encoder.backbone.parameters() if p.requires_grad],
                "lr": float(tcfg.get("lr_enc", tcfg.get("lr", 1e-5)))})
        param_groups += [
            {"params": list(encoder.proj_head.parameters()), "lr": lr_proj},
            {"params": list(swav_head.parameters()), "lr": lr_proto},
        ]
        trainable = [p for g in param_groups for p in g["params"]]
        optimizer = AdamW(param_groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))
        total_steps = max(warmup_epochs * steps_per_epoch // grad_accum, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

        best_acc, prev_acc = -1.0, None
        show_eval = len(self.eval_images) > 200
        for epoch in range(warmup_epochs):
            acc, nmi, ari = eval_clustering(encoder, swav_head, self.eval_images,
                                            self.eval_labels, instruction, self.device,
                                            encode_bs, show_progress=show_eval)
            print(f"[warmup {epoch}/{warmup_epochs}] ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
            self._log({"phase": "warmup", "epoch": epoch, "acc": acc, "nmi": nmi, "ari": ari})

            if epoch > 0 and acc < acc_floor:
                print(f"[warmup] ACC {acc:.4f} < floor {acc_floor:.2f} — early stop.")
                break
            if early_stop_on_drop and prev_acc is not None and acc < prev_acc:
                print(f"[warmup] ACC dropped {prev_acc:.4f} -> {acc:.4f} — early stop.")
                break
            prev_acc = acc
            if acc > best_acc:
                best_acc = acc
                self._save(encoder, swav_head, "warmup_best")
                print(f"  -> warmup best ACC={best_acc:.4f}")

            encoder.backbone.train()
            epoch_loss, nb = 0.0, 0
            optimizer.zero_grad()
            pbar = tqdm(range(steps_per_epoch), desc=f"Warmup {epoch + 1}/{warmup_epochs}")
            for step_i in pbar:
                idx = rng.sample(range(n), min(batch_size, n))
                va, vb = zip(*[aug(images[i]) for i in idx])
                try:
                    swav_head.normalize_prototypes()
                    sa = swav_head(encoder.encode_batch(instruction, list(va)))
                    sb = swav_head(encoder.encode_batch(instruction, list(vb)))
                    loss = swav_loss(sa, sb, temperature, epsilon, sink_iters)
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
                pbar.set_postfix(loss=f"{epoch_loss / max(nb, 1):.4f}")
            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        self._save(encoder, swav_head, "warmup_final")
        print(f"[warmup] Done. Best ACC={best_acc:.4f}")

    # ---- Phase 2: REINFORCE / PPO ------------------------------------------

    def _train_reinforce(self, encoder, swav_head, images, instruction):
        tcfg, dcfg = self.cfg["training"], self.cfg["data"]
        rl_iters = int(tcfg.get("rl_iterations", 300))
        batch_size = int(dcfg.get("images_per_batch", 8))
        G = int(tcfg.get("group_size", 8))
        temperature = float(tcfg.get("temperature", 0.5))
        eval_every = int(tcfg.get("rl_eval_every", 5))
        encode_bs = int(dcfg.get("encode_batch_size", 8))
        max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        patience = int(tcfg.get("early_stop_patience", 0))

        lr = float(tcfg.get("rl_lr", 1e-5))
        lr_enc = float(tcfg.get("rl_lr_enc", tcfg.get("lr_enc", lr)))
        lr_proj = float(tcfg.get("rl_lr_proj", tcfg.get("lr_proj", lr)))
        lr_proto = float(tcfg.get("rl_lr_proto", tcfg.get("lr_proto", lr)))

        ppo_epochs = int(tcfg.get("ppo_epochs", 4))
        ppo_clip = float(tcfg.get("ppo_clip", 0.2))
        balance_coef = float(tcfg.get("balance_coef", 1.0))
        ent_coef = float(tcfg.get("ent_coef", 0.01))
        kl_coef = float(tcfg.get("kl_coef", 0.1))
        reward_mode = str(tcfg.get("reward_mode", "hard")).lower()
        normalize_adv = bool(tcfg.get("normalize_adv", True))
        adv_eps = float(tcfg.get("adv_eps", 1e-2))
        sink_eps = float(tcfg.get("sinkhorn_epsilon", 0.05))
        sink_iters = int(tcfg.get("sinkhorn_iters", 3))

        n = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))
        rng = random.Random(self.cfg.get("seed", 42) + 7)

        # LoRA lr<=0 时冻结 backbone，每次迭代只算一次 Qwen 前向并缓存隐状态，
        # 之后多次廉价 PPO epoch 只更新投射层 + 原型，并可免费启用 KL-to-init。
        freeze_lora = lr_enc <= 0
        if freeze_lora:
            for p in encoder.backbone.parameters():
                p.requires_grad_(False)
            encoder.backbone.eval()
            ppo_epochs = max(ppo_epochs, 1)
            print("[REINFORCE] LoRA frozen — cache hidden states, train proj + prototypes.")
        else:
            print("[REINFORCE] LoRA trainable — hidden states recomputed per epoch.")

        for p in swav_head.parameters():
            p.requires_grad_(True)
        param_groups = []
        if not freeze_lora:
            param_groups.append({
                "params": [p for p in encoder.backbone.parameters() if p.requires_grad],
                "lr": lr_enc})
        param_groups += [
            {"params": list(encoder.proj_head.parameters()), "lr": lr_proj},
            {"params": list(swav_head.parameters()), "lr": lr_proto},
        ]
        trainable = [p for g in param_groups for p in g["params"]]
        optimizer = AdamW(param_groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))

        # warmup 初始策略快照，用作 KL-to-init 的参考策略。
        ref_proj = copy.deepcopy(encoder.proj_head).eval()
        for p in ref_proj.parameters():
            p.requires_grad_(False)
        ref_proto = swav_head.prototypes.weight.detach().clone()
        use_kl = freeze_lora and kl_coef > 0.0

        print(f"[REINFORCE] G={G}, ppo_epochs={ppo_epochs}, clip={ppo_clip}, "
              f"T={temperature}, reward={reward_mode}, balance={balance_coef}, "
              f"ent={ent_coef}, kl={kl_coef if use_kl else 0.0}")
        if patience:
            print(f"[REINFORCE] early stop: {patience} eval points w/o ACC improvement")

        best_acc, no_improve = -1.0, 0
        for it in tqdm(range(1, rl_iters + 1), desc="REINFORCE"):
            idx = rng.sample(range(n), min(batch_size, n))
            va, vb = zip(*[aug(images[i]) for i in idx])
            va, vb = list(va), list(vb)

            # 冻结路径：一次性缓存隐状态
            if freeze_lora:
                with torch.no_grad():
                    h_a = encoder.encode_hidden_batch(instruction, va)
                    h_b = encoder.encode_hidden_batch(instruction, vb)
            else:
                h_a = h_b = None

            # --- 采样阶段（旧策略，no grad）：动作 + 优势 ---
            with torch.no_grad():
                swav_head.normalize_prototypes()
                if freeze_lora:
                    logits_a0 = self._logits_from_hidden(encoder, swav_head, h_a, temperature)
                    logits_b0 = self._logits_from_hidden(encoder, swav_head, h_b, temperature)
                    if use_kl:
                        ref_la = F.log_softmax(self._ref_logits_from_hidden(
                            ref_proj, ref_proto, h_a, temperature), dim=1)
                        ref_lb = F.log_softmax(self._ref_logits_from_hidden(
                            ref_proj, ref_proto, h_b, temperature), dim=1)
                        ref_pa, ref_pb = ref_la.exp(), ref_lb.exp()
                else:
                    hh_a = encoder.encode_hidden_batch(instruction, va)
                    hh_b = encoder.encode_hidden_batch(instruction, vb)
                    logits_a0 = self._logits_from_hidden(encoder, swav_head, hh_a, temperature)
                    logits_b0 = self._logits_from_hidden(encoder, swav_head, hh_b, temperature)

                pa0, pb0 = F.softmax(logits_a0, dim=1), F.softmax(logits_b0, dim=1)
                dist_a, dist_b = Categorical(probs=pa0), Categorical(probs=pb0)
                ca, cb = dist_a.sample((G,)), dist_b.sample((G,))      # (G, B)
                old_logp = dist_a.log_prob(ca) + dist_b.log_prob(cb)   # (G, B)

                if reward_mode == "sinkhorn":
                    # 用对面视图的 sinkhorn 等分码作为 reward 目标：列归一化强制每个
                    # prototype 在 batch 内拿到 ~B/K 质量，塌缩到单簇会被自动压低码
                    # 质量，抗塌缩内建于 reward；码 detached，梯度仍只走 log π。
                    q_a = sinkhorn(logits_a0 * temperature, sink_eps, sink_iters)
                    q_b = sinkhorn(logits_b0 * temperature, sink_eps, sink_iters)
                    qb_at_ca = q_b.unsqueeze(0).expand(G, -1, -1).gather(2, ca.unsqueeze(-1)).squeeze(-1)
                    qa_at_cb = q_a.unsqueeze(0).expand(G, -1, -1).gather(2, cb.unsqueeze(-1)).squeeze(-1)
                    reward = 0.5 * (qb_at_ca + qa_at_cb)
                elif reward_mode == "soft":
                    pb_at_ca = pb0.unsqueeze(0).expand(G, -1, -1).gather(2, ca.unsqueeze(-1)).squeeze(-1)
                    pa_at_cb = pa0.unsqueeze(0).expand(G, -1, -1).gather(2, cb.unsqueeze(-1)).squeeze(-1)
                    reward = 0.5 * (pb_at_ca + pa_at_cb)
                else:  # hard agreement
                    reward = (ca == cb).float()

                mean_r = reward.mean(0, keepdim=True)
                if normalize_adv:
                    adv = (reward - mean_r) / (reward.std(0, keepdim=True) + adv_eps)
                else:
                    adv = reward - mean_r
                adv = adv.detach()
                reward_mean = reward.mean().item()

            # --- PPO 更新 ---
            if not freeze_lora:
                encoder.backbone.train()
            last = {}
            for _ in range(ppo_epochs):
                swav_head.normalize_prototypes()
                if freeze_lora:
                    logits_a = self._logits_from_hidden(encoder, swav_head, h_a, temperature)
                    logits_b = self._logits_from_hidden(encoder, swav_head, h_b, temperature)
                else:
                    logits_a = self._logits_from_hidden(
                        encoder, swav_head, encoder.encode_hidden_batch(instruction, va), temperature)
                    logits_b = self._logits_from_hidden(
                        encoder, swav_head, encoder.encode_hidden_batch(instruction, vb), temperature)

                logp_a_all = F.log_softmax(logits_a, dim=1)
                logp_b_all = F.log_softmax(logits_b, dim=1)
                la = logp_a_all.gather(1, ca.t()).t()      # (G, B)
                lb = logp_b_all.gather(1, cb.t()).t()
                new_logp = la + lb

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                pa, pb = F.softmax(logits_a, dim=1), F.softmax(logits_b, dim=1)
                p_bar = 0.5 * (pa.mean(0) + pb.mean(0))
                marg_entropy = -(p_bar * p_bar.clamp_min(1e-8).log()).sum()
                samp_entropy = -0.5 * (
                    (pa * pa.clamp_min(1e-8).log()).sum(1)
                    + (pb * pb.clamp_min(1e-8).log()).sum(1)).mean()

                loss = policy_loss - balance_coef * marg_entropy - ent_coef * samp_entropy
                if use_kl:
                    kl = 0.5 * ((ref_pa * (ref_la - logp_a_all)).sum(1).mean()
                                + (ref_pb * (ref_lb - logp_b_all)).sum(1).mean())
                    loss = loss + kl_coef * kl
                    last["kl"] = kl.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                last.update(policy_loss=policy_loss.item(),
                            marg_entropy=marg_entropy.item(),
                            ratio=ratio.mean().item())

            # --- 评测 / 保存 ---
            if it % eval_every == 0:
                acc, nmi, ari = eval_clustering(encoder, swav_head, self.eval_images,
                                                self.eval_labels, instruction, self.device, encode_bs)
                msg = (f"[REINFORCE it={it:4d}] reward={reward_mean:.4f} "
                       f"adv|pg|={last.get('policy_loss', 0):.4f} "
                       f"H(marg)={last.get('marg_entropy', 0):.3f} "
                       f"ratio={last.get('ratio', 1):.3f} "
                       f"ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
                if use_kl:
                    msg += f" KL={last.get('kl', 0):.4f}"
                print(msg)
                rec = {"phase": "reinforce", "iter": it, "reward": reward_mean,
                       "acc": acc, "nmi": nmi, "ari": ari,
                       "marg_entropy": last.get("marg_entropy", 0.0),
                       "ratio": last.get("ratio", 1.0)}
                if use_kl:
                    rec["kl"] = last.get("kl", 0.0)
                self._log(rec)

                if acc > best_acc:
                    best_acc, no_improve = acc, 0
                    self._save(encoder, swav_head, "rl_best")
                    print(f"  -> RL best ACC={best_acc:.4f}")
                else:
                    no_improve += 1
                    if patience and no_improve >= patience:
                        print(f"[REINFORCE] early stop: {patience} pts w/o improvement, "
                              f"best ACC={best_acc:.4f}")
                        break

        print(f"[REINFORCE] Done. Best ACC={best_acc:.4f}")

    # ---- Phase 2 (chain): DeepSeek 式 GRPO ----------------------------------

    @torch.no_grad()
    def _eval_clustering_chain(self, encoder, swav_head, images, labels, instruction,
                               max_new_tokens, show_progress=False):
        """评测口径与 chain 训练一致：贪心生成链 → 末态 hidden → 原型 argmax。"""
        encoder.backbone.eval()
        swav_head.normalize_prototypes()
        it = images
        if show_progress and len(images) > 200:
            it = tqdm(images, desc=f"eval-chain {len(images)}", leave=False)
        pred = []
        for img in it:
            inp, L, gen = encoder.generate_chains(
                instruction, img, 1, max_new_tokens, do_sample=False)
            _, h = encoder.score_chains(inp, L, gen)          # (1, hidden)
            z = encoder.proj_head(h)
            pred.append(int(swav_head(z).argmax(1).item()))
        encoder.backbone.train()
        pred = np.array(pred)
        return (cluster_acc(labels, pred),
                normalized_mutual_info_score(labels, pred),
                adjusted_rand_score(labels, pred))

    def _train_reinforce_chain(self, encoder, swav_head, images, instruction):
        """DeepSeek 式：动作=生成的推理链 token（训 LoRA）+ 从链末态采样的簇
        （训 proj_head + 原型）；reward=两视图配对链的簇一致性；两个 log p 共享
        同一组内归一化优势做联合 REINFORCE。全程 reward 驱动，无额外可导 loss。"""
        tcfg, dcfg = self.cfg["training"], self.cfg["data"]
        rl_iters = int(tcfg.get("rl_iterations", 300))
        batch_size = int(dcfg.get("images_per_batch", 4))
        G = int(tcfg.get("group_size", 4))
        temperature = float(tcfg.get("temperature", 0.5))            # 簇策略温度
        gen_temperature = float(tcfg.get("gen_temperature", 1.0))    # 生成采样温度
        max_new_tokens = int(tcfg.get("max_new_tokens", 20))
        eval_every = int(tcfg.get("rl_eval_every", 5))
        max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        patience = int(tcfg.get("early_stop_patience", 0))
        reward_mode = str(tcfg.get("reward_mode", "hard")).lower()
        normalize_adv = bool(tcfg.get("normalize_adv", True))
        adv_eps = float(tcfg.get("adv_eps", 1e-2))
        ent_coef = float(tcfg.get("ent_coef", 0.01))
        balance_coef = float(tcfg.get("balance_coef", 1.0))

        lr_enc = float(tcfg.get("rl_lr_enc", tcfg.get("lr_enc", 1e-5)))
        lr_proj = float(tcfg.get("rl_lr_proj", tcfg.get("lr_proj", 1e-5)))
        lr_proto = float(tcfg.get("rl_lr_proto", tcfg.get("lr_proto", 1e-5)))

        n = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))
        rng = random.Random(self.cfg.get("seed", 42) + 11)

        # 生成链是策略本体，LoRA 必须可训练并处于 train 模式
        encoder.backbone.train()
        for p in swav_head.parameters():
            p.requires_grad_(True)

        param_groups = [
            {"params": [p for p in encoder.backbone.parameters() if p.requires_grad],
             "lr": lr_enc},
            {"params": list(encoder.proj_head.parameters()), "lr": lr_proj},
            {"params": list(swav_head.parameters()), "lr": lr_proto},
        ]
        trainable = [p for g in param_groups for p in g["params"]]
        optimizer = AdamW(param_groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))

        print(f"[GRPO-chain] G={G}, gen_T={gen_temperature}, max_new={max_new_tokens}, "
              f"clus_T={temperature}, reward={reward_mode}, ent={ent_coef}, "
              f"balance={balance_coef}")
        if patience:
            print(f"[GRPO-chain] early stop: {patience} eval points w/o ACC improvement")

        show_eval = len(self.eval_images) > 200
        best_acc, no_improve = -1.0, 0
        pbar = tqdm(range(1, rl_iters + 1), desc="GRPO-chain")
        for it in pbar:
            idx = rng.sample(range(n), min(batch_size, n))
            optimizer.zero_grad()
            batch_reward, batch_pg, batch_marg, n_img = 0.0, 0.0, 0.0, 0
            for im in idx:
                ia, ib = aug(images[im])
                swav_head.normalize_prototypes()

                # --- rollout：两视图各采 G 条推理链（no_grad）---
                inp_a, La, gen_a = encoder.generate_chains(
                    instruction, ia, G, max_new_tokens, gen_temperature)
                inp_b, Lb, gen_b = encoder.generate_chains(
                    instruction, ib, G, max_new_tokens, gen_temperature)

                # --- 带梯度重打分：token logp（训 LoRA）+ 末态 hidden ---
                logp_chain_a, h_a = encoder.score_chains(inp_a, La, gen_a)   # (G,),(G,hid)
                logp_chain_b, h_b = encoder.score_chains(inp_b, Lb, gen_b)

                # --- 簇策略：末态 hidden → proj → swav → softmax ---
                logits_a = self._logits_from_hidden(encoder, swav_head, h_a, temperature)
                logits_b = self._logits_from_hidden(encoder, swav_head, h_b, temperature)
                logp_a_all = F.log_softmax(logits_a, dim=1)
                logp_b_all = F.log_softmax(logits_b, dim=1)
                pa, pb = logp_a_all.exp(), logp_b_all.exp()

                # --- 采样簇动作 + reward + 组内优势（no_grad）---
                with torch.no_grad():
                    ca = Categorical(probs=pa.detach()).sample()    # (G,)
                    cb = Categorical(probs=pb.detach()).sample()
                    if reward_mode == "soft":
                        reward = 0.5 * (
                            pb.detach().gather(1, ca.view(-1, 1)).squeeze(1)
                            + pa.detach().gather(1, cb.view(-1, 1)).squeeze(1))
                    else:  # hard：两视图配对链是否落到同一簇
                        reward = (ca == cb).float()
                    mean_r = reward.mean()
                    if normalize_adv:
                        adv = (reward - mean_r) / (reward.std() + adv_eps)
                    else:
                        adv = reward - mean_r
                    adv = adv.detach()

                # 簇动作 log p（可导，训 proj + 原型）
                logp_clus = (logp_a_all.gather(1, ca.view(-1, 1)).squeeze(1)
                             + logp_b_all.gather(1, cb.view(-1, 1)).squeeze(1))
                # 链 token log p（可导，训 LoRA）
                logp_chain = logp_chain_a + logp_chain_b
                pg = -(adv * (logp_chain + logp_clus)).mean()

                # 抗塌缩：组内边际熵 + 采样熵
                p_bar = 0.5 * (pa.mean(0) + pb.mean(0))
                marg_entropy = -(p_bar * p_bar.clamp_min(1e-8).log()).sum()
                samp_entropy = -0.5 * (
                    (pa * pa.clamp_min(1e-8).log()).sum(1)
                    + (pb * pb.clamp_min(1e-8).log()).sum(1)).mean()
                loss = pg - balance_coef * marg_entropy - ent_coef * samp_entropy
                (loss / len(idx)).backward()

                batch_reward += reward.mean().item()
                batch_pg += pg.item()
                batch_marg += marg_entropy.item()
                n_img += 1

            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            reward_mean = batch_reward / max(n_img, 1)

            if it % eval_every == 0:
                acc, nmi, ari = self._eval_clustering_chain(
                    encoder, swav_head, self.eval_images, self.eval_labels,
                    instruction, max_new_tokens, show_progress=show_eval)
                msg = (f"[GRPO-chain it={it:4d}] reward={reward_mean:.4f} "
                       f"|pg|={batch_pg / max(n_img, 1):.4f} "
                       f"H(marg)={batch_marg / max(n_img, 1):.3f} "
                       f"ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
                pbar.write(msg)
                pbar.set_postfix(acc=f"{acc:.4f}", reward=f"{reward_mean:.3f}", refresh=False)
                self._log({"phase": "grpo_chain", "iter": it, "reward": reward_mean,
                           "acc": acc, "nmi": nmi, "ari": ari,
                           "marg_entropy": batch_marg / max(n_img, 1)})
                if acc > best_acc:
                    best_acc, no_improve = acc, 0
                    self._save(encoder, swav_head, "rl_best")
                    pbar.write(f"  -> GRPO-chain best ACC={best_acc:.4f}")
                else:
                    no_improve += 1
                    if patience and no_improve >= patience:
                        pbar.write(f"[GRPO-chain] early stop: {patience} pts w/o improvement, "
                                   f"best ACC={best_acc:.4f}")
                        break

        print(f"[GRPO-chain] Done. Best ACC={best_acc:.4f}")

    # ---- 编排 ---------------------------------------------------------------

    def train(self) -> None:
        tcfg, dcfg = self.cfg["training"], self.cfg["data"]
        k = tcfg["k"]
        out_dim = self.cfg["model"]["projection"].get("out_dim", 256)

        print("[trainer] Loading CIFAR-10 ...")
        images, labels = load_cifar10(dcfg["data_dir"], dcfg.get("max_per_class"),
                                      dcfg.get("image_size", 64))
        print(f"[trainer] {len(images)} images.")

        eval_n = dcfg.get("eval_max_per_class")
        if eval_n is None:
            self.eval_images, self.eval_labels = images, labels
        else:
            keep = build_eval_subset(labels, int(eval_n), self.cfg.get("seed", 42))
            self.eval_images = [images[i] for i in keep]
            self.eval_labels = labels[keep]
            print(f"[trainer] Eval subset: {len(self.eval_images)} images "
                  f"({eval_n}/class); training on {len(images)}.")

        encoder = self._build_encoder()
        swav_head = self._build_swav_head(out_dim, k)

        init_ckpt = tcfg.get("init_checkpoint") or self.cfg["model"].get("lora_path")
        rl_mode = str(tcfg.get("rl_mode", "cluster")).lower()
        max_new_tokens = int(tcfg.get("max_new_tokens", 24))

        if init_ckpt and Path(init_ckpt).exists():
            self._load_checkpoint(encoder, swav_head, init_ckpt)
        elif rl_mode == "chain":
            self._kmeans_init_prototypes_chain(
                encoder, swav_head, images, INSTRUCTION, k, max_new_tokens)
        elif tcfg.get("warmup_epochs", 0) == 0:
            self._kmeans_init_prototypes(encoder, swav_head, images, INSTRUCTION,
                                         k, dcfg.get("encode_batch_size", 8))

        if rl_mode != "chain" and tcfg.get("warmup_epochs", 0) > 0:
            print("\n=== Phase 1: SwAV warmup ===")
            self._train_warmup(encoder, swav_head, images, INSTRUCTION)
            best_path = self.out_dir / "warmup_best"
            if best_path.exists() and tcfg.get("rl_from_warmup_best", True):
                self._load_checkpoint(encoder, swav_head, str(best_path))
                print("[trainer] Reloaded warmup_best before RL.")

        if tcfg.get("rl_iterations", 0) > 0:
            if rl_mode == "chain":
                print("\n=== Phase 2: GRPO-chain (DeepSeek 式生成链) ===")
                self._train_reinforce_chain(encoder, swav_head, images, INSTRUCTION)
            else:
                print("\n=== Phase 2: REINFORCE / PPO (簇分配动作) ===")
                self._train_reinforce(encoder, swav_head, images, INSTRUCTION)

        if rl_mode == "chain" and tcfg.get("rl_iterations", 0) > 0:
            acc, nmi, ari = self._eval_clustering_chain(
                encoder, swav_head, self.eval_images, self.eval_labels,
                INSTRUCTION, int(tcfg.get("max_new_tokens", 20)))
        else:
            acc, nmi, ari = eval_clustering(encoder, swav_head, self.eval_images,
                                            self.eval_labels, INSTRUCTION, self.device,
                                            dcfg.get("encode_batch_size", 8))
        print(f"\n[FINAL] ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
        self._save(encoder, swav_head, "final")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
