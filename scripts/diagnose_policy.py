#!/usr/bin/env python3
"""策略诊断：在固定的 warmup checkpoint 上扫描温度，秒级把"拍脑袋"变成"看数据"。

为什么需要它
------------
RL 的策略是 π(c|view) = softmax(cosine_similarity / T)。scores 是余弦相似度
∈ [-1, 1]（见 swav_head.py），所以温度 T 直接决定分布有多尖锐，这是纯数学，
不需要跑训练就能算：
  - T 太大 → 分布接近均匀 → 采样≈随机 → reward(两视图同簇)≈1/K → 策略梯度无信号
  - T 太小 → 分布≈one-hot → 没有探索空间

唯一耗时的步骤是"过一次大模型拿 scores"，缓存后扫所有温度都是秒级。

用法
----
    cd /home/yaner/kris/warehouse/VLC
    python scripts/diagnose_policy.py --config configs/vlm/swav_reinforce_ddp.yaml --n 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本无论从哪里启动都能 import vlc 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import yaml

from vlc.train.swav_rl_trainer import load_cifar10, TwoViewAug
from vlc.train.swav_reinforce_trainer import SwavReinforceTrainer


INSTRUCTION = (
    "What is the main object in this image? "
    "Choose from: airplane, automobile, bird, cat, deer, dog, "
    "frog, horse, ship, truck."
)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=128, help="诊断用图片数（越多越稳，越慢）")
    p.add_argument("--temps", default="0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    temps = [float(t) for t in args.temps.split(",")]

    trainer = SwavReinforceTrainer(cfg)
    dcfg, tcfg = cfg["data"], cfg["training"]
    out_dim = cfg["model"]["projection"].get("out_dim", 256)

    print("[diag] 加载 CIFAR-10 ...")
    images, labels = load_cifar10(dcfg["data_dir"], dcfg.get("max_per_class", 50),
                                  dcfg.get("image_size", 64))
    images = images[: args.n]

    encoder = trainer._build_encoder()
    swav_head = trainer._build_swav_head(out_dim, tcfg["k"])
    init_ckpt = tcfg.get("init_checkpoint")
    if init_ckpt and Path(init_ckpt).exists():
        trainer._load_checkpoint(encoder, swav_head, init_ckpt)
    else:
        print(f"[diag][警告] init_checkpoint 不存在: {init_ckpt}，用随机初始化的头。")
    encoder.backbone.eval()
    swav_head.normalize_prototypes()

    aug = TwoViewAug(dcfg.get("image_size", 64))
    encode_bs = dcfg.get("encode_batch_size", 8)

    print(f"[diag] 对 {len(images)} 张图做 two-view 增广并前向（唯一耗时步骤，缓存 scores）...")
    sa_list, sb_list = [], []
    for i in range(0, len(images), encode_bs):
        chunk = images[i:i + encode_bs]
        va, vb = zip(*[aug(im) for im in chunk])
        za = encoder.encode_batch(INSTRUCTION, list(va))
        zb = encoder.encode_batch(INSTRUCTION, list(vb))
        sa_list.append(swav_head(za).float().cpu())
        sb_list.append(swav_head(zb).float().cpu())
    sa = torch.cat(sa_list, 0)   # (N, K) 余弦相似度
    sb = torch.cat(sb_list, 0)

    K = sa.shape[1]
    rand_base = 1.0 / K
    lnK = torch.log(torch.tensor(float(K))).item()
    argmax_agree = (sa.argmax(1) == sb.argmax(1)).float().mean().item()

    print(f"\n[diag] scores 范围: [{sa.min():.3f}, {sa.max():.3f}]  (余弦相似度，理论 [-1,1])")
    print(f"[diag] K={K}，随机基线 agreement = 1/K = {rand_base:.3f}")
    print(f"[diag] argmax 两视图一致率（评测口径，与 T 无关）= {argmax_agree:.3f}")
    print(f"[diag] 当前训练 reward≈0.10，看下面哪个 T 与之对应\n")

    def scan(label, A, B):
        print(f"\n=== {label} ===")
        print(f"{'T':>6} | {'H(pi)':>7} | {'H/lnK':>6} | {'E[agree]':>9} | {'agree/rand':>10} | {'max-prob':>8}")
        print("-" * 64)
        for T in temps:
            pa = F.softmax(A / T, dim=1)
            pb = F.softmax(B / T, dim=1)
            H = -(pa * pa.clamp_min(1e-9).log()).sum(1).mean().item()
            agree = (pa * pb).sum(1).mean().item()     # E[两视图独立采样落同簇]
            maxp = pa.max(1).values.mean().item()
            print(f"{T:>6.2f} | {H:>7.3f} | {H/lnK:>6.2f} | {agree:>9.3f} | "
                  f"{agree / rand_base:>8.2f}x | {maxp:>8.3f}")

    scan("原始 scores（当前实现）", sa, sb)

    # 对照：per-row z-score 标准化，消除"scores 动态范围太窄"的问题后再除温度
    sa_z = (sa - sa.mean(1, keepdim=True)) / (sa.std(1, keepdim=True) + 1e-6)
    sb_z = (sb - sb.mean(1, keepdim=True)) / (sb.std(1, keepdim=True) + 1e-6)
    scan("标准化 scores（per-row z-score 后再 /T）", sa_z, sb_z)

    print("\n[解读]")
    print("  * E[agree] 要明显高于随机基线(agree/rand >> 1)，策略梯度才有信号。")
    print("  * H/lnK≈1 → 分布近似均匀(采样≈随机)；≈0 → 无探索。建议落在 0.3~0.7。")
    print("  * 选 agree/rand 尽量大、同时 H/lnK 不低于 ~0.3 的那个 T。")


if __name__ == "__main__":
    main()
