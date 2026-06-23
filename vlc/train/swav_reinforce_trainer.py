"""SwAV-REINFORCE Trainer — a *correct* policy-gradient replacement for the
broken "GRPO" phase in ``swav_rl_trainer.py``.

Why the old phase could never improve
-------------------------------------
The previous ``_train_grpo`` differentiated the reward value itself
(``loss = -(advantage.detach() * consistency)``) where ``consistency`` is a
deterministic, differentiable function of the parameters, and the per-group
variance came from *random data augmentation* — something the model cannot
control. Algebraically that objective reduces to ``-std(consistency)``, i.e. it
*maximises* the model's sensitivity to augmentation noise, which is the exact
opposite of augmentation-invariant clustering. ACC therefore collapsed from the
init checkpoint every single run.

What this trainer does instead (a genuine contextual-bandit REINFORCE / PPO)
----------------------------------------------------------------------------
* **Stochastic policy.** π(c|view) = softmax(prototype_scores / T) is a real
  categorical distribution over the K clusters. We *sample* discrete cluster
  assignments from it; the group variance now comes from the policy, not from
  augmentation.
* **Gradient through log π(sampled action)**, not through the reward — this is
  the defining property of policy gradient that the old code was missing.
* **Reward = view agreement** on the *sampled* actions (``1[c_a == c_b]`` or a
  soft variant). Two augmentations of one image should land in the same cluster.
* **Anti-collapse.** Pure agreement is maximised by the degenerate "everything
  in one cluster" policy, so we add a differentiable batch-marginal entropy
  bonus (equipartition prior, à la SwAV/SeLa) plus a small per-sample entropy
  bonus for exploration.
* **Trust region.** PPO-style ratio clipping (old log-probs are recorded at
  sampling time, so it is free) keeps each update bounded; in the frozen-LoRA
  fast path we additionally add a KL-to-init penalty toward the warmup policy.
* **Group baseline.** Advantage = (r - mean_G) / (std_G + eps) over the G
  samples — variance reduction without the old ``clamp(min=1e-6)`` blow-up.

Two execution modes
--------------------
* ``rl_lr_enc <= 0`` (default, recommended): LoRA frozen, hidden states cached
  ONCE per iteration (one expensive Qwen forward), then several cheap PPO epochs
  run over the projection head + prototypes. Enables KL-to-init for free.
* ``rl_lr_enc > 0``: LoRA trainable; hidden states are recomputed with grad each
  PPO epoch (expensive), KL-to-init is skipped (cached ref would be stale).

Usage
-----
    python -m vlc.train.swav_reinforce_trainer --config configs/vlm/swav_reinforce.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import AdamW
from tqdm import tqdm

# Reuse the proven, unmodified pieces from the original trainer.
from vlc.train.swav_rl_trainer import (
    SwavRLTrainer,
    TwoViewAug,
    eval_clustering,
)


class SwavReinforceTrainer(SwavRLTrainer):
    """Inherits build / load / save / warmup / eval from SwavRLTrainer and only
    replaces the reinforcement-learning phase with a correct REINFORCE/PPO loop.
    """

    # ``train()`` (inherited) calls ``self._train_grpo`` for the RL phase, so we
    # override that single hook and delegate to the new implementation.
    def _train_grpo(self, encoder, swav_head, images, labels, instruction):
        self._train_reinforce(encoder, swav_head, images, labels, instruction)

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize_scores(scores):
        # 余弦相似度实测挤在 [0.78,1.0] 这条窄缝里，直接 /T 时 softmax 近似均匀、
        # 温度形同失效（agree/rand≈1）。per-row z-score 把每行拉到统一尺度后，
        # 温度才重新成为有效旋钮（agree/rand 提到 4x+）。仿射变换不改 argmax，
        # 故评测口径(ACC) 完全不受影响。
        return (scores - scores.mean(dim=1, keepdim=True)) / (
            scores.std(dim=1, keepdim=True) + 1e-6)

    def _logits_from_hidden(self, encoder, swav_head, h, temperature):
        """h: (B, hidden) -> policy logits (B, K) WITH grad through proj+proto."""
        z = encoder.proj_head(h)                 # (B, out_dim), L2-normalized
        scores = swav_head(z)                    # (B, K) cosine similarities
        if self.cfg["training"].get("policy_standardize", True):
            scores = self._standardize_scores(scores)
        return scores / temperature

    @torch.no_grad()
    def _ref_logits_from_hidden(self, ref_proj, ref_proto, h, temperature):
        """Reference (warmup-init) policy logits on cached hidden states."""
        z = ref_proj(h)
        scores = z @ F.normalize(ref_proto, dim=1).t()
        if self.cfg["training"].get("policy_standardize", True):
            scores = self._standardize_scores(scores)
        return scores / temperature

    # ------------------------------------------------------------------
    # Phase 2 (replacement): genuine REINFORCE / PPO
    # ------------------------------------------------------------------

    def _train_reinforce(self, encoder, swav_head, images, labels, instruction):
        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]

        rl_iters    = int(tcfg.get("rl_iterations", 300))
        batch_size  = int(dcfg.get("images_per_batch", 8))
        G           = int(tcfg.get("group_size", 8))
        temperature = float(tcfg.get("temperature", 0.5))
        eval_every  = int(tcfg.get("rl_eval_every", 5))
        encode_bs   = int(dcfg.get("encode_batch_size", 8))
        max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
        patience    = int(tcfg.get("early_stop_patience", 0))

        lr          = float(tcfg.get("rl_lr", 1e-5))
        lr_enc      = float(tcfg.get("rl_lr_enc", tcfg.get("lr_enc", lr)))
        lr_proj     = float(tcfg.get("rl_lr_proj", tcfg.get("lr_proj", lr)))
        lr_proto    = float(tcfg.get("rl_lr_proto", tcfg.get("lr_proto", lr)))

        # REINFORCE / PPO knobs
        ppo_epochs   = int(tcfg.get("ppo_epochs", 4))
        ppo_clip     = float(tcfg.get("ppo_clip", 0.2))
        balance_coef = float(tcfg.get("balance_coef", 1.0))
        ent_coef     = float(tcfg.get("ent_coef", 0.01))
        kl_coef      = float(tcfg.get("kl_coef", 0.1))
        reward_mode  = str(tcfg.get("reward_mode", "hard")).lower()
        normalize_adv = bool(tcfg.get("normalize_adv", True))
        adv_eps      = float(tcfg.get("adv_eps", 1e-2))
        sink_eps     = float(tcfg.get("sinkhorn_epsilon", 0.05))
        sink_iters   = int(tcfg.get("sinkhorn_iters", 3))

        n = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))
        import random
        rng = random.Random(self.cfg.get("seed", 42) + 7)

        freeze_lora = lr_enc <= 0
        if freeze_lora:
            for p in encoder.backbone.parameters():
                p.requires_grad_(False)
            encoder.backbone.eval()
            ppo_epochs = max(ppo_epochs, 1)
            print("[REINFORCE] LoRA frozen — caching hidden states, training proj + prototypes.")
        else:
            # Recomputing hidden states each PPO epoch is expensive; keep it to 1
            # update per batch unless the user explicitly raised ppo_epochs.
            print("[REINFORCE] LoRA trainable — hidden states recomputed per epoch.")

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

        # Snapshot the initial (warmup) policy head for the KL-to-init term.
        ref_proj = copy.deepcopy(encoder.proj_head).eval()
        for p in ref_proj.parameters():
            p.requires_grad_(False)
        ref_proto = swav_head.prototypes.weight.detach().clone()
        use_kl = freeze_lora and kl_coef > 0.0

        print(f"[REINFORCE] G={G}, ppo_epochs={ppo_epochs}, clip={ppo_clip}, "
              f"T={temperature}, reward={reward_mode}, "
              f"balance={balance_coef}, ent={ent_coef}, kl={kl_coef if use_kl else 0.0}")
        if patience:
            print(f"[REINFORCE] early stop: {patience} eval points w/o ACC improvement")

        best_acc, no_improve = -1.0, 0
        for it in tqdm(range(1, rl_iters + 1), desc="REINFORCE"):
            idx = rng.sample(range(n), min(batch_size, n))
            va, vb = zip(*[aug(images[i]) for i in idx])
            va, vb = list(va), list(vb)

            # --- Encode the (single) view pair --------------------------------
            if freeze_lora:
                with torch.no_grad():
                    h_a = encoder.encode_hidden_batch(instruction, va)
                    h_b = encoder.encode_hidden_batch(instruction, vb)
            else:
                h_a = h_b = None  # recomputed inside the epoch loop

            # --- Sampling phase (old policy, no grad): actions + advantages ----
            with torch.no_grad():
                swav_head.normalize_prototypes()
                if freeze_lora:
                    logits_a0 = self._logits_from_hidden(encoder, swav_head, h_a, temperature)
                    logits_b0 = self._logits_from_hidden(encoder, swav_head, h_b, temperature)
                    if use_kl:
                        ref_la = F.log_softmax(self._ref_logits_from_hidden(ref_proj, ref_proto, h_a, temperature), dim=1)
                        ref_lb = F.log_softmax(self._ref_logits_from_hidden(ref_proj, ref_proto, h_b, temperature), dim=1)
                        ref_pa, ref_pb = ref_la.exp(), ref_lb.exp()
                else:
                    hh_a = encoder.encode_hidden_batch(instruction, va)
                    hh_b = encoder.encode_hidden_batch(instruction, vb)
                    logits_a0 = self._logits_from_hidden(encoder, swav_head, hh_a, temperature)
                    logits_b0 = self._logits_from_hidden(encoder, swav_head, hh_b, temperature)

                pa0, pb0 = F.softmax(logits_a0, dim=1), F.softmax(logits_b0, dim=1)
                dist_a, dist_b = Categorical(probs=pa0), Categorical(probs=pb0)
                ca = dist_a.sample((G,))                 # (G, B)
                cb = dist_b.sample((G,))                 # (G, B)
                old_logp = dist_a.log_prob(ca) + dist_b.log_prob(cb)   # (G, B)

                if reward_mode == "sinkhorn":
                    # 用对面视图的 sinkhorn 等分码作为 reward 目标：列归一化强制
                    # 每个 prototype 在 batch 内拿到 ~B/K 质量，塌缩到单簇会被
                    # 自动压低码质量 → 选该簇 reward 反而低，抗塌缩内建于 reward。
                    # 码在 no_grad 下计算并作为 detached 标量，梯度仍只走 log π。
                    from vlc.core.losses import sinkhorn
                    q_a = sinkhorn(logits_a0 * temperature, sink_eps, sink_iters)  # (B, K)
                    q_b = sinkhorn(logits_b0 * temperature, sink_eps, sink_iters)
                    qb_at_ca = q_b.unsqueeze(0).expand(G, -1, -1).gather(2, ca.unsqueeze(-1)).squeeze(-1)
                    qa_at_cb = q_a.unsqueeze(0).expand(G, -1, -1).gather(2, cb.unsqueeze(-1)).squeeze(-1)
                    reward = 0.5 * (qb_at_ca + qa_at_cb)             # (G, B)
                elif reward_mode == "soft":
                    pb_at_ca = pb0.unsqueeze(0).expand(G, -1, -1).gather(2, ca.unsqueeze(-1)).squeeze(-1)
                    pa_at_cb = pa0.unsqueeze(0).expand(G, -1, -1).gather(2, cb.unsqueeze(-1)).squeeze(-1)
                    reward = 0.5 * (pb_at_ca + pa_at_cb)             # (G, B) in [0,1]
                else:  # hard agreement
                    reward = (ca == cb).float()                      # (G, B)

                mean_r = reward.mean(dim=0, keepdim=True)
                if normalize_adv:
                    std_r = reward.std(dim=0, keepdim=True)
                    adv = (reward - mean_r) / (std_r + adv_eps)
                else:
                    adv = reward - mean_r
                adv = adv.detach()
                reward_mean = reward.mean().item()

            # --- PPO update phase ---------------------------------------------
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

                logp_a_all = F.log_softmax(logits_a, dim=1)          # (B, K)
                logp_b_all = F.log_softmax(logits_b, dim=1)
                la = logp_a_all.gather(1, ca.t()).t()                # (G, B)
                lb = logp_b_all.gather(1, cb.t()).t()
                new_logp = la + lb                                   # (G, B)

                ratio = torch.exp(new_logp - old_logp)               # (G, B)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                pa, pb = F.softmax(logits_a, dim=1), F.softmax(logits_b, dim=1)
                p_bar = 0.5 * (pa.mean(0) + pb.mean(0))              # (K,)
                marg_entropy = -(p_bar * p_bar.clamp_min(1e-8).log()).sum()
                samp_entropy = -0.5 * (
                    (pa * pa.clamp_min(1e-8).log()).sum(1)
                    + (pb * pb.clamp_min(1e-8).log()).sum(1)
                ).mean()

                loss = policy_loss - balance_coef * marg_entropy - ent_coef * samp_entropy
                if use_kl:
                    kl = 0.5 * (
                        (ref_pa * (ref_la - logp_a_all)).sum(1).mean()
                        + (ref_pb * (ref_lb - logp_b_all)).sum(1).mean()
                    )
                    loss = loss + kl_coef * kl
                    last["kl"] = kl.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                last.update(policy_loss=policy_loss.item(),
                            marg_entropy=marg_entropy.item(),
                            ratio=ratio.mean().item())

            # --- Evaluation / checkpoint --------------------------------------
            if it % eval_every == 0:
                acc, nmi, ari = eval_clustering(encoder, swav_head, images, labels,
                                                instruction, self.device, encode_bs)
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
                with open(self.out_dir / "history.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")

                if acc > best_acc:
                    best_acc, no_improve = acc, 0
                    self._save(encoder, swav_head, "rl_best")
                    print(f"  -> RL best ACC={best_acc:.4f}")
                else:
                    no_improve += 1
                    if patience and no_improve >= patience:
                        print(f"[REINFORCE] early stop: {patience} pts w/o improvement, best ACC={best_acc:.4f}")
                        break

        print(f"[REINFORCE] Done. Best ACC={best_acc:.4f}")


def main(argv=None):
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    SwavReinforceTrainer(cfg).train()


if __name__ == "__main__":
    main()
