"""SwAV-REINFORCE DDP Trainer — two-GPU parallel version.

Architecture
------------
* ``torchrun --standalone --nproc_per_node=2`` launches two processes.
* Rank 0 → GPU 0; Rank 1 → GPU 1.  Each rank loads the FULL Qwen3B model
  on its own GPU (``device_map={"": local_rank}``), so there is no model
  sharding — just pure data parallelism.
* With LoRA frozen (``rl_lr_enc=0``, the default), Qwen's hidden states are
  cached ONCE per iteration.  Multiple cheap PPO epochs then operate over
  the tiny proj_head + prototypes (~1-2 MB of parameters).
* Trainable-parameter gradients are manually ``dist.all_reduce``-d after
  each backward.  No DDP wrapper is needed for such small modules.
* Rank 0 handles eval, logging, and checkpointing.

Effective batch size = ``images_per_batch`` × ``world_size``.
Estimated wall-time speedup: ~1.7× vs. single GPU (bottleneck is encoding).

Launch
------
    torchrun --standalone --nproc_per_node=2 \\
        -m vlc.train.swav_reinforce_ddp_trainer \\
        --config configs/vlm/swav_reinforce_ddp.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import AdamW
from tqdm import tqdm

from vlc.train.swav_reinforce_trainer import SwavReinforceTrainer
from vlc.train.swav_rl_trainer import TwoViewAug, eval_clustering


class SwavReinforceTrainerDDP(SwavReinforceTrainer):
    """Data-parallel wrapper around the correct REINFORCE/PPO phase.

    Inherits all warmup, checkpoint, and eval logic from the parent chain.
    Only the RL phase and model-building are DDP-aware.
    """

    def __init__(self, cfg: dict) -> None:
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._is_dist    = self._world_size > 1

        if self._is_dist:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self._local_rank)

        # super().__init__ sets self.device = cuda:0 regardless of rank;
        # we override it afterwards to the rank-specific GPU.
        super().__init__(cfg)
        self.device = torch.device(f"cuda:{self._local_rank}")

    @property
    def _is_main(self) -> bool:
        return self._local_rank == 0

    # ------------------------------------------------------------------
    # Override: each rank loads its model onto ITS OWN GPU
    # ------------------------------------------------------------------

    def _build_encoder(self):
        """Load Qwen on this rank's GPU.  device_map='auto' is wrong for DDP
        because it would shard the model across ALL visible GPUs instead of
        giving each rank a self-contained copy."""
        from vlc.model.qwen_vl import load_model_and_processor, add_lora
        from vlc.model.encoder import CriterionEncoder, ProjectionHead
        from peft import PeftModel

        model_cfg = self.cfg["model"]
        model_id  = model_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        p         = model_cfg.get("projection", {})
        in_dim    = p.get("in_dim",  2048)
        mid_dim   = p.get("mid_dim",  512)
        out_dim   = p.get("out_dim",  256)

        # {"": rank} puts ALL layers on exactly one GPU — no sharding.
        backbone, processor = load_model_and_processor(
            model_id=model_id,
            device_map={"": self._local_rank},
        )

        lora_path = model_cfg.get("lora_path")
        if lora_path and Path(lora_path).exists():
            backbone = PeftModel.from_pretrained(backbone, lora_path, is_trainable=True)
        else:
            backbone = add_lora(backbone)

        device_str = f"cuda:{self._local_rank}"
        proj_head  = ProjectionHead(in_dim, mid_dim, out_dim).to(device_str)
        encoder    = CriterionEncoder(backbone, processor, proj_head)

        if model_cfg.get("gradient_checkpointing", True):
            encoder.backbone.gradient_checkpointing_enable()

        return encoder

    def _build_swav_head(self, out_dim: int, k: int):
        from vlc.model.swav_head import SwAVHead
        return SwAVHead(dim=out_dim, n_prototypes=k).to(self.device)

    def _save(self, encoder, swav_head, tag: str) -> None:
        if self._is_main:
            super()._save(encoder, swav_head, tag)

    # ------------------------------------------------------------------
    # Hook called by SwavRLTrainer.train()
    # ------------------------------------------------------------------

    def _train_grpo(self, encoder, swav_head, images, labels, instruction):
        self._train_reinforce_ddp(encoder, swav_head, images, labels, instruction)

    # ------------------------------------------------------------------
    # DDP-aware REINFORCE / PPO loop
    # ------------------------------------------------------------------

    def _train_reinforce_ddp(self, encoder, swav_head, images, labels, instruction):
        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]

        rl_iters      = int(tcfg.get("rl_iterations",   300))
        batch_size    = int(dcfg.get("images_per_batch",   8))
        G             = int(tcfg.get("group_size",         8))
        temperature   = float(tcfg.get("temperature",    0.5))
        eval_every    = int(tcfg.get("rl_eval_every",      5))
        encode_bs     = int(dcfg.get("encode_batch_size",  8))
        max_grad_norm = float(tcfg.get("max_grad_norm",  1.0))
        patience      = int(tcfg.get("early_stop_patience", 0))

        lr       = float(tcfg.get("rl_lr",       1e-5))
        lr_enc   = float(tcfg.get("rl_lr_enc",   tcfg.get("lr_enc",   lr)))
        lr_proj  = float(tcfg.get("rl_lr_proj",  tcfg.get("lr_proj",  lr)))
        lr_proto = float(tcfg.get("rl_lr_proto", tcfg.get("lr_proto", lr)))

        ppo_epochs   = int(tcfg.get("ppo_epochs",    4))
        ppo_clip     = float(tcfg.get("ppo_clip",  0.2))
        balance_coef = float(tcfg.get("balance_coef", 1.0))
        ent_coef     = float(tcfg.get("ent_coef",   0.01))
        kl_coef      = float(tcfg.get("kl_coef",    0.1))
        reward_mode  = str(tcfg.get("reward_mode", "hard")).lower()
        normalize_adv = bool(tcfg.get("normalize_adv", True))
        adv_eps      = float(tcfg.get("adv_eps",    1e-2))
        sink_eps     = float(tcfg.get("sinkhorn_epsilon", 0.05))
        sink_iters   = int(tcfg.get("sinkhorn_iters",   3))

        n   = len(images)
        aug = TwoViewAug(dcfg.get("image_size", 64))

        # Different RNG seed per rank → each rank samples different images,
        # giving true data-parallel diversity instead of redundant computation.
        rng = random.Random(self.cfg.get("seed", 42) + 7 + self._local_rank * 1337)

        freeze_lora = lr_enc <= 0
        if freeze_lora:
            for p in encoder.backbone.parameters():
                p.requires_grad_(False)
            encoder.backbone.eval()
            ppo_epochs = max(ppo_epochs, 1)

        for p in swav_head.parameters():
            p.requires_grad_(True)

        param_groups = []
        if not freeze_lora:
            param_groups.append({
                "params": [p for p in encoder.backbone.parameters() if p.requires_grad],
                "lr": lr_enc,
            })
        param_groups.extend([
            {"params": list(encoder.proj_head.parameters()), "lr": lr_proj},
            {"params": list(swav_head.parameters()),         "lr": lr_proto},
        ])
        trainable = [p for g in param_groups for p in g["params"]]
        optimizer = AdamW(param_groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))

        # Reference (warmup-init) policy for KL-to-init penalty.
        ref_proj  = copy.deepcopy(encoder.proj_head).eval()
        for p in ref_proj.parameters():
            p.requires_grad_(False)
        ref_proto = swav_head.prototypes.weight.detach().clone()
        use_kl    = freeze_lora and kl_coef > 0.0

        if self._is_main:
            eff_batch = batch_size * self._world_size
            print(
                f"[REINFORCE-DDP] world={self._world_size} "
                f"eff_batch={eff_batch} (={batch_size}×{self._world_size}) "
                f"G={G} ppo_epochs={ppo_epochs} clip={ppo_clip} "
                f"T={temperature} reward={reward_mode} "
                f"balance={balance_coef} ent={ent_coef} "
                f"kl={kl_coef if use_kl else 0.0}"
            )
            if patience:
                print(f"[REINFORCE-DDP] early stop patience={patience}")

        best_acc, no_improve = -1.0, 0

        iter_range = (
            tqdm(range(1, rl_iters + 1), desc="REINFORCE-DDP")
            if self._is_main
            else range(1, rl_iters + 1)
        )

        should_stop = False
        for it in iter_range:
            idx = rng.sample(range(n), min(batch_size, n))
            va, vb = zip(*[aug(images[i]) for i in idx])
            va, vb = list(va), list(vb)

            # --- Cache hidden states (LoRA-frozen fast path) --------------
            if freeze_lora:
                with torch.no_grad():
                    h_a = encoder.encode_hidden_batch(instruction, va)
                    h_b = encoder.encode_hidden_batch(instruction, vb)
            else:
                h_a = h_b = None  # recomputed inside epoch loop

            # --- Sampling phase: discrete actions + advantages ------------
            with torch.no_grad():
                swav_head.normalize_prototypes()
                if freeze_lora:
                    logits_a0 = self._logits_from_hidden(encoder, swav_head, h_a, temperature)
                    logits_b0 = self._logits_from_hidden(encoder, swav_head, h_b, temperature)
                    if use_kl:
                        ref_la = F.log_softmax(
                            self._ref_logits_from_hidden(ref_proj, ref_proto, h_a, temperature),
                            dim=1)
                        ref_lb = F.log_softmax(
                            self._ref_logits_from_hidden(ref_proj, ref_proto, h_b, temperature),
                            dim=1)
                        ref_pa, ref_pb = ref_la.exp(), ref_lb.exp()
                else:
                    hh_a = encoder.encode_hidden_batch(instruction, va)
                    hh_b = encoder.encode_hidden_batch(instruction, vb)
                    logits_a0 = self._logits_from_hidden(encoder, swav_head, hh_a, temperature)
                    logits_b0 = self._logits_from_hidden(encoder, swav_head, hh_b, temperature)

                pa0, pb0   = F.softmax(logits_a0, dim=1), F.softmax(logits_b0, dim=1)
                dist_a     = Categorical(probs=pa0)
                dist_b     = Categorical(probs=pb0)
                ca         = dist_a.sample((G,))   # (G, B)
                cb         = dist_b.sample((G,))
                old_logp   = dist_a.log_prob(ca) + dist_b.log_prob(cb)   # (G, B)

                if reward_mode == "sinkhorn":
                    # 用对面视图的 sinkhorn 等分码作为 reward 目标：列归一化强制
                    # 每个 prototype 在 batch 内拿到 ~B/K 质量，塌缩到单簇会被
                    # 自动压低码质量 → 选该簇 reward 反而低，抗塌缩内建于 reward。
                    # 码在 no_grad 下计算并作为 detached 标量，梯度仍只走 log π。
                    from vlc.core.losses import sinkhorn
                    q_a = sinkhorn(logits_a0 * temperature, sink_eps, sink_iters)  # (B, K)
                    q_b = sinkhorn(logits_b0 * temperature, sink_eps, sink_iters)
                    qb_at_ca = q_b.unsqueeze(0).expand(G, -1, -1).gather(
                        2, ca.unsqueeze(-1)).squeeze(-1)
                    qa_at_cb = q_a.unsqueeze(0).expand(G, -1, -1).gather(
                        2, cb.unsqueeze(-1)).squeeze(-1)
                    reward = 0.5 * (qb_at_ca + qa_at_cb)
                elif reward_mode == "soft":
                    pb_at_ca = pb0.unsqueeze(0).expand(G, -1, -1).gather(
                        2, ca.unsqueeze(-1)).squeeze(-1)
                    pa_at_cb = pa0.unsqueeze(0).expand(G, -1, -1).gather(
                        2, cb.unsqueeze(-1)).squeeze(-1)
                    reward = 0.5 * (pb_at_ca + pa_at_cb)
                else:  # hard agreement
                    reward = (ca == cb).float()

                mean_r = reward.mean(dim=0, keepdim=True)
                if normalize_adv:
                    adv = (reward - mean_r) / (reward.std(dim=0, keepdim=True) + adv_eps)
                else:
                    adv = reward - mean_r
                adv         = adv.detach()
                reward_mean = reward.mean().item()

            # --- PPO update ----------------------------------------------
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
                        encoder, swav_head,
                        encoder.encode_hidden_batch(instruction, va), temperature)
                    logits_b = self._logits_from_hidden(
                        encoder, swav_head,
                        encoder.encode_hidden_batch(instruction, vb), temperature)

                logp_a_all = F.log_softmax(logits_a, dim=1)   # (B, K)
                logp_b_all = F.log_softmax(logits_b, dim=1)
                la         = logp_a_all.gather(1, ca.t()).t()  # (G, B)
                lb         = logp_b_all.gather(1, cb.t()).t()
                new_logp   = la + lb

                ratio  = torch.exp(new_logp - old_logp)
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                pa    = F.softmax(logits_a, dim=1)
                pb    = F.softmax(logits_b, dim=1)
                p_bar = 0.5 * (pa.mean(0) + pb.mean(0))
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

                # DDP: average each trainable parameter's gradient across
                # both ranks so the update is equivalent to computing on
                # the full effective batch.
                if self._is_dist:
                    for param in trainable:
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                last.update(
                    policy_loss=policy_loss.item(),
                    marg_entropy=marg_entropy.item(),
                    ratio=ratio.mean().item(),
                )

            # --- Eval / checkpoint (rank 0 only) -------------------------
            if it % eval_every == 0:
                # Barrier: wait for all ranks to finish this iteration
                # before rank 0 runs the single-process eval.
                if self._is_dist:
                    dist.barrier()

                if self._is_main:
                    acc, nmi, ari = eval_clustering(
                        encoder, swav_head, images, labels,
                        instruction, self.device, encode_bs)
                    msg = (
                        f"[REINFORCE-DDP it={it:4d}] "
                        f"reward={reward_mean:.4f} "
                        f"|pg|={last.get('policy_loss', 0):.4f} "
                        f"H(marg)={last.get('marg_entropy', 0):.3f} "
                        f"ratio={last.get('ratio', 1):.3f} "
                        f"ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}"
                    )
                    if use_kl:
                        msg += f" KL={last.get('kl', 0):.4f}"
                    print(msg)
                    with open(self.out_dir / "history.jsonl", "a") as f:
                        f.write(json.dumps({
                            "phase": "reinforce_ddp", "iter": it,
                            "reward": reward_mean, "acc": acc, "nmi": nmi, "ari": ari,
                            "marg_entropy": last.get("marg_entropy", 0.0),
                            "ratio": last.get("ratio", 1.0),
                            **({"kl": last.get("kl", 0.0)} if use_kl else {}),
                        }) + "\n")

                    if acc > best_acc:
                        best_acc, no_improve = acc, 0
                        self._save(encoder, swav_head, "rl_best")
                        print(f"  -> RL best ACC={best_acc:.4f}")
                    else:
                        no_improve += 1

                # Broadcast the early-stop signal from rank 0 to all ranks.
                # Non-main ranks have no_improve=0 (they never evaluate), so
                # they need rank 0's decision.
                if self._is_dist:
                    stop_t = torch.tensor(0, device=self.device)
                    if self._is_main:
                        stop_t.fill_(int(patience and no_improve >= patience))
                    dist.broadcast(stop_t, src=0)
                    should_stop = bool(stop_t.item())
                else:
                    should_stop = patience > 0 and no_improve >= patience

                if should_stop:
                    if self._is_main:
                        print(
                            f"[REINFORCE-DDP] early stop: "
                            f"{patience} pts w/o improvement, "
                            f"best ACC={best_acc:.4f}"
                        )
                    break

        if self._is_main:
            print(f"[REINFORCE-DDP] Done. Best ACC={best_acc:.4f}")

    def train(self) -> None:
        super().train()
        if self._is_dist:
            dist.destroy_process_group()


def main(argv=None):
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    SwavReinforceTrainerDDP(cfg).train()


if __name__ == "__main__":
    main()
