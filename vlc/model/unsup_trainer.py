"""Unsupervised training pipeline for CriterionEncoder.

Supports two modes:
  method: dec         – DEC-style KL sharpening (default)
  method: contrastive – CC-style instance + cluster contrastive

Labels are NEVER used during training.
"""

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


class UnsupEncoderTrainer:

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.get("output_dir", "artifacts/vlm/unsup_encoder"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.method = cfg.get("training", {}).get("method", "dec")

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------

    def _build_encoder(self) -> Any:
        from vlc.model.encoder import build_encoder, load_encoder_checkpoint

        model_cfg = self.cfg.get("model", {})
        model_id = model_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        lora_path = model_cfg.get("lora_path", None)
        proj_cfg = model_cfg.get("projection", {})
        in_dim = proj_cfg.get("in_dim", 2048)
        mid_dim = proj_cfg.get("mid_dim", 512)
        out_dim = proj_cfg.get("out_dim", 256)

        if lora_path and Path(lora_path).exists():
            proj_pt = str(Path(lora_path).parent / "proj_head.pt")
            if Path(proj_pt).exists():
                print(f"[trainer] Loading checkpoint from {lora_path}")
                encoder = load_encoder_checkpoint(
                    model_id, lora_path, proj_pt, in_dim, mid_dim, out_dim
                )
            else:
                print(f"[trainer] Warm-starting LoRA from {lora_path}")
                encoder = build_encoder(model_id, lora_path, in_dim, mid_dim, out_dim)
        else:
            print("[trainer] Building encoder from scratch")
            encoder = build_encoder(model_id, None, in_dim, mid_dim, out_dim)

        if model_cfg.get("gradient_checkpointing", True):
            encoder.backbone.gradient_checkpointing_enable()
        return encoder

    def _build_dec_head(self, embed_dim: int, k: int) -> Any:
        from vlc.model.dec_head import DECHead
        head = DECHead(embed_dim=embed_dim, k=k, alpha=1.0)
        head = head.to(self.device)
        return head

    def _build_cc_heads(self, in_dim: int, k: int) -> Any:
        from vlc.model.cc_head import CCHeads
        train_cfg = self.cfg.get("training", {})
        mid_dim = self.cfg["model"]["projection"].get("mid_dim", 512)
        feat_dim = train_cfg.get("cc_feat_dim", 128)
        heads = CCHeads(in_dim=in_dim, mid_dim=mid_dim, feat_dim=feat_dim, k=k)
        return heads.to(self.device)

    def _build_dataset(self) -> Any:
        from vlc.episodes.unsup_dataset import UnsupDataset

        data_cfg = self.cfg.get("data", {})
        return UnsupDataset(
            dataset_configs=data_cfg["datasets"],
            images_per_batch=data_cfg.get("images_per_batch", 32),
            batches_per_epoch=data_cfg.get("batches_per_epoch", 200),
            mode=self.method,
            image_size=data_cfg.get("image_size", 64),
            seed=self.cfg.get("seed", 42),
        )

    # ------------------------------------------------------------------
    # DEC: encode full subset -> init centroids -> KL training loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_subset(self, encoder: Any, dataset: Any) -> dict[str, tuple]:
        """Encode all images in the dataset; group by criterion_key.

        Returns
        -------
        dict: criterion_key -> (embeddings_tensor, images_list)
        """
        from collections import defaultdict

        buckets: dict[str, list] = defaultdict(list)
        encoder.backbone.eval()
        print("[init] Encoding subset for DEC initialisation...")

        for i in tqdm(range(len(dataset)), desc="encode-subset"):
            batch = dataset[i]
            crit_key = batch["criterion_key"]
            z = encoder.encode_batch(batch["criterion"], batch["images"])
            buckets[crit_key].append(z.detach().cpu())

        result = {}
        for k, tensors in buckets.items():
            result[k] = torch.cat(tensors, dim=0)
        return result

    def _init_dec(self, encoder: Any, dec_head: Any, dataset: Any) -> None:
        embeddings_map = self._encode_subset(encoder, dataset)
        k = self.cfg["training"]["k"]
        for crit_key, emb in embeddings_map.items():
            print(f"[init] KMeans for '{crit_key}' on {emb.shape[0]} vectors...")
            dec_head.init_centroids(crit_key, emb.to(self.device))
        encoder.backbone.train()

    def _dec_step(
        self,
        encoder: Any,
        dec_head: Any,
        batch: dict,
        p_cache: dict,
        refresh_p: bool,
    ) -> torch.Tensor:
        from vlc.core.losses import dec_target_distribution, dec_kl_loss

        crit_text = batch["criterion"]
        crit_key = batch["criterion_key"]
        images = batch["images"]

        z = encoder.encode_batch(crit_text, images)  # (N, d)

        # Initialise centroids on-the-fly for unseen criteria
        if not dec_head.has_criterion(crit_key):
            with torch.no_grad():
                dec_head.init_centroids(crit_key, z.detach())

        q = dec_head(z, crit_key)  # (N, K)

        if refresh_p or crit_key not in p_cache:
            p_cache[crit_key] = dec_target_distribution(q.detach())

        p = p_cache[crit_key]
        # p may have different batch size if cached from another step — recompute if mismatch
        if p.shape[0] != q.shape[0]:
            p_cache[crit_key] = dec_target_distribution(q.detach())
            p = p_cache[crit_key]

        return dec_kl_loss(q, p)

    # ------------------------------------------------------------------
    # CC: instance + cluster contrastive step
    # ------------------------------------------------------------------

    def _cc_step(self, encoder: Any, cc_heads: Any, batch: dict) -> torch.Tensor:
        from vlc.core.losses import instance_contrastive_loss, cluster_contrastive_loss

        train_cfg = self.cfg.get("training", {})
        t_inst = float(train_cfg.get("temperature_instance", 0.5))
        t_clust = float(train_cfg.get("temperature_cluster", 1.0))
        ent_w = float(train_cfg.get("entropy_weight", 5.0))

        crit_text = batch["criterion"]

        # Strict CC: two independent heads on shared backbone hidden states
        h_a = encoder.encode_hidden_batch(crit_text, batch["views_a"])  # (N, hidden)
        h_b = encoder.encode_hidden_batch(crit_text, batch["views_b"])

        z_a, c_a = cc_heads(h_a)   # instance feat + cluster assign
        z_b, c_b = cc_heads(h_b)

        inst_loss = instance_contrastive_loss(z_a, z_b, t_inst)
        clust_loss = cluster_contrastive_loss(c_a, c_b, t_clust, ent_w)

        return inst_loss + clust_loss

    # ------------------------------------------------------------------
    # Label-free model selection: silhouette approximation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _silhouette_score(self, encoder: Any, head: Any, dataset: Any) -> float:
        """Approximate silhouette on a small probe batch (label-free)."""
        try:
            from sklearn.metrics import silhouette_score as sk_sil
        except ImportError:
            return 0.0

        encoder.backbone.eval()
        batch = dataset[0]
        crit_key = batch["criterion_key"]
        images = batch.get("images", batch.get("views_a"))

        try:
            if self.method == "dec":
                z = encoder.encode_batch(batch["criterion"], images)
                z_np = z.detach().cpu().float().numpy()
                if not head.has_criterion(crit_key):
                    return -1.0
                assignments = head.hard_assignments(z, crit_key).cpu().numpy()
            else:
                # CC: cluster from cc_heads on backbone hidden; silhouette on instance feat
                h = encoder.encode_hidden_batch(batch["criterion"], images)
                z, c = head(h)
                z_np = z.detach().cpu().float().numpy()
                assignments = c.argmax(dim=-1).cpu().numpy()
        except Exception:
            encoder.backbone.train()
            return -1.0

        encoder.backbone.train()
        if len(set(assignments.tolist())) < 2:
            return -1.0
        try:
            return float(sk_sil(z_np, assignments))
        except Exception:
            return -1.0

    # ------------------------------------------------------------------
    # Main train loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        if self.method == "dec":
            self._train_dec_global()
            return
        if self.method == "swav":
            self._train_swav()
            return
        self._train_cc()

    # ------------------------------------------------------------------
    # SwAV (Caron et al. 2020): swapped-assignment prediction with Sinkhorn
    # equipartition. Trains LoRA + cls_token + proj_head + prototypes. Two
    # augmented views; codes from one view supervise the other.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_proj(self, encoder: Any, images: list, instruction: str,
                     batch_size: int) -> torch.Tensor:
        """Encode all images -> L2-normalized projection embeddings (N, out_dim)."""
        encoder.backbone.eval()
        chunks = []
        for i in range(0, len(images), batch_size):
            z = encoder.encode_batch(instruction, images[i:i + batch_size])
            chunks.append(z.detach())
        encoder.backbone.train()
        return torch.cat(chunks, dim=0)

    def _train_swav(self) -> None:
        import random
        import numpy as np
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from vlc.model.encoder import save_encoder_checkpoint
        from vlc.model.swav_head import SwAVHead
        from vlc.core.losses import swav_loss, cluster_acc
        from vlc.episodes.unsup_dataset import build_dec_indexed, _TwoViewAug

        train_cfg = self.cfg.get("training", {})
        n_epochs = train_cfg.get("epochs", 15)
        lr = float(train_cfg.get("lr", 1e-4))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        grad_accum = train_cfg.get("grad_accum_steps", 4)
        batch_size = self.cfg["data"].get("images_per_batch", 32)
        steps_per_epoch = self.cfg["data"].get("batches_per_epoch", 40)
        encode_bs = self.cfg["data"].get("encode_batch_size", 16)
        image_size = self.cfg["data"].get("image_size", 64)
        k = train_cfg["k"]
        out_dim = self.cfg["model"]["projection"].get("out_dim", 256)

        temperature = float(train_cfg.get("temperature", 0.1))
        epsilon = float(train_cfg.get("sinkhorn_epsilon", 0.05))
        sink_iters = int(train_cfg.get("sinkhorn_iters", 3))
        freeze_proto_epochs = int(train_cfg.get("freeze_prototypes_epochs", 1))

        import torch.nn.functional as F
        from sklearn.cluster import KMeans

        encoder = self._build_encoder()
        head = SwAVHead(dim=out_dim, n_prototypes=k).to(self.device)
        images, criteria_info = build_dec_indexed(
            self.cfg["data"], image_size
        )
        n = len(images)
        crit_keys = list(criteria_info.keys())
        aug = _TwoViewAug(image_size)
        rng = random.Random(self.cfg.get("seed", 42))
        print(f"[SwAV] {n} images, criteria={crit_keys}, K={k}, "
              f"temp={temperature}, eps={epsilon}")

        # Init prototypes from KMeans on the initial projection embeddings so the
        # swapped-prediction targets are meaningful from step 1 (avoids the
        # random-prototype collapse). Uses the first criterion.
        ck0 = crit_keys[0]
        z0 = self._encode_proj(encoder, images, criteria_info[ck0]["instruction"], encode_bs)
        km = KMeans(n_clusters=k, n_init=10, random_state=self.cfg.get("seed", 42))
        km.fit(z0.cpu().float().numpy())
        centroids = torch.tensor(km.cluster_centers_, dtype=head.prototypes.weight.dtype,
                                 device=self.device)
        head.prototypes.weight.data.copy_(F.normalize(centroids, dim=1))
        print(f"[SwAV] prototypes initialised from KMeans on {n} init embeddings")

        # Trainable: LoRA + cls_token + proj_head + prototypes
        trainable = (
            [p for p in encoder.backbone.parameters() if p.requires_grad]
            + [encoder.cls_token]
            + list(encoder.proj_head.parameters())
            + list(head.parameters())
        )
        optimizer = AdamW(trainable, lr=lr,
                          weight_decay=float(train_cfg.get("weight_decay", 0.01)))
        total_steps = max(n_epochs * steps_per_epoch // grad_accum, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

        best_acc = -1.0
        for epoch in range(n_epochs):
            # 1) Monitor: assign all images via prototypes argmax (label-free loss,
            #    labels used only for reporting).
            head.normalize_prototypes()
            accs = []
            for ck in crit_keys:
                z = self._encode_proj(encoder, images, criteria_info[ck]["instruction"], encode_bs)
                scores = head(z)
                pred = scores.argmax(1).cpu().numpy()
                lab = criteria_info[ck]["labels"]
                acc = cluster_acc(lab, pred)
                ari = adjusted_rand_score(lab, pred)
                nmi = normalized_mutual_info_score(lab, pred)
                accs.append(acc)
                print(f"  [{ck}] ACC={acc:.4f} ARI={ari:.4f} NMI={nmi:.4f}")
            mean_acc = float(np.mean(accs))
            if mean_acc > best_acc:
                best_acc = mean_acc
                save_encoder_checkpoint(encoder, str(self.out_dir / "best"))
                torch.save(head.state_dict(), str(self.out_dir / "best" / "swav_head.pt"))
                print(f"  -> new best mean ACC={best_acc:.4f}")
            with open(self.out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "mean_acc": mean_acc}) + "\n")

            # 2) Train steps: swapped-assignment prediction on two views
            encoder.backbone.train()
            epoch_loss, nb = 0.0, 0
            optimizer.zero_grad()
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{n_epochs} [swav]")
            for step_i in pbar:
                ck = rng.choice(crit_keys)
                instr = criteria_info[ck]["instruction"]
                idx = rng.sample(range(n), min(batch_size, n))
                views_a, views_b = zip(*[aug(images[i]) for i in idx])
                try:
                    head.normalize_prototypes()
                    z_a = encoder.encode_batch(instr, list(views_a))   # (B, out_dim)
                    z_b = encoder.encode_batch(instr, list(views_b))
                    scores_a = head(z_a)
                    scores_b = head(z_b)
                    loss = swav_loss(scores_a, scores_b, temperature, epsilon, sink_iters)
                    (loss / grad_accum).backward()
                    epoch_loss += float(loss.item())
                    nb += 1
                except Exception as e:
                    print(f"  [skip] step {step_i}: {e}")
                    optimizer.zero_grad()
                    continue

                if (step_i + 1) % grad_accum == 0:
                    # Freeze prototype grads for the first epoch(s) for stability
                    if epoch < freeze_proto_epochs:
                        head.prototypes.weight.grad = None
                    nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if (step_i + 1) % 10 == 0:
                    pbar.set_postfix(loss=f"{epoch_loss/max(nb,1):.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")

            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{n_epochs}: loss={epoch_loss/max(nb,1):.4f}  meanACC={mean_acc:.4f}")

        print(f"Done. Best mean ACC={best_acc:.4f}, checkpoints at {self.out_dir}")

    # ------------------------------------------------------------------
    # DEC (DiEC-style): GLOBAL target distribution P, indexed per sample,
    # refreshed once per epoch. External metrics (ACC/ARI/NMI) for monitoring.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_indexed(self, encoder: Any, images: list, instruction: str,
                        batch_size: int) -> torch.Tensor:
        """Encode all images under one criterion (no grad), in the backbone
        hidden space (L2-normalized 2048-d). Returns (N, d)."""
        import torch.nn.functional as F
        encoder.backbone.eval()
        chunks = []
        for i in range(0, len(images), batch_size):
            h = encoder.encode_hidden_batch(instruction, images[i:i + batch_size])
            chunks.append(F.normalize(h.detach(), dim=-1))
        encoder.backbone.train()
        return torch.cat(chunks, dim=0)

    def _train_dec_global(self) -> None:
        import random
        import numpy as np
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from vlc.model.encoder import save_encoder_checkpoint
        from vlc.core.losses import dec_target_distribution, dec_kl_loss, cluster_acc
        from vlc.episodes.unsup_dataset import build_dec_indexed

        train_cfg = self.cfg.get("training", {})
        n_epochs = train_cfg.get("epochs", 15)
        lr = float(train_cfg.get("lr", 1e-4))
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        grad_accum = train_cfg.get("grad_accum_steps", 4)
        batch_size = self.cfg["data"].get("images_per_batch", 32)
        steps_per_epoch = self.cfg["data"].get("batches_per_epoch", 40)
        encode_bs = self.cfg["data"].get("encode_batch_size", 16)
        k = train_cfg["k"]
        # DEC clusters in the backbone hidden space (meaningful, like CC),
        # NOT a randomly-initialized proj_head.
        embed_dim = self.cfg["model"]["projection"].get("in_dim", 2048)

        encoder = self._build_encoder()
        head = self._build_dec_head(embed_dim, k)
        images, criteria_info = build_dec_indexed(
            self.cfg["data"], self.cfg["data"].get("image_size", 64)
        )
        n = len(images)
        crit_keys = list(criteria_info.keys())
        rng = random.Random(self.cfg.get("seed", 42))
        print(f"[DEC] {n} images, criteria={crit_keys}, K={k}")

        # --- KMeans init centroids per criterion (global embeddings) ---
        temp_factor = float(train_cfg.get("cluster_temperature_factor", 1.0))
        first_emb = None
        for ck in crit_keys:
            emb = self._encode_indexed(encoder, images, criteria_info[ck]["instruction"], encode_bs)
            head.init_centroids(ck, emb)
            if first_emb is None:
                first_emb = (emb, ck)

        # Auto-set distance temperature so q is soft (avoid one-hot saturation).
        # Use mean squared distance to centroids on the first criterion as scale.
        emb0, ck0 = first_emb
        with torch.no_grad():
            mu0 = head.centroid_params(ck0)
            d2 = ((emb0.unsqueeze(1) - mu0.unsqueeze(0)) ** 2).sum(-1)  # (N, k)
            head.temperature = float(d2.mean().item()) * temp_factor
        print(f"[DEC] cluster temperature set to {head.temperature:.4f}")

        # --- Build per-criterion KNN graph on init embeddings (DiEC-style) ---
        from sklearn.neighbors import NearestNeighbors
        kn = int(train_cfg.get("k_neighbors", 5))
        graph_idx: dict = {}    # ck -> (N, kn) neighbor indices (long tensor)
        graph_w: dict = {}      # ck -> (N, kn) row-normalized weights
        for ck in crit_keys:
            emb = self._encode_indexed(encoder, images, criteria_info[ck]["instruction"], encode_bs)
            q = head(emb, ck)
            pred = q.argmax(1).cpu().numpy()
            acc = cluster_acc(criteria_info[ck]["labels"], pred)
            print(f"[DEC][init] {ck}: ACC={acc:.4f}  q_max_mean={q.max(1).values.mean().item():.3f}")

            emb_np = emb.detach().cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=kn + 1).fit(emb_np)
            dists, idxs = nbrs.kneighbors(emb_np)
            # drop self (col 0); Gaussian weights on feature distance
            idxs = idxs[:, 1:]
            dists = dists[:, 1:]
            sigma = float(np.mean(dists)) + 1e-8
            w = np.exp(-(dists ** 2) / (2 * sigma ** 2))
            w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
            graph_idx[ck] = torch.tensor(idxs, dtype=torch.long, device=self.device)
            graph_w[ck] = torch.tensor(w, dtype=torch.float32, device=self.device)

        kl_weight = float(train_cfg.get("kl_weight", 0.1))
        graph_weight = float(train_cfg.get("graph_weight", 1.0))
        balance_weight = float(train_cfg.get("balance_weight", 1.0))
        print(f"[DEC] kl_w={kl_weight} graph_w={graph_weight} balance_w={balance_weight} k_neighbors={kn}")

        # DEC trains LoRA + centroids (proj_head unused in hidden-space DEC)
        trainable = (
            [p for p in encoder.backbone.parameters() if p.requires_grad]
            + list(head.parameters())
        )
        optimizer = AdamW(trainable, lr=lr,
                          weight_decay=float(train_cfg.get("weight_decay", 0.01)))
        total_steps = max(n_epochs * steps_per_epoch // grad_accum, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

        best_acc = -1.0
        for epoch in range(n_epochs):
            # 1) Refresh GLOBAL target distribution P + global Q per criterion + monitor
            p_target: dict = {}
            global_q: dict = {}
            accs = []
            for ck in crit_keys:
                emb = self._encode_indexed(encoder, images, criteria_info[ck]["instruction"], encode_bs)
                q = head(emb, ck)                       # (N, k)
                p_target[ck] = dec_target_distribution(q.detach())
                global_q[ck] = q.detach()
                pred = q.argmax(1).cpu().numpy()
                lab = criteria_info[ck]["labels"]
                acc = cluster_acc(lab, pred)
                ari = adjusted_rand_score(lab, pred)
                nmi = normalized_mutual_info_score(lab, pred)
                accs.append(acc)
                print(f"  [{ck}] ACC={acc:.4f} ARI={ari:.4f} NMI={nmi:.4f}")
            mean_acc = float(np.mean(accs))

            if mean_acc > best_acc:
                best_acc = mean_acc
                save_encoder_checkpoint(encoder, str(self.out_dir / "best"))
                print(f"  -> new best mean ACC={best_acc:.4f}")

            with open(self.out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "mean_acc": mean_acc}) + "\n")

            # 2) Train steps: KL(q_batch, P[indices])
            encoder.backbone.train()
            epoch_loss, nb = 0.0, 0
            optimizer.zero_grad()
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{n_epochs} [dec]")
            for step_i in pbar:
                ck = rng.choice(crit_keys)
                idx = rng.sample(range(n), min(batch_size, n))
                batch_imgs = [images[i] for i in idx]
                try:
                    import torch.nn.functional as F
                    h = encoder.encode_hidden_batch(criteria_info[ck]["instruction"], batch_imgs)
                    z = F.normalize(h, dim=-1)
                    q = head(z, ck)                       # (B, k)
                    p = p_target[ck][idx]
                    kl = dec_kl_loss(q, p)

                    # Graph Laplacian: pull neighbours' soft assignments together
                    idx_t = torch.tensor(idx, device=self.device)
                    neigh = graph_idx[ck][idx_t]          # (B, kn)
                    wts = graph_w[ck][idx_t]              # (B, kn)
                    q_neigh = global_q[ck][neigh]        # (B, kn, k)
                    diff2 = (q.unsqueeze(1) - q_neigh) ** 2   # (B, kn, k)
                    graph = (wts.unsqueeze(-1) * diff2).sum(-1).sum(-1).mean()

                    # Balance: push batch mean assignment toward uniform (anti-collapse)
                    mean_q = q.mean(dim=0).clamp(min=1e-8)
                    balance = (mean_q * mean_q.log()).sum()   # = -H(mean_q); min -> uniform

                    loss = (kl_weight * kl + graph_weight * graph + balance_weight * balance) / grad_accum
                    loss.backward()
                    epoch_loss += float(loss.item()) * grad_accum
                    nb += 1
                except Exception as e:
                    print(f"  [skip] step {step_i}: {e}")
                    optimizer.zero_grad()
                    continue

                if (step_i + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if (step_i + 1) % 10 == 0:
                    pbar.set_postfix(kl=f"{epoch_loss/max(nb,1):.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")

            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch+1}/{n_epochs}: KL={epoch_loss/max(nb,1):.4f}  meanACC={mean_acc:.4f}")

        print(f"Done. Best mean ACC={best_acc:.4f}, checkpoints at {self.out_dir}")

    # ------------------------------------------------------------------
    # CC training loop (unchanged)
    # ------------------------------------------------------------------

    def _train_cc(self) -> None:
        from vlc.model.encoder import save_encoder_checkpoint

        train_cfg = self.cfg.get("training", {})
        n_epochs = train_cfg.get("epochs", 10)
        lr = float(train_cfg.get("lr", 1e-4))
        grad_accum = train_cfg.get("grad_accum_steps", 4)
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        log_every = train_cfg.get("log_every", 20)
        refresh_every = train_cfg.get("refresh_p_every", 50)
        embed_dim = self.cfg["model"]["projection"].get("out_dim", 256)
        k = train_cfg["k"]

        encoder = self._build_encoder()
        dataset = self._build_dataset()
        in_dim = self.cfg["model"]["projection"].get("in_dim", 2048)

        if self.method == "dec":
            head = self._build_dec_head(embed_dim, k)
            self._init_dec(encoder, head, dataset)
            head_params = list(head.parameters())
            extra = list(encoder.proj_head.parameters())
        else:
            # Strict CC: two independent heads on backbone hidden; proj_head unused
            head = self._build_cc_heads(in_dim, k)
            head_params = list(head.parameters())
            extra = []
            print(f"[CC] Built strict two-head CC (instance + cluster), in_dim={in_dim}")

        # Collect trainable parameters
        trainable = (
            [p for p in encoder.backbone.parameters() if p.requires_grad]
            + extra
            + head_params
        )
        optimizer = AdamW(
            trainable, lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 0.01))
        )
        total_steps = max(n_epochs * len(dataset) // grad_accum, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10)

        best_score = -float("inf")   # higher silhouette = better
        p_cache: dict = {}
        global_step = 0

        for epoch in range(n_epochs):
            dataset.on_epoch_end()
            encoder.backbone.train()
            epoch_loss, n_batches = 0.0, 0
            optimizer.zero_grad()

            pbar = tqdm(range(len(dataset)), desc=f"Epoch {epoch+1}/{n_epochs} [{self.method}]")
            for step_i in pbar:
                refresh_p = (global_step % refresh_every == 0)
                try:
                    batch = dataset[step_i]
                    if self.method == "dec":
                        loss = self._dec_step(encoder, head, batch, p_cache, refresh_p)
                    else:
                        loss = self._cc_step(encoder, head, batch)
                    loss = loss / grad_accum
                    loss.backward()
                    epoch_loss += float(loss.item()) * grad_accum
                    n_batches += 1
                except Exception as e:
                    print(f"  [skip] step {step_i}: {e}")
                    optimizer.zero_grad()
                    global_step += 1
                    continue

                if (step_i + 1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (step_i + 1) % log_every == 0:
                    avg = epoch_loss / max(n_batches, 1)
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                global_step += 1

            # Flush remaining gradients
            nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss = epoch_loss / max(n_batches, 1)
            sil = self._silhouette_score(encoder, head, dataset)
            print(f"Epoch {epoch+1}/{n_epochs}: loss={train_loss:.4f}  silhouette={sil:.4f}")

            ep_dir = self.out_dir / f"epoch_{epoch+1:03d}"
            save_encoder_checkpoint(encoder, str(ep_dir))
            if self.method == "contrastive":
                torch.save(head.state_dict(), str(ep_dir / "cc_heads.pt"))

            if sil > best_score:
                best_score = sil
                best_dir = self.out_dir / "best"
                save_encoder_checkpoint(encoder, str(best_dir))
                if self.method == "contrastive":
                    torch.save(head.state_dict(), str(best_dir / "cc_heads.pt"))
                print(f"  -> new best silhouette={best_score:.4f}")

            with open(self.out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "silhouette": sil,
                }) + "\n")

        print(f"Done. Best silhouette={best_score:.4f}, checkpoints at {self.out_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    UnsupEncoderTrainer(cfg).train()
