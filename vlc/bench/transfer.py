"""Zero-shot transfer evaluation: criterion-conditioned encoder vs CLIP-KMeans.

Protocol:
  1. Load images from held-out test split
  2. Encode all images under each criterion with CriterionEncoder -> (N, 256) embeddings
  3. KMeans(K) on embeddings -> cluster assignments
  4. Evaluate: Hungarian ACC / ARI / NMI
  5. Report cross-criteria ARI (lower = criteria produce more independent partitions)

Entry point: python -m vlc eval --config configs/vlm/eval_encoder_cifar10.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlc.core.metrics import evaluate


def _cross_criteria_ari(assignments_per_criterion: dict[str, list[int]]) -> float:
    from sklearn.metrics import adjusted_rand_score
    keys = list(assignments_per_criterion.keys())
    if len(keys) < 2:
        return float("nan")
    aris = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            aris.append(adjusted_rand_score(
                assignments_per_criterion[keys[i]],
                assignments_per_criterion[keys[j]],
            ))
    return float(np.mean(aris))


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_cifar10_images_labels(data_dir: str, crit_key: str, n_test: int, seed: int = 0) -> tuple[list, list[int]]:
    from vlc.episodes.cifar10 import CRITERIA, CIFAR10_CLASSES, CIFAR_TO_SUPER

    data_dir = Path(data_dir)
    with open(data_dir / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    raw_imgs = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    cifar_labels = np.array(d[b"labels"])

    crit = CRITERIA[crit_key]
    c2s = crit["class_to_super"]
    class_names = CIFAR10_CLASSES
    super_labels = np.array([c2s[class_names[int(l)]] for l in cifar_labels])

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(raw_imgs), size=min(n_test, len(raw_imgs)), replace=False)
    from PIL import Image
    images = [Image.fromarray(raw_imgs[i]).resize((64, 64)) for i in chosen]
    labels = (super_labels[chosen] + 1).tolist()
    return images, labels


def load_cifar100_images_labels(data_dir: str, crit_key: str, n_test: int, seed: int = 0) -> tuple[list, list[int]]:
    from vlc.episodes.cifar100 import CRITERIA

    data_dir = Path(data_dir)
    with open(data_dir / "test", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    raw_imgs = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    coarse_labels = np.array(d[b"coarse_labels"])

    c2s = CRITERIA[crit_key]["coarse_to_super"]
    super_labels = np.array([c2s[int(c)] for c in coarse_labels])

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(raw_imgs), size=min(n_test, len(raw_imgs)), replace=False)
    from PIL import Image
    images = [Image.fromarray(raw_imgs[i]).resize((64, 64)) for i in chosen]
    labels = (super_labels[chosen] + 1).tolist()
    return images, labels


# ── Method runners ────────────────────────────────────────────────────────────

def run_vlc(
    images: list[Any],
    criterion: str,
    k: int,
    encoder: Any,
    batch_size: int = 8,
    **_kwargs,
) -> tuple[list[int], dict]:
    from vlc.embed.cluster import VectorClusterer
    result = VectorClusterer(encoder, batch_size).cluster(images, criterion, k)
    return result.assignments, {"embed_shape": list(result.embeddings.shape)}


def run_clip_kmeans(
    images: list[Any],
    criterion: str,
    k: int,
    clip_model: str = "ViT-B-32",
) -> tuple[list[int], dict]:
    import torch
    import open_clip
    from sklearn.cluster import KMeans

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(clip_model)

    img_tensors = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        img_feats = model.encode_image(img_tensors)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_feat = model.encode_text(tokenizer([criterion]).to(device))
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    img_np = img_feats.cpu().float().numpy()
    txt_np = text_feat.cpu().float().numpy()[0]
    proj = img_np @ txt_np[:, None]
    ortho = img_np - proj * txt_np[None, :]
    combined = np.concatenate([5.0 * proj, ortho], axis=1)

    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(combined)
    return (labels + 1).tolist(), {}


# ── Main evaluator ────────────────────────────────────────────────────────────

class TransferEvaluator:

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.get("output_dir", "artifacts/vlm/eval"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        import torch
        eval_cfg = self.cfg["eval"]
        dataset = eval_cfg["dataset"]
        criteria = eval_cfg["criteria"]
        k = eval_cfg["k"]
        methods = eval_cfg.get("methods", ["vlc", "clip_kmeans"])
        n_test = eval_cfg.get("n_test", 200)

        all_results: dict[str, dict] = {}
        vlc_asgn: dict[str, list[int]] = {}
        clip_asgn: dict[str, list[int]] = {}

        # Load encoder once
        encoder = None
        if "vlc" in methods:
            from vlc.model.encoder import load_encoder_checkpoint, build_encoder
            model_id = eval_cfg.get("vlc_model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
            lora_dir = eval_cfg.get("vlc_lora_dir")
            use_4bit = eval_cfg.get("load_in_4bit", True)
            print(f"[eval] Loading encoder (4bit={use_4bit})...")
            if lora_dir and (Path(lora_dir) / "lora").exists():
                encoder = load_encoder_checkpoint(
                    model_id=model_id,
                    lora_path=str(Path(lora_dir) / "lora"),
                    proj_path=str(Path(lora_dir) / "proj_head.pt"),
                    load_in_4bit=use_4bit,
                )
            else:
                encoder = build_encoder(model_id=model_id, load_in_4bit=use_4bit)
            encoder.eval()

        for crit_key in criteria:
            print(f"\n{'='*60}\nCriterion: {crit_key}")
            images, gt_labels = self._load_data(dataset, crit_key, eval_cfg, n_test)
            crit_text = self._get_instruction(dataset, crit_key)
            all_results[crit_key] = {}

            for method in methods:
                t0 = time.time()
                try:
                    if method == "vlc":
                        assignments, meta = run_vlc(
                            images, crit_text, k, encoder,
                            batch_size=eval_cfg.get("batch_size", 8),
                        )
                        vlc_asgn[crit_key] = assignments
                    elif method == "clip_kmeans":
                        assignments, meta = run_clip_kmeans(images, crit_text, k)
                        clip_asgn[crit_key] = assignments
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    metrics = evaluate(np.array(gt_labels) - 1, np.array(assignments) - 1)
                    elapsed = time.time() - t0
                    all_results[crit_key][method] = {**metrics, "time_s": elapsed, **meta}
                    print(f"  [{method}] ACC={metrics['acc']:.3f} ARI={metrics['ari']:.3f} "
                          f"NMI={metrics['nmi']:.3f}  ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  [{method}] FAILED: {e}")
                    all_results[crit_key][method] = {"error": str(e)}

        if encoder is not None:
            del encoder
            torch.cuda.empty_cache()

        if len(vlc_asgn) >= 2:
            print(f"\nCross-criteria ARI (VLC, ↓ better): {_cross_criteria_ari(vlc_asgn):.4f}")
        if len(clip_asgn) >= 2:
            print(f"Cross-criteria ARI (CLIP, ↓ better): {_cross_criteria_ari(clip_asgn):.4f}")

        out_path = self.out_dir / f"transfer_{dataset}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")
        self._render_table(all_results, dataset)

    def _load_data(self, dataset: str, crit_key: str, eval_cfg: dict, n_test: int):
        if dataset == "cifar10":
            return load_cifar10_images_labels(eval_cfg["data_dir"], crit_key, n_test)
        if dataset == "cifar100":
            return load_cifar100_images_labels(eval_cfg["data_dir"], crit_key, n_test)
        raise ValueError(f"Unknown dataset: {dataset}")

    def _get_instruction(self, dataset: str, crit_key: str) -> str:
        if dataset == "cifar10":
            from vlc.episodes.cifar10 import CRITERIA
            return CRITERIA[crit_key]["instruction"]
        if dataset == "cifar100":
            from vlc.episodes.cifar100 import CRITERIA
            return CRITERIA[crit_key]["instruction"]
        raise ValueError(f"Unknown dataset: {dataset}")

    def _render_table(self, results: dict, dataset: str) -> None:
        methods = sorted({m for r in results.values() for m in r})
        criteria = list(results.keys())
        header = "| criterion | " + " | ".join(f"{m} ACC / ARI / NMI" for m in methods) + " |"
        sep = "|---|" + "---|" * len(methods)
        rows = [header, sep]
        for crit in criteria:
            row = f"| {crit} |"
            for m in methods:
                r = results[crit].get(m, {})
                if "error" in r:
                    row += " ERR |"
                else:
                    row += f" {r.get('acc',0):.3f} / {r.get('ari',0):.3f} / {r.get('nmi',0):.3f} |"
            rows.append(row)
        print("\n" + "\n".join(rows))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    TransferEvaluator(cfg).run()
