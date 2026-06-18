# VLC Research Protocol (v2 — End-to-End VLM Streaming Clustering)

## 1. Problem Statement

**Vision-Language Clustering (VLC)** treats set partitioning as a learnable multimodal
capability of a vision-language model:

> **Input:** image set \(\mathcal{X} = \{x_i\}_{i=1}^N\), criterion text \(g\), cluster count \(K\)
> **Output:** partition \(\mathcal{C} = \{C_1,\ldots,C_K\}\) + natural-language cluster summaries

Unlike prior deep clustering (DEC, SCAN) the criterion is a natural-language string;
unlike prior LLM-for-clustering work (ClusterLLM, ICTC) we *train* the VLM to acquire
this capability from multi-attribute synthetic data, then evaluate **zero-shot** on
unseen datasets and unseen criteria.

Unlike the previous codebase (frozen CLIP vectors → small MLP), raw image pixels enter
the VLM directly.

---

## 2. Method: VLM Streaming Clustering

```
Episode loop (train) / Inference loop:
┌────────────────────────────────────────────────────────────────┐
│  Step t:                                                        │
│    Input:  criterion g, K, cluster cards (text), batch images  │
│    Output: JSON { assignments: [...], cards: [...] }           │
│  ─────────────────────────────────────────────────────────────  │
│  Step t+1: updated cards become new context                    │
└────────────────────────────────────────────────────────────────┘
```

**Cluster cards** are K text slots the model writes and reads:
```
C1 | sphere-shaped objects | 8 images
C2 | cube-shaped objects   | 6 images
```
Cards are the *only* cross-batch memory — context length stays O(K) regardless of N.

**SFT training:**
- Episodes synthesised from 3DShapes, CLEVR (factor-annotated synthetic data)
- Each episode = multi-turn Qwen2.5-VL conversation; loss on assistant tokens only
- Random image ordering per episode during training (→ order robustness)
- Card supervision text generated from factor value templates

**Inference:**
- Multi-order ensemble (N=3): run with N random orderings, majority-vote assignments
- JSON constraint decoding with automatic retry on malformed output

---

## 3. Hypotheses

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | Trained VLM streaming clustering outperforms CLIP+KMeans on held-out 3DShapes criteria | Transfer table, `transfer_3dshapes.json` |
| H2 | Same model, different criterion → near-zero cross-criteria ARI (instruction switching) | Cross-criteria ARI in transfer eval |
| H3 | Multi-order ensemble reduces order sensitivity vs. single run | `order_sensitivity` field in eval output |
| H4 | Zero-shot generalisation: training on 3DShapes/CLEVR → works on Cars/CUB/Flowers | Held-out dataset eval |
| H5 | Cluster cards improve interpretability vs. anonymous cluster IDs (human eval) | Appendix |

---

## 4. Experimental Setup

### Training data
| Dataset | Criteria | K |
|---------|----------|---|
| 3DShapes | object_hue, shape, object_size, orientation | 4 |
| CLEVR | color_4, shape, material | 2–4 |
| CelebA (phase 2) | hair_color, eyeglasses, smiling, ... | 2–8 |

### Held-out evaluation data (zero-shot)
| Dataset | Criteria | K |
|---------|----------|---|
| 3DShapes (test split) | same 4 criteria | 4 |
| Stanford Cars | color vs. car type | 4 |
| CUB-200 | species group, wing color, etc. | varies |
| Flowers-102 | species group | 4 |

### Metrics
- **ACC** — Hungarian-aligned cluster accuracy
- **ARI** — Adjusted Rand Index
- **NMI** — Normalised Mutual Information
- **cross-criteria ARI** (↓ better) — mean pairwise ARI between partitions from different criteria on the same images
- **order sensitivity** — mean pairwise ARI across random orderings of the same images (↑ better = more stable)

### Baselines
| Method | Description |
|--------|-------------|
| CLIP+KMeans | CLIP ViT-B-32 features, text-guided projection, KMeans |
| ICTC | Iterative constraint-based clustering (LLM generated pairs) |
| GPT-4o-mini | Zero-shot prompting, no training, few images per API call |
| VLC-lite | Legacy: batch ClusterDecoder on frozen CLIP features (our prior baseline) |
| SCAN-lite | kNN graph + Laplacian + KMeans |
| DEC-lite | Deep embedded clustering on CLIP features |

---

## 5. How to Run

```bash
# 1. Verify episode synthesis pipeline
python -m vlc build-episodes --config configs/vlm/build_episodes_preview.yaml --n 3

# 2. Overfit check (verify training loss decreases on 5 episodes)
python -m vlc train-sft --config configs/vlm/shapes3d_sft_overfit.yaml

# 3. Full 3DShapes training
python -m vlc train-sft --config configs/vlm/shapes3d_sft.yaml

# 4. Evaluation (zero-shot transfer on 3DShapes test split)
python -m vlc eval --config configs/vlm/eval_shapes3d.yaml

# 5. Export paper tables
python -m vlc export-tables --eval-dir artifacts/vlm/eval

# Legacy baseline (for comparison table)
python -m vlc train-baseline --config configs/lite/shapes3d_vlm.yaml
python -m vlc same-k --config configs/lite/shapes3d_same_k.yaml
```

---

## 6. Limitations (to state honestly in the paper)

- Training data is synthetic (rendered 3D scenes); generalisation to in-the-wild photos
  may require additional real-data fine-tuning.
- K is a required input; automatic K estimation is left for future work.
- Card names are English templates during training; paraphrase robustness not yet tested.
- VLM training requires ~24 GB VRAM (RTX 4090 or equivalent).

---

## 7. File Map

```
vlc/episodes/         Episode synthesis (3DShapes, CLEVR, catalog)
vlc/model/            Qwen2.5-VL + LoRA loading, SFT trainer
vlc/stream/           Streaming inference, cluster card state, ensemble
vlc/bench/transfer.py Zero-shot transfer evaluation
vlc/bench/same_k.py   Same-K multi-criteria benchmark (legacy baseline path)
vlc/baseline/         VLC-lite ClusterDecoder (baseline)
vlc/core/metrics.py   Hungarian ACC / ARI / NMI
configs/vlm/          Training and eval YAML configs
artifacts/vlm/        Checkpoints and evaluation outputs
```
