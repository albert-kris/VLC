# VLC Paper Outline (v2)

## Title Candidates

1. **VLC: Learning to Cluster by Instruction — End-to-End Vision-Language Clustering**
2. **Streaming Visual Clustering with Language-Guided Memory**
3. **Clustering as a Learnable Multimodal Capability**

---

## Abstract (skeleton)

We address instruction-conditioned visual clustering: given a set of images and a
natural-language criterion, partition the images into K clusters that respect the
criterion. Prior work either uses frozen embeddings with post-hoc KMeans (no
language interface) or prompts large proprietary VLMs without training (limited
accuracy). We present **VLC**, which frames streaming clustering as a *learnable*
VLM capability. A Qwen2.5-VL backbone processes images batch-by-batch, maintaining
compact **cluster cards** (natural-language cluster summaries) as cross-batch
memory. Trained via supervised fine-tuning on synthetic multi-attribute episodes,
VLC generalises **zero-shot** to unseen datasets and unseen criteria. Experiments
on 3DShapes, CLEVR, and held-out real datasets show VLC achieves state-of-the-art
ACC/ARI while producing interpretable cluster summaries and near-zero
cross-criteria ARI (strong instruction switching).

---

## 1. Introduction

**Gap:** Deep clustering is standard (DEC, SCAN, contrastive methods) but
criterion-agnostic. Recent LLM-for-clustering work (ClusterLLM, ICTC) uses language
but doesn't train a model to cluster — it prompts general-purpose LLMs.

**Challenge list:**
- Permutation invariance of cluster IDs
- Cross-batch state without blowing up context length
- Zero-shot generalisation from synthetic training to real images
- Order sensitivity of sequential VLM inference

**Contributions (4 bullets):**
1. *VLM streaming clustering*: a trained model that processes images batch-by-batch
   with natural-language criterion and cluster card memory
2. *Cluster cards*: a compact, interpretable cross-batch memory that scales O(K)
3. *Multi-attribute SFT*: synthetic episode synthesis from factor-annotated datasets
   enables zero-shot criterion generalisation
4. *Multi-order ensemble*: run N orderings + majority vote eliminates order sensitivity

---

## 2. Related Work

- Deep clustering: DEC, SCAN, DINO-cluster, contrastive clustering
- LLM for clustering: ClusterLLM, LLM-MemCluster, ICTC
- VLM: CLIP, Qwen2.5-VL, instruction tuning
- Sequential / streaming: VLA (RT-2), Chain-of-Thought for structured prediction

---

## 3. Method

### 3.1 Problem Formulation
\(\pi = f_\theta(\mathcal{X}, g, K)\) where \(g\) is natural language.
Permutation invariance: cluster IDs are labels, not indices.

### 3.2 Streaming Architecture
- Qwen2.5-VL-3B backbone (frozen + LoRA adapters)
- Multi-image interleaved input per step
- Cluster card format and update rule

### 3.3 Episode Synthesis and SFT
- Factor-annotated datasets → streaming episodes
- Random batch ordering during training (order robustness)
- Card supervision text from factor value templates
- Answer-token-only loss; bf16 + gradient checkpointing

### 3.4 Inference: Multi-Order Ensemble
- N random orderings, majority vote with Hungarian alignment
- JSON constraint decoding with retry

---

## 4. Experiments

- **Setup:** 3DShapes train split → model; 3DShapes test split + zero-shot datasets → eval
- **Table 1:** Main results — ACC / ARI / NMI for all methods × all criteria
- **Table 2:** Cross-criteria ARI (instruction switching quality)
- **Table 3:** Ablations — no ensemble vs. ensemble; no cards vs. cards
- **Figure 1:** Method diagram (episode loop with cluster cards)
- **Figure 2:** PCA of assignments for different criteria (same images → different partitions)
- **Figure 3:** Cluster card examples (interpretability)

---

## 5. Discussion & Limitations

- Synthetic-to-real generalisation gap
- K must be provided
- English criterion only (multilingual future work)

---

## 6. Conclusion

---

## Appendix

- Full episode format and Qwen-VL chat template
- Cluster card supervision text templates
- Hyperparameters (YAML configs)
- Human evaluation of cluster card quality
