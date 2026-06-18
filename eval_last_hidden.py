"""
零样本评测：Qwen2.5-VL 最后一个 token 隐状态 → KMeans → ACC/NMI/ARI。
不做任何训练，纯推理。

用法:
    python eval_last_hidden.py --n_per_class 20
    python eval_last_hidden.py --n_per_class 100
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "/home/yaner/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
DATA_DIR = Path("/home/yaner/kris/warehouse/DiEC/data/cifar-10-batches-py")

INSTRUCTION = (
    "What is the main object in this image? "
    "Choose from: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
)


@torch.no_grad()
def extract_hidden(model, processor, images: list, device) -> np.ndarray:
    """逐张前向，add_generation_prompt=True，取最后一层最后 token 隐状态。

    返回 (N, hidden_dim) float32 numpy。
    """
    all_h = []
    for img in tqdm(images, desc="extracting"):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{INSTRUCTION}\nAnswer with the category name only."},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        out = model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        all_h.append(h.numpy())
    return np.stack(all_h, axis=0)


def cluster_acc(y_true, y_pred):
    k = max(y_true.max(), y_pred.max()) + 1
    W = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        W[p, t] += 1
    ri, ci = linear_sum_assignment(W.max() - W)
    return W[ri, ci].sum() / len(y_true)


def load_cifar10(data_dir: Path, n_per_class: int):
    imgs_list, labels_list = [], []
    for i in range(1, 6):
        with open(data_dir / f"data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs_list.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels_list.extend(d[b"labels"])
    imgs = np.concatenate(imgs_list, axis=0)
    labels = np.array(labels_list)
    keep = []
    for c in range(10):
        keep.append(np.where(labels == c)[0][:n_per_class])
    keep = np.sort(np.concatenate(keep))
    return imgs[keep], labels[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=20)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="可选：加载训练好的 LoRA 权重路径")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 加载模型 ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )
    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"[*] 加载 LoRA: {args.lora_path}")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[*] 加载 CIFAR-10，每类 {args.n_per_class} 张 ...")
    imgs_np, labels = load_cifar10(DATA_DIR, args.n_per_class)
    images = [Image.fromarray(a).resize((64, 64), Image.BILINEAR) for a in imgs_np]

    print("[*] 提取隐状态 ...")
    feats = extract_hidden(model, processor, images, device)
    print(f"    特征矩阵: {feats.shape}")

    feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    print("[*] KMeans (K=10) ...")
    km = KMeans(n_clusters=10, n_init=20, random_state=42, max_iter=300)
    km.fit(feats_norm)
    pred = km.labels_

    acc = cluster_acc(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    ari = adjusted_rand_score(labels, pred)

    print(f"\n{'='*40}")
    print(f"  ACC = {acc:.4f}")
    print(f"  NMI = {nmi:.4f}")
    print(f"  ARI = {ari:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
