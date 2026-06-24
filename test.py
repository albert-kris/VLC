"""
教学脚本：从一张图像到最终聚类分配，逐步打印每个环节的维度和数值。

                   token1   token2   ...   tokenL
                     ↓        ↓              ↓
embedding层:       (2048)   (2048)  ...   (2048)   → hidden_states[0],  shape(1,L,2048)
第1层 Transformer: (2048)   (2048)  ...   (2048)   → hidden_states[1],  shape(1,L,2048)
第2层 Transformer: (2048)   (2048)  ...   (2048)   → hidden_states[2],  shape(1,L,2048)
...
第28层Transformer: (2048)   (2048)  ...   (2048)   → hidden_states[28], shape(1,L,2048)
                                                            ↑ 取这一层的最后一列
数据流：
  图像 + 提示词
      ↓  预处理
  预处理后的值(L个ID + 图像patch)
      ↓  丢进大模型
  生成29组隐状态(embedding层 + 28个Transformer层), 每组 shape(1,L,2048)
      ↓  取最后一个 token 的最后一层隐状态 (2048,), 编码到词表得到下一个 ID
  拼接回序列然后重新丢回去自回归  输入：(1,L+1) (图像被缓存了, 不进入拼接)
      ↓ 实际可以缓存KV输入: (1,1)
  自回归反复如此直到生成完毕, 得到最后一个 token 的隐状态 (2048,)
      ↓  进入聚类头 ProjectionHead 得到 z: (256,)
      ↓  z 只是一个向量不能聚类, 先 SwAVHead 打分得到 scores: (1, K)
      ↓  把 scores 变成软分配 q: (1, K)
      ↓  argmax 得到 cluster_id
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw

# ── 配置（按需修改）────────────────────────────────────────────────────────────
# 优先读环境变量 QWEN_MODEL_ID，没有则自动在 HuggingFace 缓存里查找，
# 找不到就回退到在线拉取。
def _find_model_id() -> str:
    env = os.environ.get("QWEN_MODEL_ID")
    if env:
        return env
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    pattern = "models--Qwen--Qwen2.5-VL-3B-Instruct"
    snap = cache_root / pattern / "snapshots"
    candidates = sorted(snap.glob("*")) if snap.exists() else []
    if candidates:
        return str(candidates[-1])
    return "Qwen/Qwen2.5-VL-3B-Instruct"

MODEL_ID = _find_model_id()
K = 10   # 聚类数

def sep(title=""):
    print("\n" + "═" * 60)
    if title:
        print(f"  {title}")
        print("─" * 60)


# ══════════════════════════════════════════════════════════════
# 0. 输入：图像 + 提示词
# ══════════════════════════════════════════════════════════════
sep("0. 输入：图像 + 提示词")

img = Image.new("RGB", (64, 64), color=(120, 180, 220))   # 纯色块，仅演示
ImageDraw.Draw(img).rectangle([10, 10, 54, 54], fill=(255, 100, 50))

instruction = (
    "What is the main object in this image? "
    "Choose from: airplane, automobile, bird, cat, deer, dog, "
    "frog, horse, ship, truck."
)
print(f"图像 size : {img.size}  mode={img.mode}")
print(f"提示词    : {instruction[:60]}...")


# ══════════════════════════════════════════════════════════════
# 1. 加载 Qwen2.5-VL
# ══════════════════════════════════════════════════════════════
sep("1. 加载 Qwen2.5-VL-3B 模型")

from vlc.model.qwen_vl import load_model_and_processor
from vlc.model.encoder import ProjectionHead
from vlc.model.swav_head import SwAVHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

backbone, processor = load_model_and_processor(model_id=MODEL_ID, device_map={"": device})
backbone.eval()


# ══════════════════════════════════════════════════════════════
# 2. 预处理：图像 + 文本 → 模型输入（ID + 图像 patch）
# ══════════════════════════════════════════════════════════════
sep("2. 预处理")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": f"{instruction}\nAnswer with the category name only."},
    ],
}]
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[img], return_tensors="pt")
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

L = inputs["input_ids"].shape[1]
print(f"input_ids    : {tuple(inputs['input_ids'].shape)}   ← {L} 个 token（整数ID：文字+图像占位）")
if "pixel_values" in inputs:
    print(f"pixel_values : {tuple(inputs['pixel_values'].shape)}   ← 图像被切成 patch 的像素")


# ══════════════════════════════════════════════════════════════
# 3. 自回归生成：一个一个 token 地吐，直到结束
# ══════════════════════════════════════════════════════════════
sep("3. 自回归生成（每生成一个词，就把它拼回序列再 forward 一次）")

with torch.no_grad():
    gen = backbone.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

# gen.sequences: 完整序列（输入 + 新生成的 token）
new_tokens = gen.sequences[0, L:]                       # 只看新生成的部分
answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"输入长度 L            : {L}")
print(f"新生成 token 数        : {new_tokens.shape[0]}")
print(f"模型生成的答案文字     : {answer!r}")
print(f"模型循环了 {len(gen.hidden_states)次}")
print(f"  第1步整段输入每层输出 shape: {tuple(gen.hidden_states[0][-1].shape)}  ← (1, L, 2048)")
if len(gen.hidden_states) > 1:
    print(f"  之后每步每层输出 shape: {tuple(gen.hidden_states[1][-1].shape)}  ← (1, 1, 2048)")


# ══════════════════════════════════════════════════════════════
# 4. 取「最后一个生成 token」的最后一层隐状态
# ══════════════════════════════════════════════════════════════
sep("4. 取最后一个 token 的最后一层隐状态")

# gen.hidden_states[-1] = 最后一个生成步；[-1] = 最后一层(第28层)；[0, -1, :] = 最后一个位置
h = gen.hidden_states[-1][-1][0, -1, :].float()        # (2048,)
print(f"h shape : {tuple(h.shape)}   ← 2048 维，编码了模型读完图+问题后的语义")
print(f"h 前6维 : {[round(v, 3) for v in h[:6].tolist()]}")


# ══════════════════════════════════════════════════════════════
# 5. ProjectionHead：2048 → 256，压到单位球面
# ══════════════════════════════════════════════════════════════
sep("5. ProjectionHead：2048 → 512 → 256，L2 归一化")

proj_head = ProjectionHead(in_dim=2048, mid_dim=512, out_dim=256).to(device)
proj_head.eval()
with torch.no_grad():
    z = proj_head(h.unsqueeze(0)).squeeze(0)           # (256,)
print(f"z shape   : {tuple(z.shape)}")
print(f"z 的 L2 范数: {z.norm().item():.4f}   ← 始终=1，在单位球面上")


# ══════════════════════════════════════════════════════════════
# 6. SwAVHead：z 与 K 个原型的余弦相似度
# ══════════════════════════════════════════════════════════════
sep("6. SwAVHead：z @ prototypes^T 得到 scores")

swav_head = SwAVHead(dim=256, n_prototypes=K).to(device)
swav_head.normalize_prototypes()
with torch.no_grad():
    scores = swav_head(z.unsqueeze(0))                 # (1, K)
print(f"prototypes : {tuple(swav_head.prototypes.weight.shape)}   ← K 个原型，每个 256 维")
print(f"scores     : {tuple(scores.shape)}   值: {[round(v, 3) for v in scores[0].tolist()]}")
print(f"  → 余弦相似度，越大越像那个簇（当前随机原型，训练后才有语义）")


# ══════════════════════════════════════════════════════════════
# 7. Sinkhorn：scores → 软分配 q → argmax
# ══════════════════════════════════════════════════════════════
sep("7. Sinkhorn 软分配 + argmax")

def sinkhorn(scores, epsilon=0.05, n_iters=3):
    Q = torch.exp(scores / epsilon).t()
    Q = Q / Q.sum().clamp(min=1e-8)
    K_dim, B = Q.shape
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-8) / K_dim   # 行归一：每原型等质量
        Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-8) / B       # 列归一：每图码和为1
    return (Q * B).t()

with torch.no_grad():
    q = sinkhorn(scores)                               # (1, K)
cluster_id = q.argmax(dim=1).item()
print(f"q          : {tuple(q.shape)}   值: {[round(v, 3) for v in q[0].tolist()]}")
print(f"cluster_id : {cluster_id}   ← 这张图被分到第 {cluster_id} 簇")

print("\n（注：Sinkhorn 的等分约束在 batch 内才明显，单张图只演示流程；"
      "真实训练时一个 batch 一起算，强制每个簇被平均使用，防止聚类塌缩。）")
print("\n脚本运行完毕。")
