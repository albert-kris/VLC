"""Criterion-conditioned visual encoder.

Architecture:
  Qwen2.5-VL-3B (frozen base, trainable LoRA) + trainable ProjectionHead.

Forward: (criterion_text, image) -> L2-normalized vector z in R^256.

Key design: add_generation_prompt=True so the sequence ends with
<|im_start|>assistant\\n. That final token's last-layer hidden state
encodes "model has seen image+question and is about to answer with the
category name" — empirically ACC jumps from 0.27 to 0.77 vs using the
input-sequence tail token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ProjectionHead(nn.Module):
    """Two-layer MLP: in_dim -> mid_dim -> out_dim, with L2 normalization."""

    def __init__(self, in_dim: int = 2048, mid_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CriterionEncoder(nn.Module):
    """Criterion-conditioned visual encoder.

    Uses the last-layer hidden state of the final token position (which is
    the assistant-turn start token when add_generation_prompt=True) as the
    image representation. No learnable [cluster] token needed.
    """

    def __init__(self, backbone: Any, processor: Any, proj_head: ProjectionHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.processor = processor
        self.proj_head = proj_head
        self._device = next(backbone.parameters()).device

    @property
    def device(self) -> torch.device:
        return self._device

    def _build_message(self, criterion: str, image: Image.Image) -> list[dict]:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"{criterion}\n"
                    f"Answer with the category name only."
                )},
            ],
        }]

    def _get_hidden(self, criterion: str, image: Image.Image) -> torch.Tensor:
        """Return last-layer hidden state of the final token. Shape: (hidden_dim,).

        With add_generation_prompt=True the sequence ends at <|im_start|>assistant\\n.
        That position has attended over all image + instruction tokens and
        encodes the model's decision about which category the image belongs to.
        """
        messages = self._build_message(criterion, image)
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        # (1, L, H) → take the final position → (H,)
        pooled = outputs.hidden_states[-1][0, -1, :]
        return pooled.float()

    def encode_one(self, criterion: str, image: Image.Image) -> torch.Tensor:
        """Returns z of shape (out_dim,), L2-normalized."""
        h = self._get_hidden(criterion, image)
        return self.proj_head(h.unsqueeze(0)).squeeze(0)

    def encode_hidden_one(self, criterion: str, image: Image.Image) -> torch.Tensor:
        """Returns raw last hidden state (hidden_dim,) before proj_head."""
        return self._get_hidden(criterion, image)

    def encode_batch(self, criterion: str, images: list[Image.Image]) -> torch.Tensor:
        """Encode a list of images. Returns (N, out_dim) L2-normalized."""
        return torch.stack([self.encode_one(criterion, img) for img in images], dim=0)

    def encode_hidden_batch(self, criterion: str, images: list[Image.Image]) -> torch.Tensor:
        """Return raw hidden states (N, hidden_dim)."""
        return torch.stack([self.encode_hidden_one(criterion, img) for img in images], dim=0)

    # ---- DeepSeek 式：生成推理链 + 重新打分 -------------------------------

    def _build_inputs(self, criterion: str, image: Image.Image) -> tuple[dict, int]:
        """构造单图输入（含 add_generation_prompt），返回 (inputs, 提示词长度 L)。"""
        messages = self._build_message(criterion, image)
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        return inputs, int(inputs["input_ids"].shape[1])

    @torch.no_grad()
    def generate_chains(
        self,
        criterion: str,
        image: Image.Image,
        n_samples: int,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> tuple[dict, int, list[torch.Tensor]]:
        """Rollout：对一张图生成推理链（no_grad）。

        do_sample=True 时采样 n_samples 条不同的链（训练 rollout）；
        do_sample=False 时贪心生成 1 条（评估用，n_samples 被忽略）。

        返回 (inputs, L, gen_ids_list)：
          inputs       —— 提示词输入（含 pixel_values），score_chains 复用
          L            —— 提示词 token 数
          gen_ids_list —— 每条链新生成的 token id（1D LongTensor，不含提示词）
        """
        inputs, L = self._build_inputs(criterion, image)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, return_dict_in_generate=True)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature,
                              num_return_sequences=n_samples)
        else:
            gen_kwargs.update(do_sample=False)
        gen = self.backbone.generate(**inputs, **gen_kwargs)
        seqs = gen.sequences                       # (n, L+T)
        gen_ids = [seqs[i, L:].detach() for i in range(seqs.shape[0])]
        return inputs, L, gen_ids

    def _end_token_ids(self) -> set[int]:
        tok = self.processor.tokenizer
        return {x for x in [tok.eos_token_id, tok.convert_tokens_to_ids("")] if x is not None}

    def _hidden_before_im_end(
        self, hs: torch.Tensor, L: int, gen_ids: torch.Tensor,
    ) -> torch.Tensor:
        """取 im_end/EOS 前一个生成 token 的最后一层 hidden。hs shape (L+T, H)。"""
        gt = [int(t) for t in gen_ids]
        eos_ids = self._end_token_ids()
        j = len(gt) - 1
        while j >= 0 and gt[j] in eos_ids:
            j -= 1
        return hs[L + j] if j >= 0 else hs[-1]

    @torch.no_grad()
    def encode_chain_hidden_one(
        self, criterion: str, image: Image.Image, max_new_tokens: int = 24,
    ) -> torch.Tensor:
        """generate → forward 全序列 → 返回 im_end 前一 token 的 hidden (H,)。"""
        inp, L, gen = self.generate_chains(
            criterion, image, 1, max_new_tokens, do_sample=False)
        _, h = self.score_chains(inp, L, gen)
        return h[0]

    def score_chains(
        self,
        inputs: dict,
        L: int,
        gen_ids: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """带梯度重新 forward「提示词+链」，返回 (sum_logp, chain_hidden)。

          sum_logp      —— (G,) 每条链生成 token 的 log p 之和（可导，训 LoRA）
          chain_hidden  —— (G, hidden) im_end 前一 token 的最后一层 hidden（给聚类用）

        因果 mask 保证位置 i 的 logits 只依赖前缀，故一次整段 forward 即可还原
        每个生成位置当时的条件概率（teacher forcing）。
        """
        base_ids = inputs["input_ids"][0]          # (L,)
        base_mmtype = inputs.get("mm_token_type_ids")
        # input_ids / attention_mask / mm_token_type_ids 随序列变长，需重建；
        # 其余（pixel_values / image_grid_thw）是图像侧、与序列长度无关，原样透传
        extra = {k: v for k, v in inputs.items()
                 if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
        logps, hiddens = [], []
        for g in gen_ids:
            T = int(g.shape[0])
            full_ids = torch.cat([base_ids, g], dim=0).unsqueeze(0)   # (1, L+T)
            attn = torch.ones_like(full_ids)
            kw = dict(extra)
            if base_mmtype is not None:
                # 生成 token 全是文本（type 0），在末尾补 T 个 0
                pad = torch.zeros((1, T), dtype=base_mmtype.dtype, device=base_mmtype.device)
                kw["mm_token_type_ids"] = torch.cat([base_mmtype, pad], dim=1)
            out = self.backbone(
                input_ids=full_ids,
                attention_mask=attn,
                **kw,
                output_hidden_states=True,
                return_dict=True,
            )
            # 位置 L-1 .. L+T-2 的 logits 预测生成 token g[0..T-1]
            pred_logits = out.logits[0, L - 1:L - 1 + T, :]
            logp = F.log_softmax(pred_logits.float(), dim=-1)
            token_logp = logp.gather(1, g.view(-1, 1)).squeeze(1)     # (T,)
            logps.append(token_logp.sum())
            hs = out.hidden_states[-1][0].float()
            hiddens.append(self._hidden_before_im_end(hs, L, g))
        return torch.stack(logps), torch.stack(hiddens)

    def forward(self, criterion: str, images: list[Image.Image]) -> torch.Tensor:
        return self.encode_batch(criterion, images)


def build_encoder(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    lora_path: str | None = None,
    proj_in_dim: int = 2048,
    proj_mid_dim: int = 512,
    proj_out_dim: int = 256,
    load_in_4bit: bool = False,
) -> CriterionEncoder:
    from vlc.model.qwen_vl import load_model_and_processor, add_lora

    backbone, processor = load_model_and_processor(model_id=model_id, load_in_4bit=load_in_4bit)
    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, lora_path, is_trainable=True)
    else:
        backbone = add_lora(backbone)

    device = next(backbone.parameters()).device
    proj_head = ProjectionHead(proj_in_dim, proj_mid_dim, proj_out_dim).to(device)
    return CriterionEncoder(backbone, processor, proj_head)


def load_encoder_checkpoint(
    model_id: str,
    lora_path: str,
    proj_path: str,
    proj_in_dim: int = 2048,
    proj_mid_dim: int = 512,
    proj_out_dim: int = 256,
    load_in_4bit: bool = False,
) -> CriterionEncoder:
    from vlc.model.qwen_vl import load_model_and_processor
    from peft import PeftModel

    backbone, processor = load_model_and_processor(model_id=model_id, load_in_4bit=load_in_4bit)
    backbone = PeftModel.from_pretrained(backbone, lora_path)
    device = next(backbone.parameters()).device
    proj_head = ProjectionHead(proj_in_dim, proj_mid_dim, proj_out_dim)
    proj_head.load_state_dict(torch.load(proj_path, map_location="cpu", weights_only=True))
    proj_head = proj_head.to(device)
    return CriterionEncoder(backbone, processor, proj_head)


def save_encoder_checkpoint(encoder: CriterionEncoder, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    encoder.backbone.save_pretrained(str(out / "lora"))
    torch.save(encoder.proj_head.state_dict(), str(out / "proj_head.pt"))
    print(f"Saved encoder checkpoint to {out}")
