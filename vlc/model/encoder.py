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
