"""Qwen2.5-VL model loading with optional LoRA.

Supports:
  - Qwen2.5-VL-3B-Instruct (default, fits in 24 GB with bf16 + LoRA)
  - Qwen2.5-VL-7B-Instruct (with QLoRA int4 if needed)

Usage:
    model, processor = load_model_and_processor(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    # For training with LoRA:
    model = add_lora(model, lora_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)


@dataclass
class LoRAConfig:
    """LoRA hyperparameters."""

    r: int = 64
    lora_alpha: int = 128
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def load_model_and_processor(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map: str | dict = "auto",
    load_in_4bit: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,   # kept for call-site compat
    attn_implementation: str = "flash_attention_2",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 512 * 28 * 28,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen2.5-VL model and processor.

    Parameters
    ----------
    model_id:
        HuggingFace model ID or local path.
    device_map:
        "auto" distributes across available GPUs.
    load_in_4bit:
        Enable QLoRA (4-bit quantisation via bitsandbytes). Use for 7B model.
    torch_dtype:
        bfloat16 by default; requires Ampere+ GPU (RTX 30xx/40xx).
    attn_implementation:
        "flash_attention_2" for speed; falls back to "eager" if unavailable.
    min_pixels / max_pixels:
        Controls Qwen2.5-VL's dynamic image resolution. Lower = fewer tokens.
        For 64×64 3DShapes images, min_pixels=256*28*28 is already quite large;
        the processor will downsample to at most max_pixels.
        Recommended for 3DShapes (64×64): min=64*28*28, max=256*28*28.
    """
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            quantization_config=bnb_config,
        )
    except Exception:
        # Flash attention may not be installed; fall back to eager
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
            quantization_config=bnb_config,
        )

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return model, processor


def add_lora(
    model: Qwen2_5_VLForConditionalGeneration,
    lora_cfg: LoRAConfig | None = None,
) -> Any:
    """Wrap model with PEFT LoRA adapters. Returns PeftModel."""
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

    if lora_cfg is None:
        lora_cfg = LoRAConfig()

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def save_lora_checkpoint(model: Any, path: str) -> None:
    """Save only LoRA adapter weights."""
    model.save_pretrained(path)
    print(f"Saved LoRA checkpoint to {path}")


def load_lora_checkpoint(
    base_model_id: str,
    lora_path: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, AutoProcessor]:
    """Load base model and merge LoRA weights."""
    from peft import PeftModel

    model, processor = load_model_and_processor(
        model_id=base_model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,  # noqa: argument name kept for API compat
    )
    model = PeftModel.from_pretrained(model, lora_path)
    return model, processor
