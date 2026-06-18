import random
from pathlib import Path

import numpy as np
import torch
import yaml


_FLOAT_KEYS = {
    "lr", "learning_rate", "weight_decay", "lambda_contrastive", "lambda_must_link",
    "lambda_cannot_link", "lambda_entropy", "lambda_constraint",
    "lambda_constraint_pseudo", "temperature",
}


def load_config(path: str | Path) -> dict:
    """读取 YAML 并把学习率等浮点、batch 等整型字段转成 Python 类型。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in _FLOAT_KEYS:
        if k in cfg and cfg[k] is not None:
            cfg[k] = float(cfg[k])
    for k in (
        "batch_size", "decoder_hidden", "epochs", "seed", "constraint_sample_size",
        "episode_size", "set_size",
        "min_classes_per_episode", "max_classes_per_episode",
        "min_samples_per_class", "max_samples_per_class",
        "num_clusters", "max_clusters", "max_pairs_per_batch",
        "episode_size", "kmax", "max_steps", "grad_accumulation_steps",
        "train_episodes_per_epoch", "val_episodes_per_epoch", "lm_lora_rank",
        "clip_lora_rank", "max_new_tokens", "eval_episodes",
    ):
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(cfg[k])
    return cfg


def set_seed(seed: int) -> None:
    """固定 random / numpy / torch（含 CUDA）种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """有 CUDA 则 cuda，否则 cpu。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
