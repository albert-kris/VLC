"""SwAV prototype head.

A bank of K learnable prototype vectors (a bias-free linear layer). Given an
L2-normalized embedding z (from the encoder's ProjectionHead), it returns the
similarity scores z @ prototypes^T. Prototypes are L2-normalized each step so
the scores are cosine similarities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVHead(nn.Module):
    def __init__(self, dim: int = 256, n_prototypes: int = 10) -> None:
        super().__init__()
        self.prototypes = nn.Linear(dim, n_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1)
        self.prototypes.weight.data.copy_(w)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, dim) already L2-normalized -> scores (B, K)
        return self.prototypes(z)
