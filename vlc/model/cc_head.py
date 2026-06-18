"""Contrastive Clustering heads (Li et al., AAAI 2021).

Two INDEPENDENT projection heads on top of the shared backbone features:
  - instance_head : maps backbone hidden -> L2-normalized feature (instance contrast)
  - cluster_head  : maps backbone hidden -> softmax over K (cluster contrast)

The criterion text is fed into the backbone, so the same heads produce
criterion-specific representations without needing per-criterion parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCHeads(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        mid_dim: int = 512,
        feat_dim: int = 128,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.instance_head = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, feat_dim),
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, k),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (N, in_dim) -> (z, c).

        z : (N, feat_dim) L2-normalized instance features
        c : (N, k) softmax cluster assignments
        """
        z = F.normalize(self.instance_head(h), dim=-1)
        c = torch.softmax(self.cluster_head(h), dim=-1)
        return z, c

    @torch.no_grad()
    def hard_assignments(self, h: torch.Tensor) -> torch.Tensor:
        c = torch.softmax(self.cluster_head(h), dim=-1)
        return c.argmax(dim=-1)
