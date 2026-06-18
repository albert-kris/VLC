"""DEC cluster head: per-criterion learnable centroids + Student-t soft assignment.

Each criterion string maps to a set of K learnable cluster centroids in embedding
space. During DEC self-training, the centroids are jointly optimised with LoRA+proj.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DECHead(nn.Module):
    """Manages per-criterion centroid sets and computes soft cluster assignments.

    Parameters
    ----------
    embed_dim : dimensionality of the projected embeddings (proj_head output_dim)
    k         : number of clusters
    alpha     : degrees of freedom of the Student-t kernel (default 1 as in DEC paper)
    """

    def __init__(self, embed_dim: int = 256, k: int = 10, alpha: float = 1.0,
                 temperature: float = 1.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        self.alpha = alpha
        # Distance temperature: divides squared distances before the Student-t
        # kernel. Large temperature -> softer assignments (avoids one-hot
        # saturation that makes the DEC sharpening gradient vanish).
        self.temperature = temperature
        # centroids per criterion key: registered as parameters so they are
        # optimised by the outer optimizer
        self._centroids: nn.ParameterDict = nn.ParameterDict()

    def _key(self, criterion_key: str) -> str:
        # nn.ParameterDict keys must be valid Python identifiers
        return criterion_key.replace(" ", "_").replace("-", "_")

    def has_criterion(self, criterion_key: str) -> bool:
        return self._key(criterion_key) in self._centroids

    def init_centroids(self, criterion_key: str, embeddings: torch.Tensor) -> None:
        """Initialise centroids for a criterion via KMeans on provided embeddings.

        Parameters
        ----------
        criterion_key : the criterion identifier string
        embeddings    : (N, embed_dim) float32 tensor (detached, on any device)
        """
        from sklearn.cluster import KMeans

        key = self._key(criterion_key)
        np_emb = embeddings.detach().cpu().float().numpy()
        km = KMeans(n_clusters=self.k, random_state=42, n_init=10, max_iter=300)
        km.fit(np_emb)
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32,
                                 device=embeddings.device)  # (K, d)
        self._centroids[key] = nn.Parameter(centroids)

    def forward(self, embeddings: torch.Tensor, criterion_key: str) -> torch.Tensor:
        """Compute soft cluster assignment Q.

        Parameters
        ----------
        embeddings    : (N, embed_dim) L2-normalised embeddings
        criterion_key : key into self._centroids

        Returns
        -------
        q : (N, K) soft assignments (each row sums to 1)
        """
        key = self._key(criterion_key)
        if key not in self._centroids:
            raise KeyError(
                f"Criterion '{criterion_key}' not initialised. "
                "Call init_centroids first."
            )
        mu = self._centroids[key]          # (K, d)
        # Student-t kernel: q_ij ∝ (1 + ||z_i - mu_j||^2 / (alpha*T))^(-(alpha+1)/2)
        diff = embeddings.unsqueeze(1) - mu.unsqueeze(0)   # (N, K, d)
        dist2 = (diff ** 2).sum(dim=-1)                    # (N, K)
        num = (1.0 + dist2 / (self.alpha * self.temperature)) ** (-(self.alpha + 1.0) / 2.0)
        q = num / num.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return q                           # (N, K)

    def hard_assignments(self, embeddings: torch.Tensor, criterion_key: str) -> torch.Tensor:
        """Return argmax cluster assignment. Shape: (N,)."""
        with torch.no_grad():
            q = self.forward(embeddings, criterion_key)
        return q.argmax(dim=1)

    def centroid_params(self, criterion_key: str) -> nn.Parameter:
        return self._centroids[self._key(criterion_key)]

    # ------------------------------------------------------------------
    # CC soft-head (used in contrastive mode): a linear layer per criterion
    # ------------------------------------------------------------------

    _cc_heads: nn.ModuleDict

    def get_cc_head(self, criterion_key: str, embed_dim: int, k: int) -> nn.Linear:
        """Lazily create and return a softmax cluster head for CC training."""
        if not hasattr(self, "_cc_heads"):
            self._cc_heads = nn.ModuleDict()
            # Register so optimizer picks it up
            self.add_module("_cc_heads", self._cc_heads)
        key = self._key(criterion_key)
        if key not in self._cc_heads:
            head = nn.Linear(embed_dim, k, bias=False)
            nn.init.orthogonal_(head.weight)
            # Use device of DECHead itself (set when moved to device in trainer)
            device = next(self.parameters(), torch.tensor(0)).device if list(self.parameters()) else torch.device("cpu")
            head = head.to(device)
            self._cc_heads[key] = head
        return self._cc_heads[key]
