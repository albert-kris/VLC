from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# DEC: Deep Embedded Clustering (Xie et al. 2016) – unsupervised
# ---------------------------------------------------------------------------

def dec_target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Compute sharpened target distribution P from soft assignment Q.

    P_ij = (Q_ij^2 / f_j) / sum_j' (Q_ij'^2 / f_j')
    where f_j = sum_i Q_ij  (cluster frequencies).

    Parameters
    ----------
    q : (N, K) soft assignments (each row sums to 1)
    Returns
    -------
    p : (N, K) target distribution
    """
    f = q.sum(dim=0, keepdim=True).clamp(min=1e-8)   # (1, K)
    p = (q ** 2) / f
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return p


def dec_kl_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) = sum_ij P_ij * log(P_ij / Q_ij).

    Parameters
    ----------
    q : (N, K) soft assignments
    p : (N, K) target distribution (detached from graph)
    """
    p = p.detach()
    q = q.clamp(min=1e-8)
    p_safe = p.clamp(min=1e-8)
    return (p_safe * (p_safe.log() - q.log())).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# CC: Contrastive Clustering (Li et al. 2021) – unsupervised
# ---------------------------------------------------------------------------

def instance_contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """InfoNCE between two augmented views (instance level).

    z_a, z_b : (N, d) L2-normalized embeddings
    Positives: (z_a[i], z_b[i]); negatives: all other pairs.
    """
    n = z_a.size(0)
    device = z_a.device
    z = torch.cat([z_a, z_b], dim=0)            # (2N, d)
    sim = (z @ z.t()) / temperature              # (2N, 2N)
    # mask out self-similarity
    mask_diag = torch.eye(2 * n, device=device).bool()
    sim = sim.masked_fill(mask_diag, float("-inf"))
    # positive indices: i -> i+n  and  i+n -> i
    pos_idx = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(device)
    loss = F.cross_entropy(sim, pos_idx)
    return loss


def cluster_contrastive_loss(
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    temperature: float = 1.0,
    entropy_weight: float = 5.0,
) -> torch.Tensor:
    """Cluster-level contrastive loss with entropy regularization.

    c_a, c_b : (N, K) soft cluster assignments (after softmax), two views.
    The K cluster prototypes act as the batch dimension (transpose of instance NCE).
    Entropy regularization prevents cluster collapse.
    """
    # Normalize along N (treat each cluster as an embedding over samples)
    c_a_t = F.normalize(c_a.t(), dim=1)  # (K, N)
    c_b_t = F.normalize(c_b.t(), dim=1)  # (K, N)

    k = c_a_t.size(0)
    device = c_a.device
    z = torch.cat([c_a_t, c_b_t], dim=0)         # (2K, N)
    sim = (z @ z.t()) / temperature               # (2K, 2K)
    mask_diag = torch.eye(2 * k, device=device).bool()
    sim = sim.masked_fill(mask_diag, float("-inf"))
    pos_idx = torch.cat([torch.arange(k, 2 * k), torch.arange(k)]).to(device)
    cc_loss = F.cross_entropy(sim, pos_idx)

    # Entropy regularization: maximize entropy of mean assignment to prevent collapse
    mean_assign = (c_a + c_b).mean(dim=0) / 2.0  # (K,)
    mean_assign = mean_assign.clamp(min=1e-8)
    entropy = -(mean_assign * mean_assign.log()).sum()
    # maximize entropy -> subtract (or add negative)
    return cc_loss - entropy_weight * entropy


# ---------------------------------------------------------------------------
# SwAV: Swapping Assignments between Views (Caron et al. 2020) – unsupervised
# ---------------------------------------------------------------------------

@torch.no_grad()
def sinkhorn(scores: torch.Tensor, epsilon: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp: turn prototype scores into balanced soft assignments (codes).

    Enforces that, across the batch, samples are split roughly equally over the
    K prototypes (the equipartition constraint that prevents collapse).

    Parameters
    ----------
    scores : (B, K) raw similarity to prototypes (z @ prototypes^T)
    Returns
    -------
    Q : (B, K) doubly-normalized codes (each row sums to 1)
    """
    Q = torch.exp(scores / epsilon).t()        # (K, B)
    Q = Q / Q.sum().clamp(min=1e-8)
    K, B = Q.shape
    for _ in range(n_iters):
        # normalize rows (each prototype gets equal mass)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
        Q = Q / K
        # normalize columns (each sample's code sums to 1)
        Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
        Q = Q / B
    Q = Q * B                                   # so columns sum to 1
    return Q.t()                                # (B, K)


def swav_loss(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    temperature: float = 0.1,
    epsilon: float = 0.05,
    n_iters: int = 3,
) -> torch.Tensor:
    """Swapped-prediction loss between two views.

    Codes (targets) are computed by Sinkhorn on each view's scores, then each
    view's softmax prediction is trained to match the OTHER view's code.

    scores_a, scores_b : (B, K) = z @ prototypes^T for each view.
    """
    q_a = sinkhorn(scores_a, epsilon, n_iters)   # (B, K) target for view b
    q_b = sinkhorn(scores_b, epsilon, n_iters)   # (B, K) target for view a
    log_p_a = F.log_softmax(scores_a / temperature, dim=1)
    log_p_b = F.log_softmax(scores_b / temperature, dim=1)
    loss = -0.5 * ((q_b * log_p_a).sum(dim=1).mean()
                   + (q_a * log_p_b).sum(dim=1).mean())
    return loss


def cluster_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unsupervised clustering accuracy via Hungarian assignment."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    D = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return float(sum(w[r, c] for r, c in zip(row, col))) / y_pred.size


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Parameters
    ----------
    z      : (N, d) L2-normalized feature vectors
    labels : (N,) integer class labels (0-indexed)
    temperature : scalar

    For each anchor i, positives are all j != i with labels[j] == labels[i].
    Loss averaged over all anchors that have at least one positive.
    """
    n = z.size(0)
    device = z.device

    sim = z @ z.t() / temperature            # (N, N)

    labels = labels.view(-1, 1)              # (N, 1)
    mask_pos = (labels == labels.t()).float()
    mask_pos.fill_diagonal_(0.0)

    mask_diag = torch.eye(n, device=device).bool()
    sim_no_diag = sim.masked_fill(mask_diag, float("-inf"))
    log_denom = torch.logsumexp(sim_no_diag, dim=1, keepdim=True)  # (N, 1)

    log_prob = sim - log_denom               # (N, N)

    n_pos = mask_pos.sum(dim=1)              # (N,)
    has_pos = n_pos > 0

    loss_per_anchor = -(mask_pos * log_prob).sum(dim=1) / n_pos.clamp(min=1)
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss_per_anchor[has_pos].mean()


def hungarian_cluster_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        k = logits.size(1)
        cost = torch.zeros(k, k, device=logits.device)
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                continue
            for k2 in range(k):
                cost[c, k2] = (pred[mask] != k2).float().mean()
        row, col = linear_sum_assignment(cost.cpu().numpy())
        perm = torch.zeros(k, dtype=torch.long, device=logits.device)
        for r, c in zip(row, col):
            perm[c] = r
    aligned_labels = perm[labels]
    return F.cross_entropy(logits, aligned_labels)


def entropy_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1).mean(dim=0)
    p = p.clamp(min=1e-8)
    return (p * p.log()).sum()
