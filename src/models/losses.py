"""
Loss function implementations for face recognition.

Modules
-------
- SoftmaxLoss        — standard cross-entropy baseline (cosine classifier)
- ArcFaceLoss        — additive angular margin (CVPR 2019)
- SphereFaceLoss     — multiplicative angular margin (CVPR 2017)
- TripletLoss        — FaceNet-style with hard / semi-hard / random mining

"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_EPS = 1e-7        # general epsilon
_ACOS_EPS = 1e-4   # tighter clamp before acos to avoid ±inf gradients


# ---------------------------------------------------------------------------
# Softmax baseline
# ---------------------------------------------------------------------------


class SoftmaxLoss(nn.Module):
    """
    Cosine softmax cross-entropy baseline.

    Both embeddings and weight vectors are L2-normalised before the inner
    product, making this a cosine classifier — consistent with ArcFace /
    SphereFace and a fairer baseline.

    Args:
        embedding_dim:    Input feature dimension (must match backbone output).
        num_classes:      Number of identity classes.
        scale:            Feature-scale factor *s* (same role as in ArcFace).
        label_smoothing:  Label-smoothing epsilon (0 = disabled).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _compute_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Shared logit computation used by both forward paths."""
        return F.linear(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(self.weight,  p=2, dim=1),
        ) * self.scale  # (B, C)

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) — L2-normalised
        labels: torch.Tensor,      # (B,)   — integer class indices
    ) -> torch.Tensor:
        logits = self._compute_logits(embeddings)
        return self.criterion(logits, labels)

    def forward_with_logits(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, logits) — logits are pre-softmax scaled cosines."""
        logits = self._compute_logits(embeddings)
        return self.criterion(logits, labels), logits.detach()


# ---------------------------------------------------------------------------
# ArcFace
# ---------------------------------------------------------------------------


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (Deng et al., CVPR 2019).

    Adds a fixed angular margin *m* to the target-class angle before
    computing softmax, enforcing tighter intra-class compactness and
    larger inter-class angular separation.

    Args:
        embedding_dim: Input feature dimension.
        num_classes:   Number of identity classes.
        margin:        Angular margin in radians (default 0.5 ≈ 28.6°).
        scale:         Feature-scale factor *s* (default 64).
        easy_margin:   When True, clamps cos(θ+m) to cos(θ) for θ near 0
                        — avoids instability early in training.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.easy_margin = easy_margin

        # Pre-compute constants for cos(θ + m) = cos θ · cos m − sin θ · sin m
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Fallback boundary: cos(π − m) and sin(π − m) · m
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.criterion = nn.CrossEntropyLoss()

    def _compute_logits(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Shared logit computation used by both forward paths."""
        cosine = F.linear(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(self.weight,  p=2, dim=1),
        ).clamp(-1.0 + _EPS, 1.0 - _EPS)  # (B, C)

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=_EPS))
        phi  = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0.0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale  # (B, C)

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) — L2-normalised
        labels: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        logits = self._compute_logits(embeddings, labels)
        return self.criterion(logits, labels)

    def forward_with_logits(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, logits) — logits have the angular margin applied."""
        logits = self._compute_logits(embeddings, labels)
        return self.criterion(logits, labels), logits.detach()


# ---------------------------------------------------------------------------
# SphereFace
# ---------------------------------------------------------------------------


class SphereFaceLoss(nn.Module):
    """
    SphereFace: Large-Margin Softmax (A-Softmax) (Liu et al., CVPR 2017).

    Applies a multiplicative angular margin *m* via cos(m·θ), computed
    with Chebyshev recursion for numerical stability.

    Lambda-annealing blends the SphereFace logits with plain cosine softmax
    during early training to prevent gradient instability.

    Args:
        embedding_dim: Input feature dimension.
        num_classes:   Number of identity classes.
        margin:        Multiplicative angular margin m (default 4, must be a
                       positive integer).
        scale:         Feature-scale factor (default 64).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: int = 4,
        scale: float = 64.0,
    ) -> None:
        super().__init__()

        if not isinstance(margin, int) or margin < 1:
            raise ValueError(
                f"SphereFaceLoss margin must be a positive integer, got {margin!r}"
            )

        self.margin = margin
        self.scale  = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.criterion = nn.CrossEntropyLoss()

        self._iter: int          = 0
        self._base_lambda: float = 1000.0
        self._gamma: float       = 0.12
        self._power: float       = 1.0
        self._lambda_min: float  = 5.0

    def state_dict(self, *args, **kwargs) -> dict:
        sd = super().state_dict(*args, **kwargs)
        sd["_iter"] = self._iter
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        self._iter = int(state_dict.pop("_iter", 0))
        super().load_state_dict(state_dict, strict=strict)

    def _cos_m_theta(self, cos_theta: torch.Tensor) -> torch.Tensor:
        cos_mt   = cos_theta.clone()
        cos_prev = torch.ones_like(cos_theta)
        for _ in range(1, self.margin):
            cos_mt, cos_prev = (
                2.0 * cos_theta * cos_mt - cos_prev,
                cos_mt,
            )
        return cos_mt

    def _compute_logits(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Shared logit computation used by both forward paths."""
        w_norm = F.normalize(self.weight,  p=2, dim=1)
        e_norm = F.normalize(embeddings,   p=2, dim=1)
        cosine = F.linear(e_norm, w_norm).clamp(-1.0 + _EPS, 1.0 - _EPS)  # (B, C)

        B = embeddings.size(0)
        target_cosine = cosine[torch.arange(B, device=cosine.device), labels]
        target_cos_mt = self._cos_m_theta(target_cosine)

        target_cosine_safe = target_cosine.clamp(-1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS)
        k   = (self.margin * target_cosine_safe.acos() / math.pi).floor().detach()
        phi = ((-1.0) ** k) * target_cos_mt - 2.0 * k
        phi = phi.to(cosine.dtype)
        lam = max(
            self._lambda_min,
            self._base_lambda
            * (1.0 + self._gamma * self._iter) ** (-self._power),
        )
        self._iter += 1

        logits = cosine.clone() * self.scale
        logits[torch.arange(B, device=cosine.device), labels] = (
            (lam * target_cosine + phi) / (1.0 + lam)
        ) * self.scale

        return logits  # (B, C)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._compute_logits(embeddings, labels)
        return self.criterion(logits, labels)

    def forward_with_logits(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, logits) — logits have lambda-annealed margin applied."""
        logits = self._compute_logits(embeddings, labels)
        return self.criterion(logits, labels), logits.detach()


# ---------------------------------------------------------------------------
# Triplet Loss
# ---------------------------------------------------------------------------

MiningStrategy = Literal["hard", "semi-hard", "random"]


class TripletLoss(nn.Module):
    """
    Online triplet loss with three mining strategies.

    All mining operates on a pairwise cosine-distance matrix built from
    the batch, so no explicit triplet sampling is needed in the data loader.

    Mining strategies
    -----------------
    hard:      Hardest positive (max d_ap) + hardest negative (min d_an).
    semi-hard: For each anchor, negatives that are farther than the positive
               but still within margin; falls back to hard when none exist.
    random:    Random valid positive + random valid negative per anchor;
               falls back to hard when an anchor has no valid negative.

    Args:
        margin:     Margin α (default 0.3).
        mining:     Mining strategy (default 'semi-hard').
        batch_hard: Unused — kept for config compatibility.  Passing
                    batch_hard=False logs a warning because it has no effect.
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining: MiningStrategy = "semi-hard",
        batch_hard: bool = True,
    ) -> None:
        super().__init__()
        self.margin     = margin
        self.mining     = mining
        self.batch_hard = batch_hard

        if not batch_hard:
            logger.warning(
                "TripletLoss: batch_hard=False is not implemented and has no "
                "effect.  Hard mining is always used within the chosen strategy."
            )

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) — L2-normalised
        labels: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        dist_mat = self._pairwise_cosine_dist(embeddings)  # (B, B)

        if self.mining == "hard":
            return self._hard_loss(dist_mat, labels)
        elif self.mining == "semi-hard":
            return self._semi_hard_loss(dist_mat, labels)
        else:
            return self._random_loss(dist_mat, labels)

    def forward_with_nn_acc(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Returns (loss, nn_accuracy).

        nn_accuracy: fraction of samples whose nearest neighbour (excluding
        self) shares the same identity label.  Computed from the same
        distance matrix used for mining — zero extra cost.
        """
        dist_mat = self._pairwise_cosine_dist(embeddings)  # (B, B)

        # Loss (reuse existing mining logic via the shared dist_mat)
        if self.mining == "hard":
            loss = self._hard_loss(dist_mat, labels)
        elif self.mining == "semi-hard":
            loss = self._semi_hard_loss(dist_mat, labels)
        else:
            loss = self._random_loss(dist_mat, labels)

        # NN accuracy — mask the diagonal so a sample can't be its own neighbour
        B   = dist_mat.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=dist_mat.device)
        dist_no_self       = dist_mat.masked_fill(eye, float("inf"))
        nn_labels          = labels[dist_no_self.argmin(dim=1)]   # (B,)
        nn_acc             = (nn_labels == labels).float().mean().item()

        return loss, nn_acc

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_cosine_dist(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Pairwise cosine distance: D[i,j] = 1 − cos_sim(e_i, e_j).

        Embeddings are re-normalised here as a safety measure.
        """
        e = F.normalize(embeddings, p=2, dim=1)
        sim = torch.mm(e, e.t()).clamp(-1.0 + _EPS, 1.0 - _EPS)
        return 1.0 - sim  # (B, B), in [0, 2]

    # ------------------------------------------------------------------
    # Positive / negative masks
    # ------------------------------------------------------------------

    @staticmethod
    def _masks(
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build boolean positive and negative masks, excluding the diagonal.

        Returns:
            pos_mask: (B, B) — True for same-class, off-diagonal pairs.
            neg_mask: (B, B) — True for different-class pairs.
        """
        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        eye  = torch.eye(
            labels.size(0), dtype=torch.bool, device=labels.device
        )
        return same & ~eye, ~same

    # ------------------------------------------------------------------
    # Mining strategies
    # ------------------------------------------------------------------

    def _hard_loss(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        pos_mask, neg_mask = self._masks(labels)
        d_ap = (dist_mat * pos_mask.float()).max(dim=1).values
        d_an = (dist_mat + (~neg_mask).float() * 1e6).min(dim=1).values
        return F.relu(d_ap - d_an + self.margin).mean()

    def _semi_hard_loss(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        pos_mask, neg_mask = self._masks(labels)
        d_ap = (dist_mat * pos_mask.float()).max(dim=1).values

        semi_mask = (
            neg_mask
            & (dist_mat > d_ap.unsqueeze(1))
            & (dist_mat < d_ap.unsqueeze(1) + self.margin)
        )

        if not semi_mask.any():
            logger.debug("TripletLoss: no semi-hard negatives found — using hard mining")
            return self._hard_loss(dist_mat, labels)

        valid = semi_mask.any(dim=1)
        d_an  = (dist_mat + (~semi_mask).float() * 1e6).min(dim=1).values
        loss  = F.relu(d_ap - d_an + self.margin)
        return loss[valid].mean()

    def _random_loss(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        pos_mask, neg_mask = self._masks(labels)

        has_neg = neg_mask.any(dim=1)
        if not has_neg.any():
            logger.debug("TripletLoss: no valid negatives in batch — using hard mining")
            return self._hard_loss(dist_mat, labels)

        def _rand_idx(mask: torch.Tensor) -> torch.Tensor:
            noise = torch.rand_like(dist_mat) * mask.float()
            return noise.argmax(dim=1)

        p_idx = _rand_idx(pos_mask)
        n_idx = _rand_idx(neg_mask)

        B   = dist_mat.size(0)
        idx = torch.arange(B, device=dist_mat.device)
        d_ap = dist_mat[idx, p_idx]
        d_an = dist_mat[idx, n_idx]

        loss = F.relu(d_ap - d_an + self.margin)
        return loss[has_neg].mean()