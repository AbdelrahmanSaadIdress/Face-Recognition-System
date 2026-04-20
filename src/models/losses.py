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

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) — L2-normalised
        labels: torch.Tensor,      # (B,)   — integer class indices
    ) -> torch.Tensor:
        # Both sides normalised → inner product = cosine similarity
        logits = F.linear(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(self.weight,  p=2, dim=1),
        ) * self.scale                              # (B, C)
        return self.criterion(logits, labels)


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

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) — L2-normalised
        labels: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        # Cosine similarity between embeddings and class centres
        cosine = F.linear(
            F.normalize(embeddings, p=2, dim=1),
            F.normalize(self.weight,  p=2, dim=1),
        ).clamp(-1.0 + _EPS, 1.0 - _EPS)           # (B, C)

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=_EPS))

        # cos(θ + m) via angle-addition formula
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Only apply margin when θ < π/2  (cosine > 0)
            phi = torch.where(cosine > 0.0, phi, cosine)
        else:
            # Standard fallback for θ + m > π
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Replace target-class cosine with phi; leave others unchanged
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return self.criterion(logits, labels)


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

        # FIX-4: validate margin type and range at construction time.
        # Passing a float (e.g. 2.5) produced silently wrong Chebyshev
        # results in the original code.
        if not isinstance(margin, int) or margin < 1:
            raise ValueError(
                f"SphereFaceLoss margin must be a positive integer, got {margin!r}"
            )

        self.margin = margin
        self.scale  = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.criterion = nn.CrossEntropyLoss()

        # λ-annealing state — decays with each forward pass
        self._iter: int         = 0
        self._base_lambda: float = 1000.0
        self._gamma: float       = 0.12
        self._power: float       = 1.0
        self._lambda_min: float  = 5.0

    # ------------------------------------------------------------------
    # Checkpoint support for lambda-annealing state  (FIX-3)
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs) -> dict:
        """
        Extend the standard state_dict with _iter so that lambda-annealing
        resumes from the correct position after a checkpoint restore.

        Without this, _iter resets to 0 on resume, lambda jumps back to
        ~1000, and the loss temporarily spikes as the model re-learns to
        rely on SphereFace rather than plain softmax.
        """
        sd = super().state_dict(*args, **kwargs)
        sd["_iter"] = self._iter
        return sd

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Restore _iter alongside the learnable parameters."""
        # Pop _iter before passing to the parent so it does not complain
        # about an unexpected key when strict=True.
        self._iter = int(state_dict.pop("_iter", 0))
        super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Chebyshev recursion: cos(m·θ) without explicit acos / cos
    # ------------------------------------------------------------------

    def _cos_m_theta(self, cos_theta: torch.Tensor) -> torch.Tensor:
        """
        Compute cos(m·θ) via Chebyshev recursion T_m(x) = 2x·T_{m-1}(x) − T_{m-2}(x).

        More numerically stable than computing acos → multiply → cos.

        Args:
            cos_theta: Tensor of cosine values, any shape.

        Returns:
            Tensor of cos(m·θ) values, same shape.
        """
        cos_mt   = cos_theta.clone()
        cos_prev = torch.ones_like(cos_theta)
        for _ in range(1, self.margin):
            cos_mt, cos_prev = (
                2.0 * cos_theta * cos_mt - cos_prev,
                cos_mt,
            )
        return cos_mt

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) — L2-normalised
        labels: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        w_norm = F.normalize(self.weight,   p=2, dim=1)
        e_norm = F.normalize(embeddings,    p=2, dim=1)
        cosine = F.linear(e_norm, w_norm).clamp(-1.0 + _EPS, 1.0 - _EPS)  # (B, C)

        # FIX-2: Compute the piecewise k-correction only for the target-class
        # angle (shape B) rather than the full (B, C) cosine matrix.
        #
        # The piecewise formula  phi = (-1)^k * cos(m*θ) - 2k  is derived
        # to make cos(m*θ) monotonically decreasing on [0, π].  It is only
        # mathematically meaningful for the ground-truth class logit; applying
        # it to all classes distorted non-target logits and could silently
        # produce NaN in non-target slots when cosine_safe hit an edge value.
        B = embeddings.size(0)
        target_cosine = cosine[torch.arange(B, device=cosine.device), labels]  # (B,)

        target_cos_mt = self._cos_m_theta(target_cosine)

        target_cosine_safe = target_cosine.clamp(
            -1.0 + _ACOS_EPS, 1.0 - _ACOS_EPS
        )
        k   = (self.margin * target_cosine_safe.acos() / math.pi).floor().detach()
        phi = ((-1.0) ** k) * target_cos_mt - 2.0 * k          # (B,)

        # λ-annealing: gradually shift from plain softmax → SphereFace
        lam = max(
            self._lambda_min,
            self._base_lambda
            * (1.0 + self._gamma * self._iter) ** (-self._power),
        )
        self._iter += 1

        # Blend target logit: [lam * cos(θ) + phi] / (1 + lam)
        # Non-target logits remain plain cosine, scaled.
        logits = cosine.clone() * self.scale
        logits[torch.arange(B, device=cosine.device), labels] = (
            (lam * target_cosine + phi) / (1.0 + lam)
        ) * self.scale

        return self.criterion(logits, labels)


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

        # FIX-5: warn explicitly when batch_hard=False is passed so callers
        # are not silently surprised that the flag has no effect.
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
        return 1.0 - sim                           # (B, B), in [0, 2]

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
        same = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
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
        """
        Batch-hard mining: hardest positive + hardest negative per anchor.

        d_ap = max distance among same-class pairs  (hardest positive).
        d_an = min distance among diff-class pairs  (hardest negative).
        """
        pos_mask, neg_mask = self._masks(labels)

        # Hardest positive: largest intra-class distance
        d_ap = (dist_mat * pos_mask.float()).max(dim=1).values  # (B,)

        # Hardest negative: smallest inter-class distance
        # Mask out non-negatives by adding a large constant
        d_an = (
            dist_mat + (~neg_mask).float() * 1e6
        ).min(dim=1).values                                      # (B,)

        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()

    def _semi_hard_loss(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Semi-hard mining (FaceNet):

        For each anchor, pick negatives where:
            d(a,p) < d(a,n) < d(a,p) + margin

        Falls back to hard mining when no semi-hard negatives exist in
        the current batch.
        """
        pos_mask, neg_mask = self._masks(labels)

        # Hardest positive distance per anchor
        d_ap = (dist_mat * pos_mask.float()).max(dim=1).values  # (B,)

        # Semi-hard negative region
        semi_mask = (
            neg_mask
            & (dist_mat > d_ap.unsqueeze(1))
            & (dist_mat < d_ap.unsqueeze(1) + self.margin)
        )

        # Fall back to hard mining if no semi-hard negatives found
        if not semi_mask.any():
            logger.debug("TripletLoss: no semi-hard negatives found — using hard mining")
            return self._hard_loss(dist_mat, labels)

        valid = semi_mask.any(dim=1)   # anchors that have a semi-hard neg

        d_an = (
            dist_mat + (~semi_mask).float() * 1e6
        ).min(dim=1).values                                      # (B,)

        loss = F.relu(d_ap - d_an + self.margin)
        # Average only over anchors that had a semi-hard negative — anchors
        # without one contributed nothing to d_an under the semi-hard logic.
        return loss[valid].mean()

    def _random_loss(
        self,
        dist_mat: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Random mining: one random valid positive + one random valid negative
        per anchor.

        Falls back to hard mining for any anchor that has no valid negative
        (e.g. entire batch is one identity — pathological but defensive).
        """
        pos_mask, neg_mask = self._masks(labels)

        # Detect anchors with no valid negative and fall back
        has_neg = neg_mask.any(dim=1)   # (B,) bool
        if not has_neg.any():
            logger.debug("TripletLoss: no valid negatives in batch — using hard mining")
            return self._hard_loss(dist_mat, labels)

        def _rand_idx(mask: torch.Tensor) -> torch.Tensor:
            """Sample one random True index per row; rows with no True get 0."""
            noise = torch.rand_like(dist_mat) * mask.float()
            return noise.argmax(dim=1)

        p_idx = _rand_idx(pos_mask)
        n_idx = _rand_idx(neg_mask)

        B   = dist_mat.size(0)
        idx = torch.arange(B, device=dist_mat.device)
        d_ap = dist_mat[idx, p_idx]
        d_an = dist_mat[idx, n_idx]

        loss = F.relu(d_ap - d_an + self.margin)
        # Only average over anchors that actually had a valid negative
        return loss[has_neg].mean()