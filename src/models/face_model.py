"""
src/models/face_model.py
========================
Unified FaceModel wrapper — backbone + loss head in one module.

Design contract
---------------
- Training mode  (labels provided)  → returns scalar loss tensor.
- Inference mode (labels=None)      → returns (B, embedding_dim) float32
                                       L2-normalised embeddings ONLY.
- The loss head is NEVER called at inference time, so it never leaks into
  FAISS, evaluation, or the attendance system.
- All four loss heads (softmax, arcface, sphereface, triplet) share the
  same interface through this wrapper — the training loop never needs to
  know which loss is active.

Factory
-------
Use `build_face_model(cfg, num_classes)` rather than constructing directly;
it reads ModelConfig + LossConfig from the loaded YAML config.

Example
-------
    cfg   = load_config()
    model = build_face_model(cfg, num_classes=10_572)
    model.train()

    # Training step
    loss = model(images, labels)
    loss.backward()

    # Inference / evaluation
    model.eval()
    with torch.no_grad():
        embeddings = model(images)          # (B, 512) — no labels
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.config import Config
from src.models.backbone import build_backbone
from src.models.losses import (
    ArcFaceLoss,
    SoftmaxLoss,
    SphereFaceLoss,
    TripletLoss,
)

logger = logging.getLogger(__name__)

# Maps config string → loss class
_LOSS_REGISTRY: dict[str, type] = {
    "softmax":     SoftmaxLoss,
    "arcface":     ArcFaceLoss,
    "sphereface":  SphereFaceLoss,
    "triplet":     TripletLoss,
}


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class FaceModel(nn.Module):
    """
    Complete face recognition model: backbone + loss head.

    The loss head is attached as a submodule so its parameters are included
    in `model.parameters()` and saved/loaded with the checkpoint.

    During inference the head is bypassed entirely — only the backbone
    runs, returning clean L2-normalised embeddings.

    Args:
        backbone:   Backbone network (output of `build_backbone`).
                    Must expose `.embedding_dim: int`.
        loss_head:  One of SoftmaxLoss / ArcFaceLoss / SphereFaceLoss /
                    TripletLoss.  Set to None for embedding-only models
                    (e.g. after loading a pretrained checkpoint for FAISS).
        model_name: Human-readable name used in logging / checkpoints.
    """

    def __init__(
        self,
        backbone: nn.Module,
        loss_head: Optional[nn.Module],
        model_name: str = "face_model",
    ) -> None:
        super().__init__()
        self.backbone   = backbone
        self.loss_head  = loss_head
        self.model_name = model_name

        # Expose embedding dimension for external callers (FAISS builder, etc.)
        self.embedding_dim: int = backbone.embedding_dim

        logger.info(
            "FaceModel ready — backbone=%s  loss=%s  embedding_dim=%d",
            getattr(backbone, "backbone_name", "?"),
            type(loss_head).__name__ if loss_head else "None",
            self.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,            # (B, 3, H, W) float32
        labels: Optional[torch.Tensor] = None,  # (B,) int64 — optional
    ) -> torch.Tensor:
        """
        Dual-mode forward pass.

        Training mode  → loss (scalar tensor).
        Inference mode → embeddings (B, embedding_dim).

        Args:
            images: Batch of preprocessed face crops.
            labels: Ground-truth identity indices.
                    Must be provided during training; omit for inference.

        Returns:
            Scalar loss tensor  (when labels is not None).
            (B, D) embedding tensor  (when labels is None).

        Raises:
            RuntimeError: If called in training mode without a loss head.
        """
        embeddings = self.backbone(images)          # (B, D) L2-normalised

        if labels is None:
            # Inference path — return embeddings directly
            return embeddings

        # Training path — compute loss
        if self.loss_head is None:
            raise RuntimeError(
                "FaceModel.forward called with labels but loss_head is None. "
                "Attach a loss head or call without labels for inference."
            )
        return self.loss_head(embeddings, labels)

    # ------------------------------------------------------------------
    # Convenience: extract embeddings for a batch without gradients
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalised embeddings (inference only, no grad).

        Equivalent to:
            model.eval()
            with torch.no_grad():
                emb = model(images)

        But more explicit and safe — always bypasses the loss head.

        Args:
            images: (B, 3, H, W) float32 preprocessed crops.

        Returns:
            (B, embedding_dim) float32 L2-normalised embeddings.
        """
        training_state = self.training
        self.eval()
        embeddings = self.backbone(images)
        self.train(training_state)   # restore original state
        return embeddings

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def backbone_state_dict(self) -> dict:
        """
        Return backbone-only state dict.

        Use this when saving a checkpoint for FAISS / deployment —
        the loss head weights (class centres) are not needed at inference.
        """
        return self.backbone.state_dict()

    def load_backbone_weights(self, state_dict: dict, strict: bool = True) -> None:
        """
        Load backbone weights from a state dict.

        Args:
            state_dict: Output of a previous `backbone_state_dict()` call.
            strict:     Passed to `load_state_dict` — set False when loading
                        a checkpoint with a different head size.
        """
        missing, unexpected = self.backbone.load_state_dict(
            state_dict, strict=strict
        )
        if missing:
            logger.warning("backbone load: missing keys — %s", missing)
        if unexpected:
            logger.warning("backbone load: unexpected keys — %s", unexpected)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FaceModel("
            f"name={self.model_name!r}, "
            f"embedding_dim={self.embedding_dim}, "
            f"loss={type(self.loss_head).__name__ if self.loss_head else 'None'})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_face_model(cfg: Config, num_classes: int) -> FaceModel:
    """
    Construct a FaceModel from a loaded Config object.

    Reads:
        cfg.model  → backbone name, embedding_dim, pretrained, dropout
        cfg.model.name  → which loss head to build
        cfg.loss.*      → loss-specific hyperparameters

    Args:
        cfg:         Fully loaded Config (from load_config()).
        num_classes: Number of identity classes in the training set.
                     Ignored for TripletLoss (no classification head).

    Returns:
        FaceModel ready for training.

    Raises:
        ValueError: If cfg.model.name is not a registered loss type.

    Example:
        cfg   = load_config()
        model = build_face_model(cfg, num_classes=10_572)
    """
    if cfg.model.name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown model/loss name {cfg.model.name!r}. "
            f"Valid options: {sorted(_LOSS_REGISTRY)}"
        )

    # 1. Backbone
    backbone = build_backbone(
        name          = cfg.model.backbone,
        pretrained    = cfg.model.pretrained_backbone,
        dropout       = cfg.model.dropout,
        embedding_dim = cfg.model.embedding_dim,
    )

    # if the num_classes was none then it's for evaluation or faiss building, in which case we don't need to build the loss head
    if num_classes is None:
        loss_head = None
    else:
        # 2. Loss head
        loss_head = _build_loss_head(cfg, num_classes)

    return FaceModel(
        backbone   = backbone,
        loss_head  = loss_head,
        model_name = f"{cfg.model.name}_{cfg.model.backbone}",
    )


def _build_loss_head(cfg: Config, num_classes: int) -> nn.Module:
    """
    Instantiate the correct loss head from config.

    Args:
        cfg:         Full config.
        num_classes: Number of training identities.

    Returns:
        Configured loss head nn.Module.
    """
    name = cfg.model.name
    dim  = cfg.model.embedding_dim

    if name == "softmax":
        return SoftmaxLoss(
            embedding_dim   = dim,
            num_classes     = num_classes,
            scale           = 64.0,
            label_smoothing = cfg.loss.softmax.label_smoothing,
        )

    if name == "arcface":
        return ArcFaceLoss(
            embedding_dim = dim,
            num_classes   = num_classes,
            margin        = cfg.loss.arcface.margin,
            scale         = cfg.loss.arcface.scale,
            easy_margin   = cfg.loss.arcface.easy_margin,
        )

    if name == "sphereface":
        return SphereFaceLoss(
            embedding_dim = dim,
            num_classes   = num_classes,
            margin        = cfg.loss.sphereface.margin,
            scale         = cfg.loss.sphereface.scale,
        )

    if name == "triplet":
        return TripletLoss(
            margin     = cfg.loss.triplet.margin,
            mining     = cfg.loss.triplet.mining,
            batch_hard = cfg.loss.triplet.batch_hard,
        )

    # Should never reach here due to registry check in build_face_model
    raise ValueError(f"Unhandled loss type: {name!r}")