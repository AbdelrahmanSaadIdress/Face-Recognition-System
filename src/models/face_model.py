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

    # Training step with accuracy signal
    loss, logits = model(images, labels, return_logits=True)   # classification losses
    loss, nn_acc = model(images, labels, return_logits=True)   # triplet loss

    # Inference / evaluation
    model.eval()
    with torch.no_grad():
        embeddings = model(images)          # (B, 512) — no labels
"""

from __future__ import annotations

import logging
from typing import Optional, Union

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

# Loss heads that expose classification logits
_CLASSIFICATION_LOSSES = (SoftmaxLoss, ArcFaceLoss, SphereFaceLoss)


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
        images: torch.Tensor,                    # (B, 3, H, W) float32
        labels: Optional[torch.Tensor] = None,   # (B,) int64 — optional
        return_logits: bool = False,             # request logits / nn_acc
    ) -> Union[
        torch.Tensor,                            # inference: embeddings
        torch.Tensor,                            # training, return_logits=False: loss
        tuple[torch.Tensor, torch.Tensor],       # classification + return_logits=True: (loss, logits)
        tuple[torch.Tensor, float],              # triplet + return_logits=True: (loss, nn_acc)
    ]:
        """
        Dual-mode forward pass.

        Training mode (labels provided):
            return_logits=False → loss scalar
            return_logits=True  → (loss, logits)  for classification losses
                                   (loss, nn_acc) for TripletLoss
                                   nn_acc is a plain Python float in [0, 1]

        Inference mode (labels=None) → (B, embedding_dim) embeddings.
        return_logits is ignored in inference mode.

        Args:
            images:        Batch of preprocessed face crops.
            labels:        Ground-truth identity indices.  Required for training.
            return_logits: When True in training mode, also return the logits
                           (classification heads) or NN accuracy (triplet).

        Returns:
            See type signature above.

        Raises:
            RuntimeError: If called in training mode without a loss head.
        """
        embeddings = self.backbone(images)  # (B, D) L2-normalised

        if labels is None:
            # Inference path — embeddings only, ignore return_logits
            return embeddings

        # Training path
        if self.loss_head is None:
            raise RuntimeError(
                "FaceModel.forward called with labels but loss_head is None. "
                "Attach a loss head or call without labels for inference."
            )

        if return_logits:
            if isinstance(self.loss_head, _CLASSIFICATION_LOSSES):
                # ArcFace / SphereFace / Softmax → (loss, logits)
                return self.loss_head.forward_with_logits(embeddings, labels)
            elif isinstance(self.loss_head, TripletLoss):
                # Triplet → (loss, nn_accuracy float)
                return self.loss_head.forward_with_nn_acc(embeddings, labels)

        return self.loss_head(embeddings, labels)

    # ------------------------------------------------------------------
    # Convenience: extract embeddings for a batch without gradients
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalised embeddings (inference only, no grad).

        Args:
            images: (B, 3, H, W) float32 preprocessed crops.

        Returns:
            (B, embedding_dim) float32 L2-normalised embeddings.
        """
        training_state = self.training
        self.eval()
        embeddings = self.backbone(images)
        self.train(training_state)
        return embeddings

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def backbone_state_dict(self) -> dict:
        """Return backbone-only state dict for FAISS / deployment."""
        return self.backbone.state_dict()

    def load_backbone_weights(self, state_dict: dict, strict: bool = True) -> None:
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


def build_face_model(cfg: Config, num_classes: int=None) -> FaceModel:
    """
    Construct a FaceModel from a loaded Config object.

    Args:
        cfg:         Fully loaded Config (from load_config()).
        num_classes: Number of identity classes in the training set.
                     Ignored for TripletLoss (no classification head).
                     Pass None to build an embedding-only model for FAISS.

    Returns:
        FaceModel ready for training.
    """
    if cfg.model.name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown model/loss name {cfg.model.name!r}. "
            f"Valid options: {sorted(_LOSS_REGISTRY)}"
        )

    backbone = build_backbone(
        name          = cfg.model.backbone,
        pretrained    = cfg.model.pretrained_backbone,
        dropout       = cfg.model.dropout,
        embedding_dim = cfg.model.embedding_dim,
    )

    loss_head = None if num_classes is None else _build_loss_head(cfg, num_classes)

    return FaceModel(
        backbone   = backbone,
        loss_head  = loss_head,
        model_name = f"{cfg.model.name}_{cfg.model.backbone}",
    )


def _build_loss_head(cfg: Config, num_classes: int) -> nn.Module:
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

    raise ValueError(f"Unhandled loss type: {name!r}")