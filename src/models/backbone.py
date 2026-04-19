"""
Backbone factory for face recognition models.

- Returns a (backbone, embedding_dim) pair ready for head attachment.
- Pretrained weights use torchvision defaults (ImageNet).
- The final classification layer is ALWAYS replaced — callers attach
    their own head.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# Maps config name → (torchvision factory, penultimate feature dim)
_BACKBONE_REGISTRY: dict[str, Tuple[callable, int]] = {
    "resnet50": (models.resnet50, 2048),
    "resnet100": (models.resnet101, 2048),   # torchvision has resnet101≈resnet100
    "mobilenet_v2": (models.mobilenet_v2, 1280),
}


def build_backbone(
    name: str,
    pretrained: bool = True,
    dropout: float = 0.0,
    embedding_dim: int = 512,
) -> nn.Module:
    """
    Construct a backbone network with an L2-normalised embedding head.

    The returned module's forward() accepts (B, 3, H, W) float32 tensors
    and returns (B, embedding_dim) L2-normalised embeddings.

    Args:
        name:          One of 'resnet50', 'resnet100', 'mobilenet_v2'.
        pretrained:    Load ImageNet weights when True.
        dropout:       Dropout probability before the embedding FC layer.
        embedding_dim: Output embedding size.

    Returns:
        nn.Module with forward signature (B, 3, H, W) → (B, embedding_dim).

    Raises:
        ValueError: If `name` is not registered.
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone {name!r}. "
            f"Choose from: {sorted(_BACKBONE_REGISTRY)}"
        )

    factory, feat_dim = _BACKBONE_REGISTRY[name]
    weights = "IMAGENET1K_V1" if pretrained else None
    base = factory(weights=weights)

    # Strip the original classifier
    if name.startswith("resnet"):
        base.fc = nn.Identity()
    elif name == "mobilenet_v2":
        base.classifier = nn.Identity()

    head = _EmbeddingHead(feat_dim, embedding_dim, dropout)
    model = _BackboneWithHead(base, head, name)
    logger.info(
        "Built backbone %s | pretrained=%s | embedding_dim=%d",
        name, pretrained, embedding_dim,
    )
    return model


# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------


class _EmbeddingHead(nn.Module):
    """BN → Dropout → FC → BN (no activation) → L2-norm."""

    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn2 = nn.BatchNorm1d(out_features, affine=False)

        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in) → (B, out)
        x = self.bn1(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn2(x)
        # L2 normalise → unit hypersphere
        return nn.functional.normalize(x, p=2, dim=1)


class _BackboneWithHead(nn.Module):
    """Wraps backbone + head; exposes `feature_dim` for loss heads."""

    def __init__(self, backbone: nn.Module, head: _EmbeddingHead, name: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.backbone_name = name
        self.embedding_dim: int = head.fc.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def __repr__(self) -> str:
        return (
            f"_BackboneWithHead(name={self.backbone_name!r}, "
            f"embedding_dim={self.embedding_dim})"
        )