"""
src/evaluation/evaluator.py
============================
LFW face-verification evaluator.

Usage
-----
    evaluator = LFWEvaluator(lfw_dataset, cfg, tracker)
    result    = evaluator.evaluate(model, model_name="arcface_resnet50")

    print(result.eer)        # 0.0421
    print(result.auc)        # 0.9947
    print(result.roc.fpr)    # array for plotting
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.data.lfw_dataset import LFWPairsDataset
from .metrics import (
    EvaluationSummary,
    ROCCurve,
    compute_auc,
    compute_roc,
    evaluate_model,
)
from src.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class EvaluationResult:
    """
    Complete evaluation output for one model on the LFW test set.

    Attributes
    ----------
    model_name   : str            — identifier of the evaluated model.
    summary      : EvaluationSummary — FAR, FRR, EER, AUC, F1 at operating threshold.
    roc          : ROCCurve       — full ROC curve arrays (fpr / tpr / thresholds).
    scores       : (N,) float32   — cosine-similarity score per pair.
    labels       : (N,) int32     — ground-truth binary labels (1=same, 0=different).
    embed_time_s : float          — seconds spent extracting embeddings.
    num_pairs    : int            — total evaluated pairs.
    """

    def __init__(
        self,
        model_name: str,
        summary: EvaluationSummary,
        roc: ROCCurve,
        scores: np.ndarray,
        labels: np.ndarray,
        embed_time_s: float,
    ) -> None:
        self.model_name   = model_name
        self.summary      = summary
        self.roc          = roc
        self.scores       = scores
        self.labels       = labels
        self.embed_time_s = embed_time_s
        self.num_pairs    = len(scores)

    # ------------------------------------------------------------------
    # Convenience properties — flat access to the most-used metrics
    # ------------------------------------------------------------------

    @property
    def eer(self) -> float:
        """Equal Error Rate (lower is better)."""
        return self.summary.eer

    @property
    def eer_threshold(self) -> float:
        """Threshold at which EER is achieved."""
        return self.summary.eer_threshold

    @property
    def auc(self) -> float:
        """Area under ROC curve (higher is better)."""
        return self.summary.auc

    @property
    def far(self) -> float:
        """FAR at the configured operating threshold."""
        return self.summary.far

    @property
    def frr(self) -> float:
        """FRR at the configured operating threshold."""
        return self.summary.frr

    @property
    def f1(self) -> float:
        """F1 at the configured operating threshold."""
        return self.summary.f1

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"model={self.model_name!r}, "
            f"pairs={self.num_pairs}, "
            f"EER={self.eer:.4f}, "
            f"AUC={self.auc:.4f}, "
            f"FAR={self.far:.4f}, "
            f"FRR={self.frr:.4f})"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class LFWEvaluator:
    """
    Stateless LFW face-verification evaluator.

    Extracts embeddings for every image pair in the LFW dataset, computes
    cosine similarities, and derives all verification metrics defined in
    src/evaluation/metrics.py.

    Args:
        dataset : LFWPairsDataset — pre-built, read-only test dataset.
        cfg     : Full Config — reads evaluation.* and data.* fields.
        tracker : ExperimentTracker — W&B logging (no-op when disabled).
        device  : Inference device.  Auto-detected (CUDA > MPS > CPU) if None.
    """

    def __init__(
        self,
        dataset: LFWPairsDataset,
        cfg: Config,
        tracker: ExperimentTracker,
        device: Optional[torch.device] = None,
    ) -> None:
        self._dataset  = dataset
        self._cfg      = cfg
        self._tracker  = tracker
        self._device   = device or self._auto_device()
        self._eval_cfg = cfg.evaluation

        logger.info(
            "LFWEvaluator ready | pairs=%d | device=%s | threshold=%.4f",
            len(dataset),
            self._device,
            self._eval_cfg.recognition_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: nn.Module,
        model_name: str,
        checkpoint_path: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run the full evaluation pipeline for one model.

        Pipeline
        --------
        1. Load backbone weights from checkpoint_path (optional).
        2. Extract (img1_embedding, img2_embedding) for every LFW pair.
        3. Compute per-pair cosine similarity → scores array.
        4. Compute FAR / FRR / EER / ROC / AUC / F1 via metrics.py.
        5. Log scalar metrics + ROC curve to W&B.

        Args:
            model:              Any nn.Module that produces L2-normalised embeddings.
                                FaceModel instances are handled natively via .embed().
            model_name:         Human-readable ID used for logging and W&B keys.
            checkpoint_path:    Path to a .pt checkpoint.  When provided, backbone
                                weights are loaded before inference.  Supports both
                                full Trainer checkpoints and backbone-only state dicts.

        Returns:
            EvaluationResult with all metrics and raw score / label arrays.
        """
        if checkpoint_path is not None:
            self._load_backbone(model, checkpoint_path)

        # ── Step 1: embed ───────────────────────────────────────────────
        t0 = time.perf_counter()
        emb1, emb2, labels = self._extract_pair_embeddings(model)
        embed_time = time.perf_counter() - t0

        # ── Step 2: score ───────────────────────────────────────────────
        scores = _cosine_similarity(emb1, emb2)   # (N,) float32

        # ── Step 3: metrics ─────────────────────────────────────────────
        threshold = self._eval_cfg.recognition_threshold
        n_thr     = self._eval_cfg.threshold_steps

        summary = evaluate_model(scores, labels, threshold, n_thresholds=n_thr)
        roc     = compute_roc(scores, labels, n_thresholds=n_thr)
        # AUC from the same roc object for consistency
        _ = compute_auc(roc)   # already stored in summary.auc via evaluate_model

        result = EvaluationResult(
            model_name   = model_name,
            summary      = summary,
            roc          = roc,
            scores       = scores,
            labels       = labels,
            embed_time_s = embed_time,
        )

        # ── Step 4: log ─────────────────────────────────────────────────
        self._log(result)

        logger.info(
            "LFW [%s] EER=%.4f  AUC=%.4f  FAR=%.4f  FRR=%.4f  "
            "F1=%.4f  embed=%.1fs",
            model_name,
            result.eer, result.auc,
            result.far, result.frr,
            result.f1,  embed_time,
        )
        return result

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_pair_embeddings(
        self,
        model: nn.Module,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the model over every pair in the LFW dataset.

        Returns three aligned numpy arrays:
            emb1   : (N, D) float32 — embedding of the first image in each pair.
            emb2   : (N, D) float32 — embedding of the second image in each pair.
            labels : (N,)  int32    — 1 = same identity, 0 = different.

        Implementation note
        -------------------
        Images arrive already preprocessed from LFWPairsDataset
        (resized, normalised, CHW float32).  We feed them directly to the model.
        No detection or alignment is performed here.
        """
        loader = DataLoader(
            self._dataset,
            batch_size = self._eval_cfg.batch_size,
            shuffle    = False,             # order must match labels
            num_workers = self._cfg.data.num_workers,
            pin_memory  = self._cfg.data.pin_memory,
            drop_last   = False,            # must evaluate ALL pairs
        )

        # Save training state; switch model to eval mode
        was_training = model.training
        model.eval()
        model.to(self._device)

        emb1_parts:  list[np.ndarray] = []
        emb2_parts:  list[np.ndarray] = []
        label_parts: list[np.ndarray] = []

        with torch.no_grad():
            for img1_batch, img2_batch, lbl_batch in loader:
                img1_batch = img1_batch.to(self._device, non_blocking=True)
                img2_batch = img2_batch.to(self._device, non_blocking=True)

                # Use the model's backbone only — never the loss head
                e1 = _get_embeddings(model, img1_batch)   # (B, D)
                e2 = _get_embeddings(model, img2_batch)   # (B, D)

                emb1_parts.append(e1.cpu().numpy())
                emb2_parts.append(e2.cpu().numpy())
                label_parts.append(
                    lbl_batch.numpy().astype(np.int32)
                )

        # Restore original training state
        model.train(was_training)

        emb1   = np.concatenate(emb1_parts,  axis=0).astype(np.float32)
        emb2   = np.concatenate(emb2_parts,  axis=0).astype(np.float32)
        labels = np.concatenate(label_parts, axis=0)

        assert emb1.shape[0] == len(self._dataset), (
            f"Embedding count mismatch: got {emb1.shape[0]}, "
            f"expected {len(self._dataset)}"
        )
        logger.debug(
            "Pair embeddings extracted: emb1=%s emb2=%s labels=%s",
            emb1.shape, emb2.shape, labels.shape,
        )
        return emb1, emb2, labels

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------

    def _log(self, result: EvaluationResult) -> None:
        """Push scalar metrics and the ROC curve to W&B."""
        name = result.model_name

        self._tracker.log_metrics({
            f"lfw/{name}/eer":           result.eer,
            f"lfw/{name}/eer_threshold": result.eer_threshold,
            f"lfw/{name}/auc":           result.auc,
            f"lfw/{name}/far":           result.far,
            f"lfw/{name}/frr":           result.frr,
            f"lfw/{name}/f1":            result.f1,
            f"lfw/{name}/embed_time_s":  result.embed_time_s,
        })

        self._tracker.log_roc_curve(
            fpr        = result.roc.fpr,
            tpr        = result.roc.tpr,
            auc        = result.auc,
            model_name = name,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(model: nn.Module, path: str) -> None:
        """
        Load backbone weights into model from a .pt checkpoint.

        Handles two checkpoint formats:
        1. Full Trainer checkpoint — keys prefixed with "backbone.*"
           saved via Trainer._save() → extracts backbone sub-dict.
        2. Backbone-only state dict — saved via FaceModel.backbone_state_dict().

        Uses strict=False so minor architecture differences (e.g. extra BN
        layers) don't hard-crash evaluation.

        Args:
            model : FaceModel or compatible module.
            path  : Path to the .pt checkpoint file.
        """
        ckpt = torch.load(path, map_location="cpu")

        # Full Trainer checkpoint contains "model_state"
        if "model_state" in ckpt:
            full_state = ckpt["model_state"]
            # Extract backbone sub-dict (strip "backbone." prefix)
            backbone_state = {
                k[len("backbone."):]: v
                for k, v in full_state.items()
                if k.startswith("backbone.")
            }
            logger.debug(
                "Extracted %d backbone keys from full checkpoint",
                len(backbone_state),
            )
        else:
            backbone_state = ckpt

        # Load into the model using the best available method
        if hasattr(model, "load_backbone_weights"):
            model.load_backbone_weights(backbone_state, strict=False)
        elif hasattr(model, "backbone"):
            missing, unexpected = model.backbone.load_state_dict(
                backbone_state, strict=False
            )
            if missing:
                logger.warning("Backbone load: missing keys — %s", missing[:5])
            if unexpected:
                logger.warning("Backbone load: unexpected keys — %s", unexpected[:5])
        else:
            model.load_state_dict(backbone_state, strict=False)

        logger.info("Backbone weights loaded from %s", path)

    # ------------------------------------------------------------------
    # Auto device
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def __repr__(self) -> str:
        return (
            f"LFWEvaluator("
            f"pairs={len(self._dataset)}, "
            f"device={self._device}, "
            f"threshold={self._eval_cfg.recognition_threshold})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no state)
# ---------------------------------------------------------------------------


def _get_embeddings(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Extract L2-normalised embeddings from a model without invoking the loss head.

    Priority order:
        1. model.embed(images)      — FaceModel explicit inference helper.
        2. model.backbone(images)   — direct backbone access.
        3. model(images)            — generic fallback (no labels → no loss head).

    Args:
        model:  Any face embedding model.
        images: (B, 3, H, W) float32 tensor already on the correct device.

    Returns:
        (B, D) float32 L2-normalised embeddings.
    """
    if hasattr(model, "embed"):
        return model.embed(images)
    if hasattr(model, "backbone"):
        return model.backbone(images)
    out = model(images)
    # Guard against models that return (loss, embeddings) tuples
    if isinstance(out, tuple):
        out = out[1]
    return out


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Per-pair cosine similarity between two (N, D) embedding matrices.

    Both matrices are L2-normalised before the dot product as a safety
    measure — the backbone should already produce unit vectors, but this
    costs almost nothing and makes the function robust to any model.

    Args:
        emb1: (N, D) float32 — first image embeddings.
        emb2: (N, D) float32 — second image embeddings.

    Returns:
        (N,) float32 cosine similarities in [-1, 1].
    """
    # Re-normalise to unit vectors
    n1 = np.linalg.norm(emb1, axis=1, keepdims=True).clip(min=1e-10)
    n2 = np.linalg.norm(emb2, axis=1, keepdims=True).clip(min=1e-10)
    e1 = emb1 / n1
    e2 = emb2 / n2

    # Row-wise dot product = cosine similarity for unit vectors
    return (e1 * e2).sum(axis=1).astype(np.float32)