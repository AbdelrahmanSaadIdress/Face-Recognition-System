"""
src/evaluation/metrics.py
=========================
Pure-numpy evaluation metrics for face recognition.

Metric definitions (NIST/ISO conventions)
------------------------------------------
    FAR  = FP / (FP + TN)   — fraction of impostors accepted
    FRR  = FN / (FN + TP)   — fraction of genuine pairs rejected
    EER  — threshold where FAR ≈ FRR  (interpolated)
    TAR  = 1 - FRR           — true-accept rate (= recall)
    ROC  — (FAR, TAR) curve  swept over thresholds
    AUC  — area under the ROC curve  (trapezoidal rule)
    F1   — harmonic mean of precision and recall at a fixed threshold
"""

from __future__ import annotations

import logging
import time
from typing import Callable, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_scores_labels(
    scores: np.ndarray,
    labels: np.ndarray,
) -> None:
    """
    Shared input guard used by every public metric function.

    Args:
        scores: 1-D float array of similarity / distance scores.
        labels: 1-D int array — 1 = genuine pair, 0 = impostor pair.

    Raises:
        TypeError:  If inputs are not numpy arrays.
        ValueError: If shapes mismatch, labels are not binary, or arrays
                    are empty.
    """
    if not isinstance(scores, np.ndarray):
        raise TypeError(f"scores must be np.ndarray, got {type(scores).__name__}")
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"labels must be np.ndarray, got {type(labels).__name__}")
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"scores and labels must have the same length: "
            f"{scores.shape[0]} vs {labels.shape[0]}"
        )
    if scores.shape[0] == 0:
        raise ValueError("scores and labels must not be empty.")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"labels must be binary (0/1), found values: {unique}")


def _confusion_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    """
    Compute TP, FP, TN, FN at a single threshold.

    A pair is *accepted* (predicted genuine) when score >= threshold.

    Args:
        scores:    1-D float array — higher = more similar.
        labels:    1-D int array — 1 = genuine, 0 = impostor.
        threshold: Decision boundary.

    Returns:
        (TP, FP, TN, FN) as Python ints.
    """
    predicted = (scores >= threshold).astype(np.int32)
    tp = int(np.sum((predicted == 1) & (labels == 1)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    tn = int(np.sum((predicted == 0) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))
    return tp, fp, tn, fn


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def compute_far(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """
    False Accept Rate — fraction of impostor pairs accepted.

    FAR = FP / (FP + TN)

    Args:
        scores:    1-D float32 similarity scores.
        labels:    1-D int32 ground-truth labels (1=genuine, 0=impostor).
        threshold: Decision threshold.

    Returns:
        FAR in [0.0, 1.0].  Returns 0.0 when there are no impostors.
    """
    _validate_scores_labels(scores, labels)
    _, fp, tn, _ = _confusion_at_threshold(scores, labels, threshold)
    denom = fp + tn
    if denom == 0:
        logger.warning("compute_far: no impostor pairs found; returning 0.0")
        return 0.0
    return float(fp / denom)


def compute_frr(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """
    False Reject Rate — fraction of genuine pairs rejected.

    FRR = FN / (FN + TP)

    Args:
        scores:    1-D float32 similarity scores.
        labels:    1-D int32 ground-truth labels (1=genuine, 0=impostor).
        threshold: Decision threshold.

    Returns:
        FRR in [0.0, 1.0].  Returns 0.0 when there are no genuine pairs.
    """
    _validate_scores_labels(scores, labels)
    tp, _, _, fn = _confusion_at_threshold(scores, labels, threshold)
    denom = fn + tp
    if denom == 0:
        logger.warning("compute_frr: no genuine pairs found; returning 0.0")
        return 0.0
    return float(fn / denom)


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 1000,
) -> tuple[float, float]:
    """
    Equal Error Rate — interpolated threshold where FAR ≈ FRR.

    Algorithm:
        1. Sweep `n_thresholds` linearly spaced thresholds ∈ [min, max].
        2. For each threshold compute FAR and FRR.
        3. Find adjacent thresholds where (FAR - FRR) changes sign.
        4. Linear-interpolate to find the crossing point.
    
    Lower EER = better system
    Higher EER = worse system   
    
    Args:
        scores:       1-D float32 similarity scores.
        labels:       1-D int32 ground-truth labels.
        n_thresholds: Number of threshold steps (default 1000).

    Returns:
        (eer, eer_threshold) — EER value and the corresponding threshold,
        both as Python floats.
    """
    _validate_scores_labels(scores, labels)
    thresholds = np.linspace(float(scores.min()), float(scores.max()), n_thresholds)

    fars = np.array([compute_far(scores, labels, t) for t in thresholds])
    frrs = np.array([compute_frr(scores, labels, t) for t in thresholds])

    diff = fars - frrs
    # Find index where difference changes sign (FAR crosses below FRR)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # No crossing — return the threshold minimising |FAR - FRR|
        idx = int(np.argmin(np.abs(diff)))
        eer = float((fars[idx] + frrs[idx]) / 2.0)
        return eer, float(thresholds[idx])

    idx = int(sign_changes[0])
    # Linear interpolation between idx and idx+1
    d0, d1 = diff[idx], diff[idx + 1]
    t0, t1 = thresholds[idx], thresholds[idx + 1]
    eer_threshold = float(t0 - d0 * (t1 - t0) / (d1 - d0))
    eer = float((compute_far(scores, labels, eer_threshold) +
                compute_frr(scores, labels, eer_threshold)) / 2.0)
    return eer, eer_threshold


class ROCCurve(NamedTuple):
    """Container for a computed ROC curve."""
    fpr: np.ndarray   # 1-D float32, false-positive rates (= FAR)
    tpr: np.ndarray   # 1-D float32, true-positive rates  (= 1 - FRR)
    thresholds: np.ndarray  # 1-D float32 corresponding thresholds


def compute_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 1000,
) -> ROCCurve:
    """
    Compute the Receiver Operating Characteristic curve.

    Sweeps thresholds from high (strict) to low (lenient) so the curve
    runs left-to-right as per convention: (0,0) → (1,1).

    Args:
        scores:       1-D float32 similarity scores.
        labels:       1-D int32 ground-truth labels (1=genuine, 0=impostor).
        n_thresholds: Resolution of the threshold sweep (default 1000).

    Returns:
        ROCCurve(fpr, tpr, thresholds) — all arrays of length n_thresholds,
        dtype float32, sorted ascending by fpr.
    """
    _validate_scores_labels(scores, labels)
    thresholds = np.linspace(float(scores.max()), float(scores.min()), n_thresholds)

    fprs: list[float] = []
    tprs: list[float] = []

    for t in thresholds:
        tp, fp, tn, fn = _confusion_at_threshold(scores, labels, t)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fprs.append(fpr)
        tprs.append(tpr)

    fpr_arr = np.array(fprs, dtype=np.float32)
    tpr_arr = np.array(tprs, dtype=np.float32)

    # Sort by fpr for correct AUC computation
    sort_idx = np.argsort(fpr_arr)
    return ROCCurve(
        fpr=fpr_arr[sort_idx],
        tpr=tpr_arr[sort_idx],
        thresholds=thresholds[sort_idx],
    )


def compute_auc(roc: ROCCurve) -> float:
    """
    Area Under the ROC Curve using the trapezoidal rule.

    Args:
        roc: ROCCurve named-tuple (output of `compute_roc`).

    Returns:
        AUC in [0.0, 1.0] as a Python float.

    Raises:
        ValueError: If roc.fpr and roc.tpr have different lengths or are empty.
    """
    fpr, tpr = roc.fpr, roc.tpr
    if len(fpr) != len(tpr):
        raise ValueError(
            f"fpr and tpr must have equal length: {len(fpr)} vs {len(tpr)}"
        )
    if len(fpr) == 0:
        raise ValueError("ROC arrays must not be empty.")
    auc = float(np.trapz(tpr, fpr))
    # Clamp to [0, 1] — negative area means scores are inverted
    return max(0.0, min(1.0, auc))


def compute_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """
    F1 score at a fixed decision threshold.

    F1 = 2 * (precision * recall) / (precision + recall)
       = 2 * TP / (2 * TP + FP + FN)

    Args:
        scores:    1-D float32 similarity scores.
        labels:    1-D int32 ground-truth labels (1=genuine, 0=impostor).
        threshold: Fixed operating threshold.

    Returns:
        F1 in [0.0, 1.0].  Returns 0.0 when TP = 0.
    """
    _validate_scores_labels(scores, labels)
    tp, fp, _, fn = _confusion_at_threshold(scores, labels, threshold)
    denom = 2 * tp + fp + fn
    if denom == 0:
        logger.warning("compute_f1: all metrics are zero; returning 0.0")
        return 0.0
    return float(2 * tp / denom)


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------


class LatencyStats(NamedTuple):
    """Inference latency summary statistics (all values in milliseconds)."""
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_fps: float


def compute_latency_stats(
    inference_fn: Callable[[], None],
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> LatencyStats:
    """
    Benchmark inference latency of a callable.

    Executes `warmup_runs` calls (discarded), then times `timed_runs`
    calls and returns percentile statistics.

    Args:
        inference_fn: Zero-argument callable wrapping one forward pass.
        warmup_runs:  Number of warm-up iterations to prime caches/JIT.
        timed_runs:   Number of timed iterations.

    Returns:
        LatencyStats with mean, p50, p95, p99, min, max (ms), and fps.

    Raises:
        ValueError: If timed_runs < 1.
    """
    if timed_runs < 1:
        raise ValueError(f"timed_runs must be >= 1, got {timed_runs}")

    # Warm-up
    for _ in range(warmup_runs):
        inference_fn()

    # Timed runs
    latencies_ms: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        inference_fn()
        latencies_ms.append((time.perf_counter() - t0) * 1_000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    throughput_fps = 1_000.0 / mean_ms if mean_ms > 0 else float("inf")

    return LatencyStats(
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        throughput_fps=throughput_fps,
    )


# ---------------------------------------------------------------------------
# Convenience: full evaluation summary at a single threshold
# ---------------------------------------------------------------------------


class EvaluationSummary(NamedTuple):
    """All scalar metrics at a fixed operating threshold."""
    far: float
    frr: float
    f1: float
    eer: float
    eer_threshold: float
    auc: float
    threshold: float


def evaluate_model(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    n_thresholds: int = 1000,
) -> EvaluationSummary:
    """
    Compute all scalar metrics in one call.

    Computes FAR, FRR, F1 at `threshold`, plus EER and AUC over a full
    threshold sweep.

    Args:
        scores:       1-D float32 similarity scores (higher = more similar).
        labels:       1-D int32 ground-truth labels (1=genuine, 0=impostor).
        threshold:    Fixed operating point for FAR / FRR / F1.
        n_thresholds: Resolution for EER and ROC computation.

    Returns:
        EvaluationSummary named-tuple with all metrics as Python floats.
    """
    _validate_scores_labels(scores, labels)
    far = compute_far(scores, labels, threshold)
    frr = compute_frr(scores, labels, threshold)
    f1  = compute_f1(scores, labels, threshold)
    eer, eer_thr = compute_eer(scores, labels, n_thresholds)
    roc = compute_roc(scores, labels, n_thresholds)
    auc = compute_auc(roc)

    logger.info(
        "Evaluation @ threshold=%.4f — FAR=%.4f  FRR=%.4f  F1=%.4f  "
        "EER=%.4f  AUC=%.4f",
        threshold, far, frr, f1, eer, auc,
    )
    return EvaluationSummary(
        far=far,
        frr=frr,
        f1=f1,
        eer=eer,
        eer_threshold=eer_thr,
        auc=auc,
        threshold=threshold,
    )