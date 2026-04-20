"""
W&B experiment tracking wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)

# Lazy import — only resolve at runtime to avoid hard dep when disabled
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class ExperimentTracker:
    """
    Thin, typed wrapper around Weights & Biases.

    Usage
    -----
    tracker = ExperimentTracker(cfg)
    tracker.start_run(run_name="arcface_resnet50", config_overrides={})
    tracker.log_metrics({"loss": 0.42, "epoch": 1})
    tracker.log_roc(fpr=..., tpr=..., auc=0.99)
    tracker.finish_run()
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._enabled = cfg.wandb.enabled and _WANDB_AVAILABLE
        self._run: Any = None

        if cfg.wandb.enabled and not _WANDB_AVAILABLE:
            logger.warning(
                "wandb is enabled in config but the `wandb` package is not "
                "installed. All tracking calls will be silently skipped. "
                "Install with: pip install wandb"
            )

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        run_name: str,
        config_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialise a W&B run.

        Args:
            run_name:           Human-readable run identifier.
            config_overrides:   Extra key/value pairs merged into the logged
                                config (e.g. sweep parameters).
        """
        if not self._enabled:
            logger.info("Tracking disabled — skipping run init (%s)", run_name)
            return
        import dataclasses

        run_config = {
            **dataclasses.asdict(self._cfg),
            **(config_overrides or {}),
        }

        self._run = _wandb.init(
            project=self._cfg.wandb.project,
            entity=self._cfg.wandb.entity,
            name=run_name,
            config=run_config,
            tags=self._cfg.wandb.tags,
            notes=self._cfg.wandb.notes,
            # finish_previous=True,
        )
        # 1. High-frequency step-level metric
        self._run.define_metric("global_step")
        self._run.define_metric("train/step_loss", step_metric="global_step")

        # 2. Epoch-level metrics — be explicit, avoid the wildcard clash
        self._run.define_metric("epoch")
        self._run.define_metric("train/loss",    step_metric="epoch")
        self._run.define_metric("train/acc",     step_metric="epoch")
        self._run.define_metric("train/nn_acc",  step_metric="epoch")
        self._run.define_metric("train/lr",      step_metric="epoch")
        self._run.define_metric("val/loss",      step_metric="epoch")
        self._run.define_metric("val/acc",       step_metric="epoch")
        self._run.define_metric("val/nn_acc",    step_metric="epoch")
        self._run.define_metric("epoch_time_s",  step_metric="epoch")
        logger.info("W&B run started: %s", run_name)

    def finish_run(self) -> None:
        """Mark the current run as complete and flush all pending data."""
        if not self._enabled or self._run is None:
            return
        self._run.finish()
        self._run = None
        logger.info("W&B run finished")

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: dict[str, Union[float, int]],
        step: Optional[int] = None,
    ) -> None:
        """
        Log a dict of scalar metrics.

        Args:
            metrics: Flat dict — e.g. {"loss": 0.3, "far": 0.01}.
            step:    Global step counter.  Auto-incremented by W&B if None.
        """
        if not self._enabled or self._run is None:
            return
        self._run.log(metrics, step=step)

    def log_benchmark_table(
        self,
        rows: list[dict[str, Any]],
        table_name: str = "benchmark_results",
    ) -> None:
        """
        Log a comparison table (one row per model).

        Args:
            rows:       List of dicts — each dict is one table row.
            table_name: Key under which the table appears in the W&B UI.
        """
        if not self._enabled or self._run is None:
            return

        if not rows:
            logger.warning("log_benchmark_table called with empty rows — skipping")
            return

        columns = list(rows[0].keys())
        table = _wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[c] for c in columns])

        self._run.log({table_name: table})

    # ------------------------------------------------------------------
    # Curve logging
    # ------------------------------------------------------------------

    def log_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        model_name: str,
    ) -> None:
        """
        Log an ROC curve as a W&B custom chart.

        Args:
            fpr:        1-D array of false-positive rates.
            tpr:        1-D array of true-positive rates.
            auc:        Area under the ROC curve.
            model_name: Label for the legend entry.
        """
        if not self._enabled or self._run is None:
            return

        table = _wandb.Table(
            columns=["fpr", "tpr", "model"],
            data=[[float(f), float(t), model_name] for f, t in zip(fpr, tpr)],
        )
        self._run.log(
            {
                f"roc_curve/{model_name}": _wandb.plot.line(
                    table, "fpr", "tpr", title=f"ROC — {model_name} (AUC={auc:.4f})"
                ),
                f"auc/{model_name}": auc,
            }
        )

    # ------------------------------------------------------------------
    # Artifact / file logging
    # ------------------------------------------------------------------

    def log_model_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Upload a model checkpoint as a W&B artifact.

        Args:
            checkpoint_path: Path to the saved checkpoint file.
            metadata:        Optional dict attached to the artifact.
        """
        if not self._enabled or self._run is None:
            return

        path = Path(checkpoint_path)
        if not path.exists():
            logger.error("Checkpoint not found, skipping upload: %s", path)
            return

        artifact = _wandb.Artifact(
            name=f"checkpoint-{self._run.name}",
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)
        logger.info("Checkpoint uploaded to W&B: %s", path.name)

    # ------------------------------------------------------------------
    # Latency stats
    # ------------------------------------------------------------------

    def log_latency_stats(
        self,
        model_name: str,
        mean_ms: float,
        p95_ms: float,
        p99_ms: float,
        throughput_fps: float,
    ) -> None:
        """
        Log inference latency and throughput statistics.

        Args:
            model_name:     Identifier logged as a prefix.
            mean_ms:        Mean latency in milliseconds.
            p95_ms:         95th-percentile latency in milliseconds.
            p99_ms:         99th-percentile latency in milliseconds.
            throughput_fps: Throughput in images per second.
        """
        self.log_metrics(
            {
                f"latency/{model_name}/mean_ms": mean_ms,
                f"latency/{model_name}/p95_ms": p95_ms,
                f"latency/{model_name}/p99_ms": p99_ms,
                f"throughput/{model_name}/fps": throughput_fps,
            }
        )

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish_run()

    def __repr__(self) -> str:
        state = "active" if self._run is not None else "idle"
        return (
            f"ExperimentTracker("
            f"project={self._cfg.wandb.project!r}, "
            f"enabled={self._enabled}, "
            f"state={state})"
        )