"""
src/training/trainer.py
=======================
Unified training engine for all face recognition models.

Usage
-----
    trainer = Trainer(model, optimizer, scheduler, cfg, tracker)
    best_metric = trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.config import Config
from src.models.losses import TripletLoss
from src.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Model-agnostic training engine.

    Args:
        model:      FaceModel (backbone + loss head).
        optimizer:  Any torch Optimizer.
        scheduler:  Any torch LR scheduler (stepped once per epoch).
        cfg:        Full Config object.
        tracker:    ExperimentTracker (no-op when W&B disabled).
        device:     'cuda' / 'cpu' / 'mps' — auto-detected if None.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        cfg: Config,
        tracker: ExperimentTracker,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.tracker = tracker
        self.device = device or self._auto_device()

        self.model.to(self.device)

        self._use_amp: bool = cfg.training.amp and self.device.type == "cuda"
        self._scaler: GradScaler = GradScaler(enabled=self._use_amp)

        self._checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_loss: float = float("inf")
        self._start_epoch: int = 0

        # Detect once whether this is a triplet experiment so we pick the
        # right accuracy metric throughout the whole run.
        self._is_triplet: bool = isinstance(
            getattr(self.model, "loss_head", None), TripletLoss
        )

        logger.info(
            "Trainer ready | device=%s | amp=%s | checkpoint_dir=%s | "
            "acc_metric=%s",
            self.device,
            self._use_amp,
            self._checkpoint_dir,
            "nn_acc" if self._is_triplet else "acc",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> float:
        """
        Run the full training loop for cfg.training.epochs epochs.

        Returns:
            Best validation loss observed (or final train loss if no val).
        """
        total_epochs = self.cfg.training.epochs
        logger.info(
            "Starting training: epochs=%d, start_epoch=%d",
            total_epochs,
            self._start_epoch,
        )

        acc_key = "nn_acc" if self._is_triplet else "acc"

        for epoch in range(self._start_epoch, total_epochs):
            epoch_start = time.time()

            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            val_result            = self._val_epoch(val_loader, epoch) if val_loader else None
            val_loss              = val_result[0] if val_result is not None else None
            val_acc               = val_result[1] if val_result is not None else None

            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            epoch_time = time.time() - epoch_start

            last_global_step = (epoch + 1) * len(train_loader) - 1

            metrics = {
                "train/loss":         train_loss,
                f"train/{acc_key}":   train_acc,
                "train/lr":           current_lr,
                "epoch":              epoch,
                "epoch_time_s":       epoch_time,
            }
            if val_loss is not None:
                metrics["val/loss"]       = val_loss
                metrics[f"val/{acc_key}"] = val_acc

            # Use global step → always monotonically increasing
            self.tracker.log_metrics(metrics, step=last_global_step)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | train_%s=%.4f | "
                "val_loss=%s | val_%s=%s | lr=%.6f | %.1fs",
                epoch + 1, total_epochs,
                train_loss,
                acc_key, train_acc,
                f"{val_loss:.4f}"  if val_loss is not None else "—",
                acc_key,
                f"{val_acc:.4f}"   if val_acc  is not None else "—",
                current_lr,
                epoch_time,
            )

            monitor = val_loss if val_loss is not None else train_loss
            self._maybe_save_checkpoint(epoch, monitor)

        logger.info("Training complete. Best val loss: %.4f", self._best_val_loss)
        return self._best_val_loss

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> int:
        """Restore model + optimizer + scheduler from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])

        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))
        epoch = ckpt.get("epoch", 0)
        self._start_epoch = epoch + 1

        logger.info(
            "Resumed from checkpoint %s | epoch=%d | best_val_loss=%.4f",
            path, epoch, self._best_val_loss,
        )
        return epoch

    # ------------------------------------------------------------------
    # Internal: train / val epoch
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> tuple[float, float]:
        """
        Run one full training epoch.

        Returns:
            (mean_loss, mean_acc_or_nn_acc) over all batches.
        """
        self.model.train()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = len(loader)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self._use_amp):
                loss, acc_signal = self.model(
                    images, labels, return_logits=True
                )

            self._scaler.scale(loss).backward()

            if self.cfg.training.gradient_clip > 0.0:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.gradient_clip,
                )

            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.item()

            # acc_signal is a logit tensor for classification losses and a
            # plain float for triplet — normalise to a scalar here.
            if isinstance(acc_signal, torch.Tensor):
                total_acc += (
                    acc_signal.argmax(dim=1) == labels
                ).float().mean().item()
            else:
                total_acc += acc_signal   # already a float from forward_with_nn_acc

            # Per-step logging
            if (batch_idx + 1) % self.cfg.wandb.log_freq == 0:
                step = epoch * n_batches + batch_idx
                self.tracker.log_metrics(
                    {"train/step_loss": loss.item()}, step=step
                )

        return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(
        self, loader: DataLoader, epoch: int
    ) -> tuple[float, float]:
        """
        Run one full validation epoch (no gradients).

        Returns:
            (mean_loss, mean_acc_or_nn_acc) over all validation batches.
        """
        self.model.eval()
        total_loss = 0.0
        total_acc  = 0.0
        n_batches  = len(loader)

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self._use_amp):
                loss, acc_signal = self.model(
                    images, labels, return_logits=True
                )

            total_loss += loss.item()

            if isinstance(acc_signal, torch.Tensor):
                total_acc += (
                    acc_signal.argmax(dim=1) == labels
                ).float().mean().item()
            else:
                total_acc += acc_signal

        return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Internal: checkpointing
    # ------------------------------------------------------------------

    def _maybe_save_checkpoint(self, epoch: int, monitor: float) -> None:
        model_name = getattr(self.model, "model_name", "face_model")

        if (epoch + 1) % self.cfg.training.save_every_n_epochs == 0:
            path = self._checkpoint_dir / f"{model_name}_epoch_{epoch+1:04d}.pt"
            self._save(path, epoch, monitor)

        if monitor < self._best_val_loss:
            self._best_val_loss = monitor
            path = self._checkpoint_dir / f"{model_name}_best.pt"
            self._save(path, epoch, monitor)
            logger.info("New best checkpoint: loss=%.4f → %s", monitor, path.name)

        path = self._checkpoint_dir / f"{model_name}_latest.pt"
        self._save(path, epoch, monitor)

    def _save(self, path: Path, epoch: int, monitor: float) -> None:
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_loss":   self._best_val_loss,
                "monitor":         monitor,
                "config": {
                    "model_name":    self.cfg.model.name,
                    "backbone":      self.cfg.model.backbone,
                    "embedding_dim": self.cfg.model.embedding_dim,
                },
            },
            path,
        )
        logger.debug("Saved checkpoint: %s", path)

    # ------------------------------------------------------------------
    # Helpers
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
            f"Trainer("
            f"model={getattr(self.model, 'model_name', '?')}, "
            f"device={self.device}, "
            f"amp={self._use_amp}, "
            f"epochs={self.cfg.training.epochs})"
        )