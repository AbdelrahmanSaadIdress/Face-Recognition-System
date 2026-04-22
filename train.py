"""
train.py
========
Training entrypoint for the face recognition benchmarking system.

Usage
-----
    python train.py
    python train.py --config configs/arcface_resnet50.yaml
    python train.py --resume checkpoints/arcface_resnet50_latest.pt
    python train.py --set training.learning_rate=0.01 model.name=triplet
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import Config, load_config, override_config
from src.data.train_dataset import TrainFaceDataset
from src.models.face_model import build_face_model
from src.training.trainer import Trainer
from src.tracking import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a face recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--set", "-s", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--val-split", type=float, default=0.3)
    parser.add_argument("--run-name", type=str, default="Experiment")
    return parser.parse_args()


def _parse_overrides(overrides: list[str]) -> dict:
    result: dict = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        key_path, raw_value = item.split("=", 1)
        try:
            value: object = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                if raw_value.lower() in ("true", "false"):
                    value = raw_value.lower() == "true"
                else:
                    value = raw_value
        parts = key_path.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Seeds set to %d", seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap_dataset(ds):
    """Unwrap a Subset to get the underlying TrainFaceDataset."""
    if hasattr(ds, "dataset"):
        return ds.dataset
    return ds


def _get_num_classes(loader: DataLoader) -> int:
    """Safely get num_identities regardless of Subset wrapping."""
    return _unwrap_dataset(loader.dataset).num_identities


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


def build_dataloaders(
    cfg: Config,
    val_split: float,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and optional val DataLoaders.

    - Triplet: train loader uses PKSampler (P identities x K images per batch).
    - Others:  train loader uses standard random shuffle.
    - Val loader is ALWAYS a plain random DataLoader — never PKSampler.
      This is important: val_nn_acc must be computed globally across all val
      embeddings in trainer._val_epoch, not per-batch.
    """
    full_dataset = TrainFaceDataset(
        data_cfg=cfg.data,
        preproc_cfg=cfg.preprocessing,
        split="train",
    )

    if val_split > 0.0:
        n_total = len(full_dataset)
        n_val   = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        train_ds, val_ds = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.project.seed),
        )

        # Rebuild val dataset with augmentation disabled
        val_ds_clean = TrainFaceDataset(
            data_cfg=cfg.data,
            preproc_cfg=cfg.preprocessing,
            split="val",
        )
        from torch.utils.data import Subset
        val_ds_clean = Subset(val_ds_clean, val_ds.indices)  # type: ignore[attr-defined]

        logger.info("Dataset split: train=%d  val=%d", n_train, n_val)
    else:
        train_ds    = full_dataset
        val_ds_clean = None
        logger.info("No validation split (val_split=0)")

    # ── Train loader ────────────────────────────────────────────────────────
    if cfg.model.name == "triplet":
        from src.data.pk_sampler import PKSampler

        # FIX: use sensible defaults — k=16 gives batch_size=512
        pk_p = getattr(cfg.loss.triplet, "pk_p", 32)
        pk_k = getattr(cfg.loss.triplet, "pk_k", 16)  # was 4, now 16

        sampler = PKSampler(train_ds, p=pk_p, k=pk_k, drop_incomplete=True)

        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.num_workers > 0,
        )
        logger.info(
            "Triplet train loader: PKSampler | p=%d | k=%d | batch_size=%d",
            pk_p, pk_k, pk_p * pk_k,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
            persistent_workers=cfg.data.num_workers > 0,
        )

    # ── Val loader — always plain random, never PKSampler ──────────────────
    val_loader: Optional[DataLoader] = None
    if val_ds_clean is not None:
        val_loader = DataLoader(
            val_ds_clean,
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.num_workers > 0,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
        nesterov=True,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> torch.optim.lr_scheduler._LRScheduler:
    sched_name = cfg.training.lr_scheduler
    epochs     = cfg.training.epochs
    warmup     = cfg.training.warmup_epochs

    if sched_name == "cosine":
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup, 1), eta_min=1e-6,
        )
    elif sched_name == "step":
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.lr_step_size, gamma=cfg.training.lr_gamma,
        )
    elif sched_name == "multistep":
        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.training.lr_milestones, gamma=cfg.training.lr_gamma,
        )
    else:
        raise ValueError(f"Unknown lr_scheduler {sched_name!r}.")

    if warmup > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, base_scheduler], milestones=[warmup],
        )

    return base_scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    if args.set:
        overrides = _parse_overrides(args.set)
        cfg = override_config(cfg, overrides)
        logger.info("Applied overrides: %s", overrides)

    seed_everything(cfg.project.seed)

    logger.info("Building datasets…")
    train_loader, val_loader = build_dataloaders(cfg, val_split=args.val_split)

    # FIX: safe num_classes extraction regardless of Subset wrapping
    num_classes = _get_num_classes(train_loader)
    logger.info("num_classes (identities): %d", num_classes)

    logger.info("Building model: %s + %s", cfg.model.name, cfg.model.backbone)
    model = build_face_model(cfg, num_classes=num_classes)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    tracker  = ExperimentTracker(cfg)
    run_name = (
        args.run_name
        or f"{cfg.model.name}_{cfg.model.backbone}_seed{cfg.project.seed}"
    )
    tracker.start_run(
        run_name=run_name,
        config_overrides={
            "num_classes": num_classes,
            "val_split":   args.val_split,
            "resume":      args.resume,
        },
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        tracker=tracker,
    )

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)
        trainer.load_checkpoint(resume_path)

    logger.info("=" * 60)
    logger.info("Starting run: %s", run_name)
    logger.info("  model      : %s", cfg.model.name)
    logger.info("  backbone   : %s", cfg.model.backbone)
    logger.info("  embedding  : %d", cfg.model.embedding_dim)
    logger.info("  epochs     : %d", cfg.training.epochs)
    logger.info("  batch_size : %d", cfg.training.batch_size)
    logger.info("  lr         : %g", cfg.training.learning_rate)
    logger.info("  scheduler  : %s", cfg.training.lr_scheduler)
    logger.info("  amp        : %s", cfg.training.amp)
    logger.info("=" * 60)

    best_loss = trainer.fit(train_loader, val_loader)
    logger.info("Training finished. Best monitored loss: %.4f", best_loss)

    best_ckpt = Path(cfg.training.checkpoint_dir) / f"{model.model_name}_best.pt"
    if best_ckpt.exists():
        tracker.log_model_checkpoint(
            best_ckpt,
            metadata={
                "model_name":    cfg.model.name,
                "backbone":      cfg.model.backbone,
                "embedding_dim": cfg.model.embedding_dim,
                "num_classes":   num_classes,
                "best_loss":     best_loss,
            },
        )

    tracker.finish_run()
    logger.info("Done.")


if __name__ == "__main__":
    main()