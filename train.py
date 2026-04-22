"""
train.py
========
Training entrypoint for the face recognition benchmarking system.

Usage
-----
    # Default config (configs/base.yaml)
    python train.py

    # Custom config
    python train.py --config configs/arcface_resnet50.yaml

    # Resume from checkpoint
    python train.py --resume checkpoints/arcface_resnet50_latest.pt

    # Ad-hoc overrides (no YAML edit needed)
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

# ── project imports ────────────────────────────────────────────────────────
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
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--set", "-s",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Ad-hoc config overrides, e.g.  "
            "--set training.learning_rate=0.01 model.name=triplet"
        ),
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.3,
        help="Fraction of training data held out for validation (0 = no val).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="Experiment",
        help="W&B run name. Auto-generated from model+backbone if omitted.",
    )
    return parser.parse_args()


def _parse_overrides(overrides: list[str]) -> dict:
    """
    Convert ['training.learning_rate=0.01', 'model.name=triplet']
    into a nested dict {'training': {'learning_rate': 0.01}, 'model': {'name': 'triplet'}}.
    """
    result: dict = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")
        key_path, raw_value = item.split("=", 1)

        # Attempt numeric cast
        try:
            value: object = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                # Handle booleans
                if raw_value.lower() in ("true", "false"):
                    value = raw_value.lower() == "true"
                else:
                    value = raw_value

        # Build nested dict from dotted key
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
    """Set all global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops — slight perf cost, required for exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Seeds set to %d", seed)


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------


# def build_dataloaders(
#     cfg: Config,
#     val_split: float,
# ) -> tuple[DataLoader, Optional[DataLoader]]:
#     """
#     Build train (and optional val) DataLoaders from the training dataset.

#     Args:
#         cfg:       Full config.
#         val_split: Fraction of data to hold out for validation.
#                    0.0 → no validation loader returned.

#     Returns:
#         (train_loader, val_loader)   val_loader is None when val_split=0.
#     """
#     full_dataset = TrainFaceDataset(
#         data_cfg=cfg.data,
#         preproc_cfg=cfg.preprocessing,
#         split="train",
#     )

#     if val_split > 0.0:
#         n_total = len(full_dataset)
#         n_val   = max(1, int(n_total * val_split))
#         n_train = n_total - n_val

#         train_ds, val_ds = random_split(
#             full_dataset,
#             [n_train, n_val],
#             generator=torch.Generator().manual_seed(cfg.project.seed),
#         )
#         # Validation split: disable augmentation by wrapping in a no-aug dataset
#         # We rebuild with split="val" for a clean no-aug pipeline
#         val_ds_clean = TrainFaceDataset(
#             data_cfg=cfg.data,
#             preproc_cfg=cfg.preprocessing,
#             split="val",
#         )
#         # Use the same indices the random_split assigned
#         from torch.utils.data import Subset
#         val_ds_clean = Subset(val_ds_clean, val_ds.indices)  # type: ignore[attr-defined]

#         logger.info("Dataset split: train=%d  val=%d", n_train, n_val)
#     else:
#         train_ds = full_dataset
#         val_ds_clean = None
#         logger.info("No validation split (val_split=0)")

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=cfg.training.batch_size,
#         shuffle=True,
#         num_workers=cfg.data.num_workers,
#         pin_memory=cfg.data.pin_memory,
#         drop_last=True,               # keeps batch sizes uniform (important for BN)
#         persistent_workers=cfg.data.num_workers > 0,
#     )

#     val_loader: Optional[DataLoader] = None
#     if val_ds_clean is not None:
#         val_loader = DataLoader(
#             val_ds_clean,
#             batch_size=cfg.training.batch_size * 2,   # larger batches OK without grad
#             shuffle=False,
#             num_workers=cfg.data.num_workers,
#             pin_memory=cfg.data.pin_memory,
#             persistent_workers=cfg.data.num_workers > 0,
#         )

#     return train_loader, val_loader

def build_dataloaders(
    cfg: Config,
    val_split: float,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optional val) DataLoaders from the training dataset.

    For Triplet Loss, the train loader uses PKSampler to guarantee that
    every batch contains P identities × K images each — a requirement for
    meaningful hard mining. ArcFace and SphereFace use standard random
    shuffling, unchanged from previous rounds.

    Args:
        cfg:       Full config.
        val_split: Fraction of data to hold out for validation.
                   0.0 → no validation loader returned.

    Returns:
        (train_loader, val_loader)   val_loader is None when val_split=0.
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
        val_ds_clean = TrainFaceDataset(
            data_cfg=cfg.data,
            preproc_cfg=cfg.preprocessing,
            split="val",
        )
        from torch.utils.data import Subset
        val_ds_clean = Subset(val_ds_clean, val_ds.indices)  # type: ignore[attr-defined]

        logger.info("Dataset split: train=%d  val=%d", n_train, n_val)
    else:
        train_ds = full_dataset
        val_ds_clean = None
        logger.info("No validation split (val_split=0)")

    # ── Train loader — PK sampling for triplet, random shuffle for everything else ──
    if cfg.model.name == "triplet":
        from src.data.pk_sampler import PKSampler

        pk_p = getattr(cfg.loss.triplet, "pk_p", 32)
        pk_k = getattr(cfg.loss.triplet, "pk_k", 4)

        sampler = PKSampler(train_ds, p=pk_p, k=pk_k, drop_incomplete=True)

        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,          # batch_sampler controls batch construction
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

    # ── Val loader — always random, unchanged for all models ──
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

def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """
    Build SGD optimizer with momentum and weight decay.

    Uses SGD because it consistently outperforms Adam for face recognition
    with large-margin losses on standard benchmarks.

    Args:
        model: The FaceModel.
        cfg:   Full config.

    Returns:
        Configured SGD optimizer.
    """
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
    """
    Build LR scheduler from config.

    Supported: 'cosine', 'step', 'multistep'.

    Args:
        optimizer: The optimizer to schedule.
        cfg:       Full config.

    Returns:
        LR scheduler.

    Raises:
        ValueError: If cfg.training.lr_scheduler is unrecognised.
    """
    sched_name = cfg.training.lr_scheduler
    epochs     = cfg.training.epochs
    warmup     = cfg.training.warmup_epochs

    if sched_name == "cosine":
        # CosineAnnealingLR after warmup
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup, 1),
            eta_min=1e-6,
        )
    elif sched_name == "step":
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.lr_step_size,
            gamma=cfg.training.lr_gamma,
        )
    elif sched_name == "multistep":
        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.training.lr_milestones,
            gamma=cfg.training.lr_gamma,
        )
    else:
        raise ValueError(
            f"Unknown lr_scheduler {sched_name!r}. "
            "Valid: 'cosine', 'step', 'multistep'"
        )

    if warmup > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, base_scheduler],
            milestones=[warmup],
        )

    return base_scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # ── 1. Load config ────────────────────────────────────────────────────
    cfg = load_config(args.config)

    if args.set:
        overrides = _parse_overrides(args.set)
        cfg = override_config(cfg, overrides)
        logger.info("Applied overrides: %s", overrides)

    # ── 2. Seed everything ────────────────────────────────────────────────
    seed_everything(cfg.project.seed)

    # ── 3. Build datasets + loaders ───────────────────────────────────────
    logger.info("Building datasets…")
    train_loader, val_loader = build_dataloaders(cfg, val_split=args.val_split)

    # Infer num_classes from the training dataset
    # (TrainFaceDataset's label_map is keyed by identity name)
    num_classes: int = train_loader.dataset.dataset.num_identities  # type: ignore[attr-defined]
    logger.info("num_classes (identities): %d", num_classes)

    # ── 4. Build model ────────────────────────────────────────────────────
    logger.info("Building model: %s + %s", cfg.model.name, cfg.model.backbone)
    model = build_face_model(cfg, num_classes=num_classes)

    # ── 5. Build optimizer + scheduler ───────────────────────────────────
    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg)

    # ── 6. Build tracker + start W&B run ─────────────────────────────────
    tracker  = ExperimentTracker(cfg)
    run_name = (
        args.run_name
        or f"{cfg.model.name}_{cfg.model.backbone}_seed{cfg.project.seed}"
    )
    tracker.start_run(
        run_name=run_name,
        config_overrides={
            "num_classes":  num_classes,
            "val_split":    args.val_split,
            "resume":       args.resume,
        },
    )

    # ── 7. Build trainer ──────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        tracker=tracker,
    )

    # ── 8. Resume (optional) ──────────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)
        trainer.load_checkpoint(resume_path)

    # ── 9. Train ──────────────────────────────────────────────────────────
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

    # ── 10. Upload best checkpoint as W&B artifact ────────────────────────
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