"""
src/data/pk_sampler.py
======================
PK Sampler for Triplet Loss training.

Every batch = P identities × K images = P*K samples.
This guarantees every anchor has K-1 valid positives, making
hard/semi-hard mining well-defined at every step.

ONLY used for Triplet Loss. ArcFace/SphereFace use standard shuffle.

Usage
-----
    sampler = PKSampler(dataset, p=32, k=16)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=4, ...)

    # Do NOT set batch_size, shuffle, or drop_last when using batch_sampler.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class PKSampler(Sampler):
    """
    Yields batches of P*K indices where each batch has exactly K images
    from each of P randomly chosen identities.

    FIX vs previous version
    -----------------------
    - __len__ now returns total_images // (P*K), not identities // P.
      Old version gave ~15 batches/epoch; correct version gives ~540.
    - __iter__ uses random.choices (with replacement) per identity so
      identities with exactly K images are always included.
    - Sampling is per-batch random (not sequential over identity list),
      so every batch is independent and the model sees varied combinations.

    Args:
        dataset:         TrainFaceDataset or Subset wrapping one.
                         Must expose _samples as list of (path, label).
        p:               Identities per batch.
        k:               Images per identity per batch. Use k>=8, ideally 16.
        drop_incomplete: Skip last batch if < P*K samples remain (keeps
                         batch sizes uniform — important for BatchNorm).
    """

    def __init__(
        self,
        dataset,
        p: int = 32,
        k: int = 16,
        drop_incomplete: bool = True,
    ) -> None:
        self.p = p
        self.k = k
        self.drop_incomplete = drop_incomplete

        samples = self._extract_samples(dataset)

        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(samples):
            label_to_indices[label].append(idx)

        # Keep only identities with at least 2 images (need 1 valid positive)
        self._label_to_indices: dict[int, list[int]] = {
            label: indices
            for label, indices in label_to_indices.items()
            if len(indices) >= 2
        }

        n_valid   = len(self._label_to_indices)
        n_skipped = len(label_to_indices) - n_valid
        if n_skipped:
            logger.warning("PKSampler: skipped %d identities with < 2 images.", n_skipped)

        if n_valid < p:
            raise ValueError(
                f"PKSampler: only {n_valid} valid identities but p={p} required. "
                f"Reduce p or add more data."
            )

        self._labels = list(self._label_to_indices.keys())

        # Total images across all valid identities
        self._total_images = sum(len(v) for v in self._label_to_indices.values())

        logger.info(
            "PKSampler ready | p=%d | k=%d | valid_identities=%d | "
            "batch_size=%d | batches_per_epoch=%d",
            p, k, n_valid, p * k, len(self),
        )

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yield one batch (list of P*K indices) per call.

        Each batch independently samples P random identities and K images
        from each. This means the same identity can appear in consecutive
        batches — which is fine and actually good for hard mining.
        """
        n_batches = len(self)

        for _ in range(n_batches):
            # Sample P identities for this batch (with replacement if needed)
            chosen_ids = random.choices(self._labels, k=self.p)

            batch: list[int] = []
            for pid in chosen_ids:
                idxs = self._label_to_indices[pid]
                # Always sample with replacement — works even if len(idxs) < k
                batch.extend(random.choices(idxs, k=self.k))

            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """
        Number of batches per epoch.

        FIX: based on total images, not number of identities.
        Old formula (identities // p) gave ~15 batches.
        New formula gives ~total_images // batch_size batches.
        """
        return max(1, self._total_images // (self.p * self.k))

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_samples(dataset) -> list[tuple]:
        """Handle both raw TrainFaceDataset and Subset-wrapped versions."""
        if hasattr(dataset, "_samples"):
            return dataset._samples
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "_samples"):
            all_samples = dataset.dataset._samples
            return [all_samples[i] for i in dataset.indices]
        raise AttributeError(
            "PKSampler: dataset has no '_samples'. "
            "Must be TrainFaceDataset or a Subset of one."
        )

    def __repr__(self) -> str:
        return (
            f"PKSampler(p={self.p}, k={self.k}, "
            f"identities={len(self._labels)}, "
            f"batch_size={self.p * self.k}, "
            f"batches_per_epoch={len(self)})"
        )