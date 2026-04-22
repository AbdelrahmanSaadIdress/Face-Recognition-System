"""
src/data/pk_sampler.py
======================
PK Sampler — structured batch construction for Triplet Loss training.

Why this exists
---------------
Standard random shuffling fills batches by drawing images uniformly across
all identities. With thousands of identities, most batches end up with only
one image per person, leaving every anchor without a valid positive in the
batch. The triplet loss then silently collapses to near-zero gradients from
the very first epoch.

PK sampling fixes this by constructing every batch as:
    P identities × K images each  →  batch of P*K images

This guarantees every anchor has K-1 valid positives in its batch, making
hard and semi-hard mining well-defined at every training step.

This sampler is ONLY used for Triplet Loss. ArcFace and SphereFace use the
standard DataLoader shuffle and are unaffected.

Usage
-----
    sampler = PKSampler(dataset, p=32, k=4)
    loader  = DataLoader(dataset, batch_sampler=sampler, ...)

Note: use `batch_sampler=` (not `sampler=`), and do NOT set `batch_size`,
`shuffle`, or `drop_last` on the DataLoader when using batch_sampler.
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
    Yields batches of size P*K where each batch contains exactly K images
    from each of P randomly chosen identities.

    Args:
        dataset:       TrainFaceDataset (or any Subset wrapping one).
                       Must expose a `_samples` list of (path, label) tuples,
                       or a wrapped dataset that does via `.dataset._samples`.
        p:             Number of identities per batch.
        k:             Number of images per identity per batch.
        drop_incomplete: If True, skip the last batch when fewer than P
                         identities remain. Keeps batch sizes uniform —
                         recommended for BatchNorm stability.

    Raises:
        ValueError: If p > number of valid identities (those with >= k images).
    """

    def __init__(
        self,
        dataset,
        p: int = 32,
        k: int = 4,
        drop_incomplete: bool = True,
    ) -> None:
        self.p = p
        self.k = k
        self.drop_incomplete = drop_incomplete

        # Support both raw TrainFaceDataset and Subset-wrapped versions
        samples = self._extract_samples(dataset)

        # Build label → [indices] map
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(samples):
            label_to_indices[label].append(idx)

        # Keep only identities that have at least K images
        self._label_to_indices: dict[int, list[int]] = {
            label: indices
            for label, indices in label_to_indices.items()
            if len(indices) >= k
        }

        n_valid = len(self._label_to_indices)
        n_skipped = len(label_to_indices) - n_valid

        if n_skipped > 0:
            logger.warning(
                "PKSampler: skipped %d identities with fewer than k=%d images.",
                n_skipped, k,
            )

        if n_valid < p:
            raise ValueError(
                f"PKSampler: only {n_valid} identities have >= {k} images, "
                f"but p={p} identities are required per batch. "
                f"Reduce p or lower k."
            )

        self._labels = list(self._label_to_indices.keys())
        logger.info(
            "PKSampler ready | p=%d | k=%d | valid_identities=%d | "
            "batch_size=%d | approx_batches_per_epoch=%d",
            p, k, n_valid, p * k,
            len(self) ,
        )

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yield one batch (list of P*K indices) per iteration.

        Each epoch shuffles identities and images independently so the
        model sees different combinations every epoch.
        """
        labels = self._labels.copy()
        random.shuffle(labels)

        batch: list[int] = []

        for label in labels:
            indices = self._label_to_indices[label].copy()
            random.shuffle(indices)
            # Take exactly K images; cycle if fewer available (shouldn't
            # happen given the >= k filter above, but defensive)
            chosen = indices[:self.k]
            batch.extend(chosen)

            if len(batch) == self.p * self.k:
                yield batch
                batch = []

        # Yield the last incomplete batch only if allowed
        if batch and not self.drop_incomplete:
            yield batch

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self._labels) // self.p
        return n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_samples(dataset) -> list[tuple]:
        """
        Retrieve the raw (path, label) samples list from the dataset,
        handling both raw TrainFaceDataset and torch.utils.data.Subset.
        """
        # Direct dataset
        if hasattr(dataset, "_samples"):
            return dataset._samples

        # Subset wrapping a TrainFaceDataset
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "_samples"):
            all_samples = dataset.dataset._samples
            return [all_samples[i] for i in dataset.indices]

        raise AttributeError(
            "PKSampler could not find '_samples' on the dataset or its "
            "wrapped '.dataset'. Ensure the dataset is a TrainFaceDataset "
            "or a Subset of one."
        )

    def __repr__(self) -> str:
        return (
            f"PKSampler("
            f"p={self.p}, k={self.k}, "
            f"identities={len(self._labels)}, "
            f"batch_size={self.p * self.k})"
        )