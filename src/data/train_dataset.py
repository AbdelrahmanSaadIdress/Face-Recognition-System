"""
Training dataset — identity-labelled face crops.

Expected directory layout:
    train_path/
        <identity_name>/
            img_001.jpg
            img_002.jpg
            ...
        <identity_name>/
            ...

Contract
--------
- Each subdirectory is one identity.
- Images are pre-cropped faces; no detection is run.
- Augmentation is applied on-the-fly when `split="train"`.
- `PreprocessingPipeline` handles resize + normalize + HWC→CHW.
- Returns (image_tensor, label_int).
- Validates min_identities and min_images_per_identity at init time.
- Labels are contiguous integers 0 … N-1, alphabetically sorted by identity.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.config import AugmentationConfig, DataConfig, PreprocessingConfig
from src.data.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Augmentation transforms   (each: np.ndarray uint8 → np.ndarray uint8)
# ---------------------------------------------------------------------------


def _random_horizontal_flip(image: np.ndarray, prob: float) -> np.ndarray:
    if random.random() < prob:
        return cv2.flip(image, 1)
    return image


def _random_brightness_contrast(
    image: np.ndarray,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    """
    Apply random brightness and contrast jitter.

    brightness / contrast are the max delta magnitudes.
    Sampled uniformly in [-delta, +delta].
    """
    b_delta = random.uniform(-brightness, brightness) * 255.0
    c_factor = 1.0 + random.uniform(-contrast, contrast)
    out = image.astype(np.float32)
    out = out * c_factor + b_delta
    return np.clip(out, 0, 255).astype(np.uint8)


def _random_blur(image: np.ndarray, prob: float, kernel_size: int) -> np.ndarray:
    if random.random() < prob:
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(image, (k, k), sigmaX=0)
    return image


def _random_occlusion(
    image: np.ndarray,
    prob: float,
    max_ratio: float,
) -> np.ndarray:
    """
    Zero-out a random rectangular patch (coarse dropout / occlusion).

    Args:
        image:     HxWx3 uint8.
        prob:      Probability of applying the occlusion.
        max_ratio: Maximum fraction of the image dimension to occlude.
    """
    if random.random() >= prob:
        return image

    h, w = image.shape[:2]
    patch_h = random.randint(1, max(1, int(h * max_ratio)))
    patch_w = random.randint(1, max(1, int(w * max_ratio)))
    y0 = random.randint(0, h - patch_h)
    x0 = random.randint(0, w - patch_w)

    out = image.copy()
    out[y0 : y0 + patch_h, x0 : x0 + patch_w] = 0
    return out


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


class AugmentationPipeline:
    """
    On-the-fly augmentation for training.

    Applied BEFORE `PreprocessingPipeline` (operates on uint8 HxWx3).
    All transforms are parametrised by `AugmentationConfig`.

    Args:
        aug_cfg: AugmentationConfig controlling probabilities / magnitudes.
    """

    def __init__(self, aug_cfg: AugmentationConfig) -> None:
        self._cfg = aug_cfg

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a uint8 HxWx3 face crop.

        Args:
            image: HxWx3 uint8 RGB array.

        Returns:
            Augmented HxWx3 uint8 RGB array.
        """
        if not self._cfg.enabled:
            return image

        image = _random_horizontal_flip(image, self._cfg.horizontal_flip_prob)
        image = _random_brightness_contrast(
            image,
            self._cfg.brightness_jitter,
            self._cfg.contrast_jitter,
        )
        image = _random_blur(image, self._cfg.blur_prob, self._cfg.blur_kernel_size)
        image = _random_occlusion(
            image,
            self._cfg.occlusion_prob,
            self._cfg.occlusion_max_ratio,
        )
        return image

    def __repr__(self) -> str:
        return (
            f"AugmentationPipeline(enabled={self._cfg.enabled}, "
            f"flip={self._cfg.horizontal_flip_prob}, "
            f"blur_prob={self._cfg.blur_prob}, "
            f"occlusion_prob={self._cfg.occlusion_prob})"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TrainFaceDataset(Dataset):
    """
    Identity-labelled training dataset for face recognition.

    Items: (preprocessed_image, identity_label)
        preprocessed_image : float32 (3, H, W), normalised.
        identity_label     : int in [0, num_identities).

    Args:
        data_cfg:       DataConfig — paths, image_size, min requirements.
        preproc_cfg:    PreprocessingConfig — normalisation params.
        split:          "train" enables augmentation; "val" disables it.
        transform:      Optional extra torchvision-style transform applied
                        after the preprocessing pipeline.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        preproc_cfg: PreprocessingConfig,
        split: str = "train",
        transform: Optional[object] = None,
    ) -> None:
        self._split = split
        self._transform = transform

        self._pipeline = PreprocessingPipeline(
            preproc_cfg=preproc_cfg,
            image_size=data_cfg.image_size,
            apply_detection=False,
        )
        self._augment: Optional[AugmentationPipeline] = (
            AugmentationPipeline(data_cfg.augmentation)
            if split == "train"
            else None
        )

        root = Path(data_cfg.train_path)
        self._samples, self._label_map = self._scan_directory(
            root=root,
            min_identities=data_cfg.min_identities,
            min_images=data_cfg.min_images_per_identity,
        )

        logger.info(
            "TrainFaceDataset (%s): %d identities, %d samples — %s",
            split,
            self.num_identities,
            len(self._samples),
            root,
        )

    # ------------------------------------------------------------------
    # Directory scan
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_directory(
        root: Path,
        min_identities: int,
        min_images: int,
    ) -> tuple[list[tuple[Path, int]], dict[str, int]]:
        """
        Walk `root`, build (image_path, label) sample list.

        Args:
            root:            Dataset root — one subdir per identity.
            min_identities:  Minimum acceptable identity count.
            min_images:      Minimum images required per identity.

        Returns:
            samples:   List of (Path, int) tuples.
            label_map: Dict mapping identity_name → integer label.

        Raises:
            FileNotFoundError: If `root` does not exist.
            ValueError:        If dataset fails minimum requirements.
        """
        if not root.exists():
            raise FileNotFoundError(f"Training data root not found: {root}")

        identity_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir()]
        )

        if not identity_dirs:
            raise ValueError(f"No identity subdirectories found in {root}")

        label_map: dict[str, int] = {}
        samples: list[tuple[Path, int]] = []
        skipped: list[str] = []

        for label_idx, identity_dir in enumerate(identity_dirs):
            images = [
                p for p in identity_dir.iterdir()
                if p.suffix.lower() in _SUPPORTED_EXTENSIONS
            ]
            if len(images) < min_images:
                skipped.append(identity_dir.name)
                continue

            label_map[identity_dir.name] = len(label_map)
            for img_path in sorted(images):
                samples.append((img_path, label_map[identity_dir.name]))

        if skipped:
            logger.warning(
                "Skipped %d identities with < %d images: %s%s",
                len(skipped),
                min_images,
                ", ".join(skipped[:5]),
                "..." if len(skipped) > 5 else "",
            )

        num_identities = len(label_map)
        if num_identities < min_identities:
            raise ValueError(
                f"Dataset has only {num_identities} valid identities "
                f"(minimum required: {min_identities}). "
                f"Ensure at least {min_images} images per identity."
            )

        return samples, label_map

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        img_path, label = self._samples[idx]

        image = self._load_image(img_path)

        if self._augment is not None:
            image = self._augment(image)

        preprocessed = self._pipeline(image)

        if self._transform is not None:
            preprocessed = self._transform(preprocessed)

        return preprocessed, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """
        Load an image as HxWx3 uint8 RGB.

        Args:
            path: Path to image file.

        Returns:
            HxWx3 uint8 RGB numpy array.

        Raises:
            FileNotFoundError: If the file does not exist.
            OSError:           If PIL cannot open the file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Training image not found: {path}")
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Properties / utilities
    # ------------------------------------------------------------------

    @property
    def num_identities(self) -> int:
        """Number of unique identities in the dataset."""
        return len(self._label_map)

    @property
    def label_map(self) -> dict[str, int]:
        """Read-only mapping from identity name to integer label."""
        return dict(self._label_map)

    def get_identity_name(self, label: int) -> str:
        """
        Reverse-lookup: integer label → identity name.

        Args:
            label: Integer class label.

        Returns:
            Identity name string.

        Raises:
            KeyError: If label is out of range.
        """
        inverse = {v: k for k, v in self._label_map.items()}
        if label not in inverse:
            raise KeyError(f"Label {label} not in dataset (0–{self.num_identities - 1})")
        return inverse[label]

    def class_counts(self) -> dict[int, int]:
        """
        Return per-identity image counts.

        Returns:
            Dict mapping label → count.
        """
        counts: dict[int, int] = {}
        for _, label in self._samples:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"TrainFaceDataset("
            f"split={self._split!r}, "
            f"identities={self.num_identities}, "
            f"samples={len(self._samples)}, "
            f"augment={self._augment is not None})"
        )