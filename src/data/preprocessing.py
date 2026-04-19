"""
Preprocessing pipeline for face recognition.

Design decisions
----------------
- `apply_detection=False` always — caller guarantees crops contain only faces.
- Pipeline: resize → (optional channel swap) → to-float → normalize.
- Output is a (3, H, W) float32 numpy array, values in ~[-1, 1].
- No torch dependency here; callers wrap output in tensors themselves.
- Fully callable: `pipeline(image) -> np.ndarray`.
- Thread-safe: all state is read-only after construction.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import cv2
import numpy as np

from src.config import PreprocessingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual transforms  (each: np.ndarray → np.ndarray)
# ---------------------------------------------------------------------------


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize image to (size × size) using bilinear interpolation.

    Args:
        image: HxWx3 uint8 array.
        size:  Target edge length in pixels.

    Returns:
        (size × size × 3) uint8 array.
    """
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _to_float(image: np.ndarray) -> np.ndarray:
    """
    Cast uint8 [0, 255] → float32 [0.0, 1.0].

    Args:
        image: HxWx3 uint8 array.

    Returns:
        HxWx3 float32 array.
    """
    return image.astype(np.float32) / 255.0


def _normalize(
    image: np.ndarray,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """
    Per-channel mean/std normalisation.

    Formula: (pixel - mean) / std

    Args:
        image: HxWx3 float32 array in [0, 1].
        mean:  3-element list of per-channel means.
        std:   3-element list of per-channel stds.

    Returns:
        HxWx3 float32 array.
    """
    mean_arr = np.array(mean, dtype=np.float32)   # (3,)
    std_arr  = np.array(std,  dtype=np.float32)   # (3,)
    return (image - mean_arr) / std_arr


def _hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """
    Transpose HxWxC → CxHxW (PyTorch convention).

    Args:
        image: HxWxC float32 array.

    Returns:
        CxHxW float32 array.
    """
    return np.transpose(image, (2, 0, 1))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """
    Stateless, callable preprocessing pipeline for face crops.

    Applies in order:
        1. Validate input shape / dtype
        2. Resize to image_size × image_size
        3. Cast to float32 [0, 1]
        4. Normalize (mean / std)
        5. Transpose to CxHxW

    Args:
        preproc_cfg:      PreprocessingConfig with mean/std.
        image_size:       Target spatial resolution (from DataConfig).
        apply_detection:  Must be False — detection is outside scope.

    Usage:
        pipeline = PreprocessingPipeline(preproc_cfg, image_size=112)
        tensor_chw = pipeline(face_crop_hwc_uint8)
    """

    def __init__(
        self,
        preproc_cfg: PreprocessingConfig,
        image_size: int,
        apply_detection: bool = False,
    ) -> None:
        if apply_detection:
            raise ValueError(
                "PreprocessingPipeline does not perform detection. "
                "Pass apply_detection=False and supply pre-cropped faces."
            )

        self._image_size = image_size
        self._mean = preproc_cfg.normalization_mean
        self._std  = preproc_cfg.normalization_std

        # Build the ordered list of transforms once at construction.
        self._transforms: list[Callable[[np.ndarray], np.ndarray]] = [
            lambda img: _resize(img, self._image_size),
            _to_float,
            lambda img: _normalize(img, self._mean, self._std),
            _hwc_to_chw,
        ]

        logger.debug(
            "PreprocessingPipeline ready — size=%d, mean=%s, std=%s",
            image_size,
            self._mean,
            self._std,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(image: np.ndarray) -> None:
        """
        Raise informative errors for malformed inputs.

        Args:
            image: Expected HxWx3 uint8 array.

        Raises:
            TypeError:  If not a numpy array.
            ValueError: If shape or dtype is wrong.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(image).__name__}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected HxWx3 array, got shape {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 input (0–255), got dtype {image.dtype}. "
                "Convert to uint8 before calling the pipeline."
            )
        if image.size == 0:
            raise ValueError("Received an empty image (zero pixels).")

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing pipeline on a single face crop.

        Args:
            image: HxWx3 uint8 RGB face crop.

        Returns:
            (3, image_size, image_size) float32 array, normalised.

        Raises:
            TypeError / ValueError: on invalid input (see `_validate`).
        """
        self._validate(image)
        result = image
        for transform in self._transforms:
            result = transform(result)
        return result

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def process_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Process a list of face crops into a stacked batch array.

        Args:
            images: List of HxWx3 uint8 arrays.

        Returns:
            (N, 3, image_size, image_size) float32 array.

        Raises:
            ValueError: If the list is empty.
        """
        if not images:
            raise ValueError("process_batch received an empty list.")
        processed = [self(img) for img in images]
        return np.stack(processed, axis=0)

    # ------------------------------------------------------------------
    # Inverse (for visualization / debugging)
    # ------------------------------------------------------------------

    def inverse_normalize(self, tensor: np.ndarray) -> np.ndarray:
        """
        Reverse normalisation for visualization purposes.

        Args:
            tensor: (3, H, W) or (H, W, 3) float32 normalised array.

        Returns:
            (H, W, 3) uint8 array in [0, 255].
        """
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            img = np.transpose(tensor, (1, 2, 0))
        else:
            img = tensor.copy()

        mean = np.array(self._mean, dtype=np.float32)
        std  = np.array(self._std,  dtype=np.float32)
        img  = img * std + mean
        img  = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def __repr__(self) -> str:
        return (
            f"PreprocessingPipeline("
            f"image_size={self._image_size}, "
            f"mean={self._mean}, "
            f"std={self._std})"
        )