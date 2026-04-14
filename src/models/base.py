"""
FaceModel — the core interface every model implementation must honour.

Rules:
    - No implementation logic here — pure contract.
    - Every method that subclasses override is @abstractmethod.
    - `preprocess` is the only concrete method: it chains detect → align
        so subclasses cannot break the pipeline order.
    - Type annotations are strict; numpy dtypes are documented in docstrings.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FaceModel(ABC):
    """
    Abstract base for all face recognition models.

    Lifecycle
    ---------
    1. Instantiate with a config dict (populated from Config).
    2. Call `load()` to initialise weights / move to device.
    3. Use `preprocess()` for inference-time image preparation.
    4. Call `get_embedding()` on the prepared face crop.
    5. During training call `training_step()` on each batch.

    All image arrays are HxWxC uint8 or float32 in [0, 1] / [-1, 1]
    depending on the pipeline stage — each method documents its contract.
    """

    def __init__(self, model_config: dict[str, Any]) -> None:
        """
        Args:
            model_config:   The `model` sub-dict from Config, plus
                            the relevant loss sub-dict passed at construction.
                            Subclasses must call super().__init__(model_config).
        """
        self._config = model_config
        self._is_loaded: bool = False
        logger.debug("%s initialised (not yet loaded)", self.__class__.__name__)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and move tensors to the target device.

        Must set `self._is_loaded = True` on success.

        Raises:
            RuntimeError: If weights cannot be found or loaded.
        """
        ...

    def _assert_loaded(self) -> None:
        """Guard: raise if `load()` has not been called."""
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__}.load() must be called before "
                "using the model."
            )

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------

    @abstractmethod
    def detect_face(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and crop the primary face in an image.

        Args:
            image: HxWx3 uint8 BGR or RGB image.

        Returns:
            Cropped face region as HxWx3 uint8 array.

        Raises:
            ValueError: If no face is detected.
        """
        ...

    @abstractmethod
    def align_face(self, image: np.ndarray) -> np.ndarray:
        """
        Align a detected face crop using facial landmarks.

        Args:
            image: HxWx3 uint8 face crop (output of `detect_face`).

        Returns:
            Aligned, resized face as (image_size x image_size x 3) uint8.
        """
        ...

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline: detect → align.

        This is the ONLY concrete method in the base class.
        Do not override it — override `detect_face` and `align_face`.

        Args:
            image: Raw HxWx3 uint8 input image.

        Returns:
            Aligned face ready for `get_embedding`.
        """
        self._assert_loaded()
        face = self.detect_face(image)
        return self.align_face(face)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Compute the L2-normalised embedding vector for a preprocessed face.

        Args:
            face: (image_size x image_size x 3) float32 array,
                  values normalised to model-specific range.

        Returns:
            1-D float32 array of shape (embedding_dim,), L2-normalised.
        """
        ...

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_loss(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute the training loss for a batch.

        Args:
            embeddings: (batch_size, embedding_dim) float32 array.
            labels:     (batch_size,) int64 identity label array.

        Returns:
            Scalar loss value (Python float).
        """
        ...

    @abstractmethod
    def training_step(self, batch: dict[str, Any]) -> float:
        """
        Execute one training iteration: forward + loss + backward + step.

        Args:
            batch: Dict containing at minimum:
                   - "images":  (B, C, H, W) float32 tensor or ndarray
                   - "labels":  (B,) int64 tensor or ndarray

        Returns:
            Scalar loss for this step (Python float, detached from graph).
        """
        ...

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"backbone={self._config.get('backbone', 'unknown')}, "
            f"embedding_dim={self._config.get('embedding_dim', 'unknown')}, "
            f"status={status})"
        )