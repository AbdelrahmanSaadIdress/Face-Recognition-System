"""
LFW (Labeled Faces in the Wild) dataset — verification protocol.

Contract
--------
- Reads the standard pairs CSV (columns: id, name1, img_idx1, name2, img_idx2, label).
- Loads pre-cropped face images from `lfw_faces_dir`; NO detection is performed here.
- Every image goes through `PreprocessingPipeline` (normalize + resize only).
- Returns (img1, img2, label) where label ∈ {0, 1}.
- Used EXCLUSIVELY for evaluation, never for training.

Directory layout expected:
    lfw_faces_dir/
        <identity_name>/
            <identity_name>_0001.jpg
            ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import DataConfig, PreprocessingConfig
from src.data.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


class LFWPairsDataset(Dataset):
    """
    PyTorch Dataset for the LFW face-verification pairs protocol.

    Each item is a tuple: (img1, img2, label)
        img1, img2 : float32 tensors of shape (3, H, W), normalised.
        label      : int — 1 = same identity, 0 = different identity.

    Args:
        data_cfg:    DataConfig with paths and image_size.
        preproc_cfg: PreprocessingConfig with normalisation params.
        split:       Informational only ("test" by default).
    """

    # Expected CSV column names — must match your test.csv header.
    _COL_IMG1  = "Image1"
    _COL_IMG2  = "Image2"
    _COL_LABEL = "class"
    _COL_FACE_PRESENT = "face_present"
    _SIMILAR_LABEL = ("similar", 1)
    _DIFFERENT_LABEL = ("different", 0)

    def __init__(
        self,
        data_cfg: DataConfig,
        preproc_cfg: PreprocessingConfig,
        split: str = "test",
        transform: Optional[object] = None,
    ) -> None:
        self._faces_dir = Path(data_cfg.lfw_faces_dir)
        self._image_size = data_cfg.image_size
        self._split = split
        self._transform = transform

        self._pipeline = PreprocessingPipeline(
            preproc_cfg=preproc_cfg,
            image_size=data_cfg.image_size,
            apply_detection=False,   # crops already provided
        )

        pairs_csv = Path(data_cfg.lfw_pairs_csv)
        self._pairs = self._load_pairs(pairs_csv)
        logger.info(
            "LFWPairsDataset (%s): %d pairs from %s",
            split,
            len(self._pairs),
            pairs_csv,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pairs(self, csv_path: Path) -> pd.DataFrame:
        """Read and validate the pairs CSV."""
        if not csv_path.exists():
            raise FileNotFoundError(f"LFW pairs CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required = {
            self._COL_IMG1,
            self._COL_IMG2,
            self._COL_LABEL,
            self._COL_FACE_PRESENT
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"LFW pairs CSV is missing columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        invalid_labels = df[self._COL_LABEL].unique()
        if not set(invalid_labels).issubset({"similar", "different"}):
            raise ValueError(
                f"Label column must contain only 'similar' or 'different', got: {invalid_labels}"
            )

        return df.reset_index(drop=True)

    def _build_image_path(self, file_name: str) -> Path:
        """
        Reconstruct the standard LFW filename.
        e.g. identity='Aaron_Eckhart', img_idx=1
            → lfw_faces_dir/Aaron_Eckhart/Aaron_Eckhart_0001.jpg
        """
        return self._faces_dir / file_name

    def _load_image(self, file_name: str) -> np.ndarray:
        """Load a face crop as HxWx3 uint8 RGB numpy array."""
        path = self._build_image_path(file_name)
        if not path.exists():
            raise FileNotFoundError(f"LFW face image not found: {path}")
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        row = self._pairs.iloc[idx]
        img1_raw = self._load_image(str(row[self._COL_IMG1]))
        img2_raw = self._load_image(str(row[self._COL_IMG2]))
        label    = int(1 if row[self._COL_LABEL] == self._SIMILAR_LABEL[0] else 0)

        img1 = self._pipeline(img1_raw)
        img2 = self._pipeline(img2_raw)

        if self._transform is not None:
            img1 = self._transform(img1)
            img2 = self._transform(img2)

        return img1, img2, label

    # ------------------------------------------------------------------
    # Helpers for evaluation code
    # ------------------------------------------------------------------

    @property
    def num_pairs(self) -> int:
        """Total number of pairs in the dataset."""
        return len(self._pairs)

    @property
    def num_positive_pairs(self) -> int:
        """Number of same-identity (positive) pairs."""
        return int((self._pairs[self._COL_LABEL] == 1).sum())

    @property
    def num_negative_pairs(self) -> int:
        """Number of different-identity (negative) pairs."""
        return int((self._pairs[self._COL_LABEL] == 0).sum())

    # def get_labels(self) -> np.ndarray:
    #     """Return all pair labels as a (N,) int32 numpy array."""
    #     return self._pairs[self._COL_LABEL].to_numpy(dtype=np.int32)
    def get_labels(self) -> np.ndarray:
        return (self._pairs[self._COL_LABEL] == "similar").to_numpy(dtype=np.int32)

    def __repr__(self) -> str:
        return (
            f"LFWPairsDataset(split={self._split!r}, "
            f"pairs={self.num_pairs}, "
            f"pos={self.num_positive_pairs}, "
            f"neg={self.num_negative_pairs})"
        )