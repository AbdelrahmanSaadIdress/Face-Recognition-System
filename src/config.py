"""
Typed, validated configuration loader.

Uses only stdlib (dataclasses + yaml) — no pydantic required.
All parameters live in configs/base.yaml.

Usage:
    from src.config import load_config
    cfg = load_config()
    print(cfg.model.embedding_dim)
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    name: str
    version: str
    seed: int = 42


@dataclass
class WandbConfig:
    project: str
    enabled: bool = True
    entity: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: str = ""
    log_freq: int = 50


@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    brightness_jitter: float = 0.3
    contrast_jitter: float = 0.3
    blur_prob: float = 0.2
    blur_kernel_size: int = 5
    occlusion_prob: float = 0.2
    occlusion_max_ratio: float = 0.3


@dataclass
class DataConfig:
    lfw_path: str
    train_path: str
    processed_path: str
    augmented_path: str
    lfw_pairs_csv: str
    lfw_faces_dir: str
    image_size: int = 112
    num_workers: int = 4
    pin_memory: bool = True
    min_identities: int = 100
    min_images_per_identity: int = 5
    max_identities: int = 1000
    max_images_per_identity: int = 100
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class PreprocessingConfig:
    detector: str = "mtcnn"
    alignment_landmarks: int = 5
    normalization_mean: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    normalization_std: list = field(default_factory=lambda: [0.5, 0.5, 0.5])

    def __post_init__(self) -> None:
        valid_detectors = {"mtcnn", "retinaface"}
        if self.detector not in valid_detectors:
            raise ValueError(
                f"preprocessing.detector must be one of {valid_detectors}, "
                f"got {self.detector!r}"
            )
        for attr in ("normalization_mean", "normalization_std"):
            v = getattr(self, attr)
            if len(v) != 3:
                raise ValueError(
                    f"preprocessing.{attr} must have exactly 3 elements, got {len(v)}"
                )


@dataclass
class ModelConfig:
    name: str = "arcface"
    backbone: str = "resnet50"
    embedding_dim: int = 512
    pretrained_backbone: bool = True
    dropout: float = 0.0

    _VALID_NAMES: dataclasses.InitVar = None
    _VALID_BACKBONES: dataclasses.InitVar = None

    def __post_init__(self, _n: None = None, _b: None = None) -> None:
        valid_names = {"softmax", "facenet", "sphereface", "arcface", "triplet"}
        valid_backbones = {"resnet50", "resnet100", "mobilenet_v2"}
        if self.name not in valid_names:
            raise ValueError(
                f"model.name must be one of {valid_names}, got {self.name!r}"
            )
        if self.backbone not in valid_backbones:
            raise ValueError(
                f"model.backbone must be one of {valid_backbones}, got {self.backbone!r}"
            )


@dataclass
class ArcFaceLossConfig:
    margin: float = 0.5
    scale: float = 64.0
    easy_margin: bool = False


@dataclass
class SphereFaceLossConfig:
    margin: int = 4
    scale: float = 64.0


@dataclass
class TripletLossConfig:
    margin: float = 0.3
    mining: str = "semi-hard"
    batch_hard: bool = True

    def __post_init__(self) -> None:
        valid = {"hard", "semi-hard", "random"}
        if self.mining not in valid:
            raise ValueError(
                f"triplet.mining must be one of {valid}, got {self.mining!r}"
            )


@dataclass
class SoftmaxLossConfig:
    label_smoothing: float = 0.0


@dataclass
class LossConfig:
    arcface: ArcFaceLossConfig = field(default_factory=ArcFaceLossConfig)
    sphereface: SphereFaceLossConfig = field(default_factory=SphereFaceLossConfig)
    triplet: TripletLossConfig = field(default_factory=TripletLossConfig)
    softmax: SoftmaxLossConfig = field(default_factory=SoftmaxLossConfig)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    lr_scheduler: str = "cosine"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    lr_milestones: list = field(default_factory=lambda: [30, 60, 90])
    warmup_epochs: int = 5
    gradient_clip: float = 5.0
    amp: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10


@dataclass
class RobustnessConfig:
    blur_sigmas: list = field(default_factory=lambda: [1.0, 2.0, 3.0])
    brightness_deltas: list = field(default_factory=lambda: [-0.3, -0.15, 0.15, 0.3])
    occlusion_ratios: list = field(default_factory=lambda: [0.1, 0.2, 0.3])


@dataclass
class EvaluationConfig:
    batch_size: int = 128
    distance_metric: str = "cosine"
    threshold_min: float = 0.0
    threshold_max: float = 1.0
    threshold_steps: int = 1000
    recognition_threshold: float = 0.6
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    def __post_init__(self) -> None:
        if self.threshold_min >= self.threshold_max:
            raise ValueError(
                f"evaluation.threshold_min ({self.threshold_min}) must be "
                f"< threshold_max ({self.threshold_max})"
            )
        if not (self.threshold_min <= self.recognition_threshold <= self.threshold_max):
            raise ValueError(
                f"recognition_threshold ({self.recognition_threshold}) must be in "
                f"[{self.threshold_min}, {self.threshold_max}]"
            )


@dataclass
class FaissConfig:
    index_type: str = "flat_l2"
    index_dir: str = "data/processed/faiss_indexes"
    nprobe: int = 32
    hnsw_m: int = 32
    top_k: int = 5


@dataclass
class AttendanceConfig:
    database_url: str = "sqlite:///data/processed/attendance.db"
    confidence_threshold: float = 0.7
    cooldown_seconds: int = 30


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2
    max_upload_size_mb: int = 10
    cors_origins: list = field(default_factory=lambda: ["*"])
    log_level: str = "info"


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 10
    timed_runs: int = 100
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    project: ProjectConfig
    wandb: WandbConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    faiss: FaissConfig
    attendance: AttendanceConfig
    api: ApiConfig
    benchmark: BenchmarkConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Maps field names to their concrete dataclass type so _from_dict can recurse.
_NESTED_TYPE_MAP: dict[type, dict[str, type]] = {
    Config: {
        "project": ProjectConfig,
        "wandb": WandbConfig,
        "data": DataConfig,
        "preprocessing": PreprocessingConfig,
        "model": ModelConfig,
        "loss": LossConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "faiss": FaissConfig,
        "attendance": AttendanceConfig,
        "api": ApiConfig,
        "benchmark": BenchmarkConfig,
    },
    DataConfig: {"augmentation": AugmentationConfig},
    LossConfig: {
        "arcface": ArcFaceLossConfig,
        "sphereface": SphereFaceLossConfig,
        "triplet": TripletLossConfig,
        "softmax": SoftmaxLossConfig,
    },
    EvaluationConfig: {"robustness": RobustnessConfig},
}


def _from_dict(cls: type, raw: dict) -> object:
    """
    Recursively construct a dataclass from a (possibly nested) dict.
    Unknown keys are silently ignored so new YAML fields don't break
    older code that hasn't been updated yet.
    """
    nested = _NESTED_TYPE_MAP.get(cls, {})
    kwargs: dict = {}

    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name.startswith("_") or f.name not in raw:
            continue
        value = raw[f.name]
        if f.name in nested and isinstance(value, dict):
            kwargs[f.name] = _from_dict(nested[f.name], value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (non-destructive)."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "base.yaml"


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load and validate the YAML configuration.

    Args:
        config_path: Path to a YAML file.  Defaults to configs/base.yaml.
                     Can also be set via FACE_RECOGNITION_CONFIG env var.

    Returns:
        Fully validated Config instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError:        If any field fails cross-field validation.
    """
    path = Path(
        config_path
        or os.environ.get("FACE_RECOGNITION_CONFIG", _DEFAULT_CONFIG_PATH)
    )

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    return _from_dict(Config, raw)  # type: ignore[return-value]


def override_config(base: Config, overrides: dict) -> Config:
    """
    Return a new Config with selected fields overridden.

    Useful for hyperparameter sweeps without touching YAML files.

    Args:
        base:       Existing validated Config.
        overrides:  Nested dict of fields to override.
                    Example: {"training": {"learning_rate": 0.01}}

    Returns:
        New validated Config with overrides applied.
    """
    merged = _deep_merge(dataclasses.asdict(base), overrides)
    return _from_dict(Config, merged)  # type: ignore[return-value]