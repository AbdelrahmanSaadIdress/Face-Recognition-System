import argparse
import sys
import logging

import torch


from src.config import Config, load_config
from src.data.lfw_dataset import LFWPairsDataset
from src.tracking import ExperimentTracker
from src.evaluation.evaluator import LFWEvaluator



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

parser = argparse.ArgumentParser(
    description="Evaluate a face verification model on the LFW dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# ── Required ────────────────────────────────────────────────────────
parser.add_argument(
    "--model-name",
    required=True,
    help="Human-readable model identifier used in logs and W&B keys.",
)

# ── Data ────────────────────────────────────────────────────────────
parser.add_argument(
    "--lfw-dir",
    default="data/lfw",
    help="Root directory of the LFW dataset.",
)
parser.add_argument(
    "--pairs-file",
    default=None,
    help="Path to the LFW pairs.txt file. Uses dataset default if omitted.",
)

# ── Checkpoint ──────────────────────────────────────────────────────
parser.add_argument(
    "--checkpoint",
    default=None,
    metavar="PATH",
    help="Path to a .pt checkpoint (full Trainer or backbone-only state dict).",
)

# ── Config ──────────────────────────────────────────────────────────
parser.add_argument(
    "--config",
    default="configs/default.yaml",
    metavar="PATH",
    help="Path to the YAML config file.",
)

# ── Inference ───────────────────────────────────────────────────────
parser.add_argument(
    "--device",
    default=None,
    choices=["cuda", "mps", "cpu"],
    help="Inference device. Auto-detected (CUDA > MPS > CPU) if omitted.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=None,
    help="Operating threshold override. Uses cfg.evaluation.recognition_threshold if omitted.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size override. Uses cfg.evaluation.batch_size if omitted.",
)

# ── Tracking ────────────────────────────────────────────────────────
parser.add_argument(
    "--no-tracking",
    action="store_true",
    help="Disable W&B logging.",
)

args = parser.parse_args()

# ── Build config ────────────────────────────────────────────────────
cfg = load_config(args.config)

if args.threshold is not None:
    cfg.evaluation.recognition_threshold = args.threshold
if args.batch_size is not None:
    cfg.evaluation.batch_size = args.batch_size

# ── Build dataset ───────────────────────────────────────────────────
if args.lfw_dir:
    cfg.data.lfw_faces_dir = args.lfw_dir          # override faces directory
if args.pairs_file:
    cfg.data.lfw_pairs_csv = args.pairs_file        # override pairs CSV path

dataset = LFWPairsDataset(
    data_cfg=cfg.data,
    preproc_cfg=cfg.preprocessing,
)
# ── Build model ─────────────────────────────────────────────────────
from src.models import build_face_model  # noqa: E402

model = build_face_model(cfg)

# ── Device ──────────────────────────────────────────────────────────
device = torch.device(args.device) if args.device else None

# ── Tracker ─────────────────────────────────────────────────────────
if args.no_tracking:
    cfg.wandb.enabled = False

tracker = ExperimentTracker(cfg) 

# ── Evaluate ────────────────────────────────────────────────────────
evaluator = LFWEvaluator(dataset, cfg, tracker, device=device)

result = evaluator.evaluate(
    model,
    model_name      = args.model_name,
    checkpoint_path = args.checkpoint,
)

# ── Print summary ───────────────────────────────────────────────────
print("\n── LFW Evaluation Results ──────────────────────────")
print(f"  Model      : {result.model_name}")
print(f"  Pairs      : {result.num_pairs}")
print(f"  EER        : {result.eer:.4f}")
print(f"  EER thresh : {result.eer_threshold:.4f}")
print(f"  AUC        : {result.auc:.4f}")
print(f"  FAR        : {result.far:.4f}")
print(f"  FRR        : {result.frr:.4f}")
print(f"  F1         : {result.f1:.4f}")
print(f"  Embed time : {result.embed_time_s:.1f}s")
print("────────────────────────────────────────────────────\n")

sys.exit(0)




# # Minimal
# python -m src.evaluation.evaluator \
#     --model-name arcface_resnet50 \
#     --checkpoint checkpoints/sphereface_resnet50_with_t_0.5080.pt

# # With overrides
# python -m eval.py \
#     --model-name arcface_resnet50 \
#     --checkpoint checkpoints/sphereface_resnet50_with_t_0.5080.pt \
#     --lfw-dir data/raw/Faces \
#     --config configs/base.yaml \
#     --threshold 0.3481 \
#     --batch-size 128 \
#     --device cuda \
#     --no-tracking