"""
build_database.py
=================
Attendance system — database population pipeline.

Image split per identity:
    - 10 images  →  embeddings  →  stored in ChromaDB   (the "gallery")
    -  5 images  →  saved to disk as test probes        (held-out, never in ChromaDB)

After all identities are registered the script runs a Top-K accuracy
evaluation: each probe image is embedded, queried against ChromaDB, and we
check whether the correct identity appears in the top-1 / top-3 / top-5
results.  Results are printed and saved to eval_results.json.

Preprocessing uses your project's own PreprocessingPipeline
(src/data/preprocessing.py) — the exact same pipeline used during training —
so embeddings are produced under identical conditions.

Usage:
    python build_database.py \
        --checkpoint checkpoints/arcface_resnet50_latest.pt \
        --dataset    data/raw/Identities \
        --config     configs/base.yaml

"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from pymongo import MongoClient
import chromadb

# ── project imports ──────────────────────────────────────────────────────────
from src.config import load_config
from src.models.face_model import build_face_model
from src.data.preprocessing import PreprocessingPipeline 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_database")

# ─────────────────────────────────────────────
# DEFAULTS  (all overridable via CLI)
# ─────────────────────────────────────────────
MAX_IDENTITIES    = 100
GALLERY_SIZE      = 10    # images per identity  →  stored in ChromaDB
PROBE_SIZE        = 5     # images per identity  →  held-out for evaluation
TEST_IMG_DIR      = "test_images"

MONGO_URI         = "mongodb+srv://abdo:abdo@cluster0.7v7faph.mongodb.net/?appName=Cluster0"
MONGO_DB          = "attendance"
MONGO_COL         = "identities"

CHROMA_PATH       = "./chroma_db"
CHROMA_COLL       = "face_embeddings"

VALID_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EVAL_RESULTS_FILE = "eval_results.json"


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
def load_model(checkpoint_path: str | None, cfg, device: torch.device) -> torch.nn.Module:
    """
    Build FaceModel (backbone only, no loss head) and load checkpoint weights.
    Returns the model in eval mode.
    """
    model = build_face_model(cfg, num_classes=None)   # inference mode — no loss head
    model.to(device)

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt       = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)   # support full Trainer checkpoints

        # If saved as full FaceModel, keys start with "backbone."
        backbone_state = {
            k.replace("backbone.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        if backbone_state:
            model.backbone.load_state_dict(backbone_state, strict=False)
            log.info("Loaded backbone weights from %s", ckpt_path.name)
        else:
            model.load_state_dict(state_dict, strict=False)
            log.info("Loaded full model weights from %s", ckpt_path.name)
    else:
        log.warning("No checkpoint — using ImageNet-pretrained backbone only.")

    model.eval()
    return model


# ─────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────
@torch.no_grad()
def get_embedding(
    model:    torch.nn.Module,
    img_bgr:  np.ndarray,              # OpenCV BGR uint8 HxWx3
    pipeline: PreprocessingPipeline,
    device:   torch.device,
) -> np.ndarray:
    """
    BGR image  →  1-D float32 L2-normalised embedding.

    Conversion chain (matches training exactly):
        BGR uint8  →  RGB uint8           (OpenCV loads BGR, pipeline expects RGB)
        RGB uint8  →  CxHxW float32       (PreprocessingPipeline: resize→float→norm→transpose)
        CxHxW     →  (1,C,H,W) tensor    (add batch dim)
        tensor    →  (1, 512) embedding  (FaceModel backbone)
        squeeze   →  (512,) numpy
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # HxWx3 uint8 RGB
    chw     = pipeline(img_rgb)                           # (3, H, W) float32
    tensor  = torch.from_numpy(chw).unsqueeze(0).to(device)  # (1, 3, H, W)
    emb     = model(tensor)                               # (1, embedding_dim)
    return emb.squeeze().cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def collect_images(identity_dir: Path) -> list[Path]:
    imgs = [p for p in identity_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS]
    random.shuffle(imgs)
    return imgs


# ─────────────────────────────────────────────
# Top-K evaluation
# ─────────────────────────────────────────────
def evaluate_topk(
    model:       torch.nn.Module,
    pipeline:    PreprocessingPipeline,
    device:      torch.device,
    chroma_col,
    test_probes: list[dict],
    ks:          list[int] = [1, 3, 5],
) -> dict:
    """
    Query each held-out probe image against ChromaDB and compute Top-K accuracy.

    A probe is a "hit at K" if the correct identity's mongo_id appears
    anywhere in the top-K results returned by ChromaDB.

    ChromaDB returns cosine *distance* ∈ [0, 2].
    Similarity = 1 − dist/2  →  1.0 = identical, 0.0 = orthogonal.
    """
    max_k  = max(ks)
    hits   = {k: 0 for k in ks}
    total  = 0
    failed = 0
    per_identity: dict[str, dict] = {}

    log.info("─" * 55)
    log.info("Evaluating %d probe images (Top-%d) …", len(test_probes), max_k)

    for probe in test_probes:
        identity_name = probe["identity_name"]
        true_mongo_id = probe["mongo_id"]
        img_path      = probe["path"]

        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("  Cannot read probe %s — skip", img_path.name)
            failed += 1
            continue

        emb    = get_embedding(model, img, pipeline, device)
        result = chroma_col.query(
            query_embeddings = [emb.tolist()],
            n_results        = max_k,
            include          = ["metadatas", "distances"],
        )
        returned_ids = [m.get("mongo_id") for m in result["metadatas"][0]]

        # per-identity bookkeeping
        if identity_name not in per_identity:
            per_identity[identity_name] = {f"top{k}": 0 for k in ks}
            per_identity[identity_name]["total"] = 0
        per_identity[identity_name]["total"] += 1
        total += 1

        for k in ks:
            if true_mongo_id in returned_ids[:k]:
                hits[k] += 1
                per_identity[identity_name][f"top{k}"] += 1

    if total == 0:
        log.error("No probes evaluated successfully.")
        return {}

    results: dict = {
        "total_probes":  total,
        "failed_probes": failed,
    }
    for k in ks:
        results[f"top{k}_hits"]     = hits[k]
        results[f"top{k}_accuracy"] = round(hits[k] / total, 4)

    results["per_identity"] = [
        {
            "identity": name,
            "total":    stats["total"],
            **{
                f"top{k}_acc": round(stats[f"top{k}"] / max(stats["total"], 1), 4)
                for k in ks
            },
        }
        for name, stats in per_identity.items()
    ]
    return results


def print_eval_results(results: dict, ks: list[int] = [1, 3, 5]) -> None:
    log.info("═" * 55)
    log.info("Top-K Accuracy Results")
    log.info("  Total probes : %d", results["total_probes"])
    log.info("  Failed reads : %d", results["failed_probes"])
    for k in ks:
        log.info(
            "  Top-%d        : %d / %d  (%.2f%%)",
            k,
            results[f"top{k}_hits"],
            results["total_probes"],
            results[f"top{k}_accuracy"] * 100,
        )
    log.info("═" * 55)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root.resolve()}")

    # ── Config ──────────────────────────────────────────────────────────
    cfg    = load_config(args.config)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    # ── Preprocessing pipeline — identical to training ────────────────────
    # Input contract: RGB uint8 HxWx3  →  output: CxHxW float32 normalised
    # (we handle BGR→RGB conversion in get_embedding before calling this)
    pipeline = PreprocessingPipeline(
        preproc_cfg     = cfg.preprocessing,   # carries mean, std, detector choice
        image_size      = cfg.data.image_size, # 112 px (from base.yaml)
        apply_detection = False,               # images are already face crops
    )
    log.info("Preprocessing pipeline: %s", pipeline)

    # ── Model ────────────────────────────────────────────────────────────
    log.info("Loading model …")
    model = load_model(args.checkpoint, cfg, device)
    log.info("Model ready — embedding_dim=%d", model.embedding_dim)

    # ── MongoDB ──────────────────────────────────────────────────────────
    mongo_client = MongoClient(args.mongo_uri)
    collection   = mongo_client[args.mongo_db][args.mongo_col]
    log.info("MongoDB: %s / %s / %s", args.mongo_uri, args.mongo_db, args.mongo_col)

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    chroma_col    = chroma_client.get_or_create_collection(
        name     = args.chroma_coll,
        metadata = {"hnsw:space": "cosine"},
    )
    log.info("ChromaDB: %s  (path=%s)", args.chroma_coll, args.chroma_path)

    # ── Output directories ───────────────────────────────────────────────
    test_root = Path(args.test_img_dir)
    test_root.mkdir(parents=True, exist_ok=True)

    # ── Identity loop ─────────────────────────────────────────────────────
    identity_dirs    = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    identity_dirs    = identity_dirs[:MAX_IDENTITIES]
    total_registered = 0
    test_probes: list[dict] = []

    for idx, identity_dir in enumerate(identity_dirs, start=1):
        identity_name = identity_dir.name
        all_images    = collect_images(identity_dir)

        # ── Split gallery / probes ────────────────────────────────────────
        if len(all_images) < GALLERY_SIZE:
            log.warning(
                "[%d/%d] SKIP %s — only %d images (need %d for gallery)",
                idx, len(identity_dirs), identity_name,
                len(all_images), GALLERY_SIZE,
            )
            continue

        gallery_imgs = all_images[:GALLERY_SIZE]
        probe_imgs   = (
            all_images[GALLERY_SIZE: GALLERY_SIZE + PROBE_SIZE]
            if len(all_images) >= GALLERY_SIZE + PROBE_SIZE
            else []
        )

        log.info(
            "[%d/%d] %s — gallery=%d, probes=%d",
            idx, len(identity_dirs), identity_name,
            len(gallery_imgs), len(probe_imgs),
        )

        # ── 1. Compute gallery embeddings ─────────────────────────────────
        embeddings: list[np.ndarray] = []
        for img_path in gallery_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("    Cannot read %s — skipping", img_path.name)
                continue
            emb = get_embedding(model, img, pipeline, device)
            embeddings.append(emb)

        if not embeddings:
            log.warning("    SKIP %s — no valid embeddings produced", identity_name)
            continue

        # ── 2. Save identity to MongoDB ───────────────────────────────────
        result   = collection.insert_one({
            "name":          identity_name,
            "gallery_count": len(embeddings),
            "probe_count":   len(probe_imgs),
        })
        mongo_id = str(result.inserted_id)
        log.info("    MongoDB _id: %s", mongo_id)

        # ── 3. Store all 10 gallery embeddings in ChromaDB ────────────────
        chroma_col.add(
            ids        = [f"{mongo_id}_{i}" for i in range(len(embeddings))],
            embeddings = [e.tolist() for e in embeddings],
            metadatas  = [
                {"mongo_id": mongo_id, "identity_name": identity_name}
                for _ in embeddings
            ],
        )
        log.info("    Stored %d gallery embeddings in ChromaDB", len(embeddings))

        # ── 4. Copy probe images + track for evaluation ───────────────────
        if probe_imgs:
            probe_dir = test_root / identity_name
            probe_dir.mkdir(parents=True, exist_ok=True)
            for img_path in probe_imgs:
                dest = probe_dir / img_path.name
                shutil.copy(img_path, dest)
                test_probes.append({
                    "identity_name": identity_name,
                    "mongo_id":      mongo_id,
                    "path":          dest,
                })

        total_registered += 1

    log.info("=" * 55)
    log.info("Registration done — %d / %d identities.", total_registered, len(identity_dirs))
    log.info("Total probe images collected: %d", len(test_probes))

    # ── Top-K Evaluation ─────────────────────────────────────────────────
    if test_probes:
        ks      = [1, 3, 5]
        results = evaluate_topk(
            model       = model,
            pipeline    = pipeline,
            device      = device,
            chroma_col  = chroma_col,
            test_probes = test_probes,
            ks          = ks,
        )
        print_eval_results(results, ks)

        out_path = Path(EVAL_RESULTS_FILE)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        log.info("Evaluation results saved → %s", out_path)
    else:
        log.warning("No probe images available — skipping evaluation.")

    mongo_client.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser( description="Populate MongoDB + ChromaDB and evaluate Top-K accuracy.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",   default=None, help="Path to trained .pt checkpoint.")
    p.add_argument("--dataset",      default="data/raw/identities", help="Root folder: <dataset>/<identity_name>/<images>.")
    p.add_argument("--config",       default="configs/base.yaml")
    p.add_argument("--device",       default=None, choices=["cuda", "cpu", "mps"])
    p.add_argument("--test-img-dir", default=TEST_IMG_DIR, dest="test_img_dir", help="Folder where the 5 held-out probe images per identity are saved.")
    p.add_argument("--mongo-uri",    default=MONGO_URI,   dest="mongo_uri")
    p.add_argument("--mongo-db",     default=MONGO_DB,    dest="mongo_db")
    p.add_argument("--mongo-col",    default=MONGO_COL,   dest="mongo_col")
    p.add_argument("--chroma-path",  default=CHROMA_PATH, dest="chroma_path")
    p.add_argument("--chroma-coll",  default=CHROMA_COLL, dest="chroma_coll")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())