"""
realtime_attendance.py
======================
Real-time face recognition attendance system.

Pipeline (per frame):
    1. Capture frame from webcam via OpenCV.
    2. Detect faces using MTCNN (matches base.yaml detector setting).
    3. For each detected face crop:
        a. BGR → RGB → PreprocessingPipeline (same as training) → tensor.
        b. FaceModel backbone → 512-d L2-normalised embedding.
        c. Query ChromaDB for nearest-neighbour (cosine similarity).
        d. If similarity ≥ threshold:
                - Extract mongo_id from result metadata.
                - Look up student name in MongoDB.
                - Draw name + score on frame.
        e. Otherwise label face as "Unknown".
    4. Display annotated frame live.  Press 'q' to quit.

Preprocessing uses your project's own PreprocessingPipeline
(src/data/preprocessing.py) — the same pipeline used during training.

Usage:
    python realtime_attendance.py \
        --checkpoint checkpoints/sphereface_resnet50_with_t_0.5080.pt \
        --config     configs/base.yaml

    # Use a video file instead of webcam:
    python realtime_attendance.py --checkpoint ... --source path/to/video.mp4

Requirements:
    pip install pymongo chromadb opencv-python torch torchvision facenet-pytorch
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from bson import ObjectId
from pymongo import MongoClient
import chromadb

# ── project imports ──────────────────────────────────────────────────────────
from src.config import load_config
from src.models.face_model import build_face_model
from src.data.preprocessing import PreprocessingPipeline   # ← your training pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("realtime_attendance")

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
# MONGO_URI         = "mongodb://localhost:27017"
MONGO_URI         = "mongodb+srv://abdo:abdo@cluster0.7v7faph.mongodb.net/?appName=Cluster0"
MONGO_DB          = "attendance"
MONGO_COL         = "identities"
CHROMA_PATH       = "./chroma_db"
CHROMA_COLL       = "face_embeddings"
COOLDOWN_SECONDS  = 5     # min seconds between printing the same name to stdout


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
def load_model(checkpoint_path: str | None, cfg, device: torch.device) -> torch.nn.Module:
    model = build_face_model(cfg, num_classes=None)   # backbone only — inference mode
    model.to(device)

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt       = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)

        backbone_state = {
            k.replace("backbone.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        if backbone_state:
            model.backbone.load_state_dict(backbone_state, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        log.info("Loaded weights from %s", ckpt_path.name)
    else:
        log.warning("No checkpoint — using ImageNet-pretrained backbone.")

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

    Conversion chain (identical to build_database.py and training):
      BGR uint8  →  RGB uint8          (pipeline expects RGB)
      RGB uint8  →  CxHxW float32      (PreprocessingPipeline: resize→float→norm→HWC→CHW)
      CxHxW     →  (1,C,H,W) tensor   (add batch dim, send to device)
      tensor    →  (1, 512) embedding (FaceModel backbone forward)
      squeeze   →  (512,) numpy
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)       # HxWx3 uint8 RGB
    chw     = pipeline(img_rgb)                               # (3, H, W) float32
    tensor  = torch.from_numpy(chw).unsqueeze(0).to(device)  # (1, 3, H, W)
    emb     = model(tensor)                                   # (1, embedding_dim)
    return emb.squeeze().cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────
# Face detector  (MTCNN — matches base.yaml)
# ─────────────────────────────────────────────
def build_detector(device: torch.device):
    """MTCNN from facenet-pytorch. Falls back to Haar cascade if not installed."""
    try:
        from facenet_pytorch import MTCNN
        detector = MTCNN(
            keep_all       = True,
            device         = device,
            post_process   = False,   # we want raw pixel crops
            select_largest = False,
        )
        log.info("MTCNN detector ready on %s", device)
        return detector
    except ImportError:
        log.warning("facenet-pytorch not installed — using Haar cascade fallback.")
        return None


def detect_and_crop(
    frame_bgr: np.ndarray,
    detector,
    image_size: int,
) -> list[tuple[np.ndarray, tuple]]:
    """
    Detect faces and return list of (crop_bgr, (x1, y1, x2, y2)).
    Works with both MTCNN and the Haar-cascade fallback.
    """
    results: list[tuple[np.ndarray, tuple]] = []
    h, w = frame_bgr.shape[:2]

    if detector is not None:
        # ── MTCNN path ────────────────────────────────────────────────────
        frame_rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, _       = detector.detect(frame_rgb)   # (N,4) float or None
        if boxes is None:
            return results
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append((frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)))
    else:
        # ── Haar cascade fallback ─────────────────────────────────────────
        gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, bw, bh) in (faces if len(faces) else []):
            results.append((frame_bgr[y:y+bh, x:x+bw], (x, y, x+bw, y+bh)))

    return results


# ─────────────────────────────────────────────
# ChromaDB query
# ─────────────────────────────────────────────
def query_chroma(chroma_col, embedding: np.ndarray, n_results: int = 1):
    """
    Return (mongo_id, similarity) for the best match.

    ChromaDB cosine distance ∈ [0, 2] → similarity = 1 − dist/2 ∈ [0, 1].
    Returns (None, None) if no results.
    """
    result = chroma_col.query(
        query_embeddings = [embedding.tolist()],
        n_results        = n_results,
        include          = ["metadatas", "distances"],
    )
    if not result["ids"] or not result["ids"][0]:
        return None, None

    distance   = result["distances"][0][0]
    similarity = 1.0 - (distance / 2.0)
    mongo_id   = result["metadatas"][0][0].get("mongo_id")
    return mongo_id, similarity


# ─────────────────────────────────────────────
# MongoDB lookup
# ─────────────────────────────────────────────
def lookup_name(collection, mongo_id: str) -> str | None:
    try:
        doc = collection.find_one({"_id": ObjectId(mongo_id)})
        return doc["name"] if doc else None
    except Exception as e:
        log.error("MongoDB lookup failed for id=%s: %s", mongo_id, e)
        return None


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 200)
WHITE  = (255, 255, 255)
FONT   = cv2.FONT_HERSHEY_SIMPLEX


def draw_result(
    frame:      np.ndarray,
    box:        tuple,
    label:      str,
    similarity: float | None,
    known:      bool,
) -> None:
    x1, y1, x2, y2 = box
    color = GREEN if known else RED
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = label if similarity is None else f"{label}  {similarity:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4), FONT, 0.6, WHITE, 2)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), FONT, 0.7, YELLOW, 2)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:

    # ── Config ──────────────────────────────────────────────────────────
    cfg    = load_config(args.config)
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    # ── Preprocessing pipeline — identical to training & build_database ──
    # Input contract: RGB uint8 HxWx3 → output: CxHxW float32 normalised
    # BGR→RGB conversion is done inside get_embedding before calling this.
    pipeline = PreprocessingPipeline(
        preproc_cfg     = cfg.preprocessing,
        image_size      = cfg.data.image_size,
        apply_detection = False,
    )
    log.info("Preprocessing pipeline: %s", pipeline)

    # ── Model ────────────────────────────────────────────────────────────
    model    = load_model(args.checkpoint, cfg, device)
    log.info("Model ready — embedding_dim=%d", model.embedding_dim)

    # ── Detector ─────────────────────────────────────────────────────────
    detector = build_detector(device)

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
    log.info("ChromaDB collection: %s", args.chroma_coll)

    # ── Camera / video source ─────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source!r}")

    # Similarity threshold: CLI override → config value → safe default
    threshold = (
        args.threshold
        if args.threshold is not None
        else cfg.evaluation.recognition_threshold
    )
    log.info("Similarity threshold: %.2f", threshold)
    log.info("Press 'q' to quit.")

    last_seen: dict[str, float] = {}   # mongo_id → last attendance timestamp
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            log.info("Stream ended.")
            break

        # ── FPS ───────────────────────────────────────────────────────────
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # # ── Detect faces ──────────────────────────────────────────────────
        face_crops = detect_and_crop(frame, detector, cfg.data.image_size)

        for crop_bgr, box in face_crops:
            if crop_bgr.size == 0:
                continue

            # ── Embed ─────────────────────────────────────────────────────
            embedding = get_embedding(model, crop_bgr, pipeline, device)

            # ── Query ChromaDB ────────────────────────────────────────────
            mongo_id, similarity = query_chroma(chroma_col, embedding)

            # ── Resolve identity ──────────────────────────────────────────
            known = False
            label = "Unknown"

            if mongo_id and similarity is not None and similarity >= threshold:
                name = lookup_name(collection, mongo_id)
                if name:
                    label = name
                    known = True

                    # Attendance log with per-person cooldown
                    if time.time() - last_seen.get(mongo_id, 0) >= COOLDOWN_SECONDS:
                        last_seen[mongo_id] = time.time()
                        print(
                            f"[ATTENDANCE]  {name:<30}  "
                            f"similarity={similarity:.3f}"
                        )

            # ── Annotate frame ────────────────────────────────────────────
            draw_result(frame, box, label, similarity if known else None, known)

        draw_fps(frame, fps)
        cv2.imshow("Attendance System ", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("User quit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    mongo_client.close()
    log.info("All connections closed.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time face recognition attendance system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  default=None, help="Path to trained .pt checkpoint.")
    p.add_argument("--config",      default="configs/base.yaml")
    p.add_argument("--source",      default="1", help="Webcam index (0, 1 …) or path to a video file.")
    p.add_argument("--threshold",   type=float, default=None, help="Similarity threshold override (default: from config).")
    p.add_argument("--device",      default=None, choices=["cuda", "cpu", "mps"])
    p.add_argument("--mongo-uri",   default=MONGO_URI,   dest="mongo_uri")
    p.add_argument("--mongo-db",    default=MONGO_DB,    dest="mongo_db")
    p.add_argument("--mongo-col",   default=MONGO_COL,   dest="mongo_col")
    p.add_argument("--chroma-path", default=CHROMA_PATH, dest="chroma_path")
    p.add_argument("--chroma-coll", default=CHROMA_COLL, dest="chroma_coll")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())