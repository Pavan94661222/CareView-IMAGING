"""Central configuration for the Pneumonia Detection system.

Every value can be overridden with an environment variable so that the same
code base can be used for training (``pneumonia.train``), offline evaluation
(``pneumonia.evaluate``) and the FastAPI backend without touching the source.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = Path(os.getenv("PNEUMONIA_MODELS_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.getenv("PNEUMONIA_DATA_DIR", PROJECT_ROOT / "data"))
REPORTS_DIR = Path(os.getenv("PNEUMONIA_REPORTS_DIR", PROJECT_ROOT / "reports"))

#: Self-contained Keras archive produced by ``pneumonia.train``.
DEFAULT_MODEL_PATH = Path(
    os.getenv("PNEUMONIA_MODEL_PATH", MODELS_DIR / "pneumonia_classifier.keras")
)
#: JSON side-car describing how the model must be fed (backbone, size, threshold).
DEFAULT_METADATA_PATH = Path(
    os.getenv("PNEUMONIA_METADATA_PATH", MODELS_DIR / "model_metadata.json")
)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
KAGGLE_DATASET = os.getenv("PNEUMONIA_KAGGLE_DATASET", "paultimothymooney/chest-xray-pneumonia")

#: Optional pre-existing location of the extracted dataset (skips kagglehub).
LOCAL_DATASET_DIR = os.getenv("PNEUMONIA_DATASET_DIR") or None

#: Class order is fixed and must never change: index 0 -> NORMAL, index 1 -> PNEUMONIA.
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
POSITIVE_CLASS = "PNEUMONIA"

SPLITS = ("train", "val", "test")

#: Hard caps on how many images are taken **per class** from each split.
#: ``0`` disables the cap for that split.  Capping keeps experiments quick and
#: the classes balanced without ever duplicating an image.
MAX_TRAIN_PER_CLASS = int(os.getenv("PNEUMONIA_MAX_TRAIN_PER_CLASS", 800))
MAX_VAL_PER_CLASS = int(os.getenv("PNEUMONIA_MAX_VAL_PER_CLASS", 50))
MAX_TEST_PER_CLASS = int(os.getenv("PNEUMONIA_MAX_TEST_PER_CLASS", 150))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")

# --------------------------------------------------------------------------
# Model / preprocessing
# --------------------------------------------------------------------------
#: Images are resized to this size and scaled to the [0, 255] range.
#: Any backbone specific normalisation lives *inside* the Keras model so the
#: backend can never drift from the training pipeline.
IMG_SIZE = (int(os.getenv("PNEUMONIA_IMG_SIZE", 224)),) * 2

BACKBONE = os.getenv("PNEUMONIA_BACKBONE", "efficientnetv2b0")

#: Decision threshold applied to P(PNEUMONIA). Overwritten by training.
DEFAULT_THRESHOLD = float(os.getenv("PNEUMONIA_THRESHOLD", 0.5))

#: Test-time augmentation: average the model over a few deterministic views of
#: the same image. Costs a couple of extra forward passes, typically worth
#: one to two accuracy points.
TTA_ENABLED = os.getenv("PNEUMONIA_TTA", "1").strip().lower() not in {"0", "false", "no"}

#: Number of averaged views used when TTA is active (1 disables it, max 4).
TTA_VIEWS = int(os.getenv("PNEUMONIA_TTA_VIEWS", 4))

# --------------------------------------------------------------------------
# Backend / upload validation
# --------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("PNEUMONIA_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/x-ms-bmp",
    "image/tiff",
    "application/octet-stream",
}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "MPO", "JPG"}

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "PNEUMONIA_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if o.strip()
]

RANDOM_SEED = int(os.getenv("PNEUMONIA_SEED", 42))
