"""Model artifact loading and the prediction service used by the backend.

The saved ``.keras`` file is self-contained: it receives RGB images resized to
``img_size`` and scaled to ``[0, 255]``.  That is exactly what
:func:`pneumonia.preprocessing.preprocess_bytes` returns, so training and
serving share one pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

from . import config
from .preprocessing import preprocess_bytes, to_batch


# --------------------------------------------------------------------------
# Metadata helpers
# --------------------------------------------------------------------------
def default_metadata() -> Dict[str, Any]:
    return {
        "model_name": "pneumonia_classifier",
        "backbone": config.BACKBONE,
        "img_size": list(config.IMG_SIZE),
        "class_names": list(config.CLASS_NAMES),
        "positive_class": config.POSITIVE_CLASS,
        "threshold": config.DEFAULT_THRESHOLD,
        "input_range": [0.0, 255.0],
        "input_color": "RGB",
        "tta": config.TTA_ENABLED,
        "tta_views": config.TTA_VIEWS,
        "created_at": None,
    }


def save_metadata(metadata: Dict[str, Any], path: Path = config.DEFAULT_METADATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def load_metadata(path: Path = config.DEFAULT_METADATA_PATH) -> Dict[str, Any]:
    path = Path(path)
    metadata = default_metadata()
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            if isinstance(stored, dict):
                metadata.update(stored)
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
            print(f"[inference] WARNING: could not read metadata ({exc}); using defaults.")
    return metadata


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Test-time augmentation
# --------------------------------------------------------------------------
def _central_zoom(batch: np.ndarray, scale: float) -> np.ndarray:
    """Centre-crop ``scale`` of the image and resize it back to the original size."""
    height, width = int(batch.shape[1]), int(batch.shape[2])
    crop_h, crop_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    crop = batch[:, top : top + crop_h, left : left + crop_w, :]
    resized = tf.image.resize(crop, (height, width), method="bilinear", antialias=True)
    return resized.numpy().astype(np.float32, copy=False)


def tta_views(batch: np.ndarray, n_views: int = config.TTA_VIEWS) -> List[np.ndarray]:
    """Deterministic views of the same batch, averaged at prediction time.

    The views mirror the training augmentation (horizontal flip and a mild
    zoom) so the model only sees in-distribution variations. At most four
    distinct views exist; the returned list is shorter if fewer are requested.
    """
    views: List[np.ndarray] = [batch]
    if n_views <= 1:
        return views

    views.append(np.ascontiguousarray(batch[:, :, ::-1, :]))
    if n_views <= 2:
        return views

    zoomed = _central_zoom(batch, 0.90)
    views.append(zoomed)
    if n_views <= 3:
        return views

    views.append(np.ascontiguousarray(zoomed[:, :, ::-1, :]))
    return views


# --------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------
class ModelNotAvailableError(RuntimeError):
    """Raised when the trained artifact is missing or cannot be loaded."""


class PneumoniaPredictor:
    """Thread-safe, lazy-loading wrapper around the trained classifier."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        load: bool = True,
    ):
        self.model_path = Path(model_path or config.DEFAULT_MODEL_PATH)
        self.metadata_path = Path(metadata_path or config.DEFAULT_METADATA_PATH)
        self.metadata = load_metadata(self.metadata_path)
        self.threshold = float(self.metadata.get("threshold", config.DEFAULT_THRESHOLD))
        self.img_size = tuple(self.metadata.get("img_size", config.IMG_SIZE))
        self.backbone = str(self.metadata.get("backbone", config.BACKBONE))
        self.class_names = list(self.metadata.get("class_names", config.CLASS_NAMES))
        self.tta = bool(self.metadata.get("tta", config.TTA_ENABLED))
        self.tta_views = int(self.metadata.get("tta_views", config.TTA_VIEWS))
        self._model: Optional[tf.keras.Model] = None
        self._load_error: Optional[str] = None
        if load:
            self.load()

    # -- lifecycle ------------------------------------------------------
    def load(self) -> "PneumoniaPredictor":
        if not self.model_path.is_file():
            self._load_error = (
                f"Trained model not found at '{self.model_path}'. "
                "Run 'python -m pneumonia.train' first."
            )
            self._model = None
            return self
        try:
            # Importing the module registers the custom layers used by the graph.
            from . import model as _model_module  # noqa: F401

            self._model = tf.keras.models.load_model(self.model_path)
            self._load_error = None
        except Exception as exc:  # pragma: no cover - depends on artifact
            self._model = None
            self._load_error = f"Failed to load model: {exc}"
        return self

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def require_ready(self) -> tf.keras.Model:
        if self._model is None:
            raise ModelNotAvailableError(self._load_error or "Model is not available.")
        return self._model

    # -- inference ------------------------------------------------------
    def probabilities(self, images: np.ndarray, tta: Optional[bool] = None) -> np.ndarray:
        """Raw P(PNEUMONIA) for a ``(N, H, W, 3)`` canonical batch.

        With test-time augmentation enabled the probability is averaged over a
        few deterministic views of each image.
        """
        model = self.require_ready()
        batch = np.asarray(images, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[np.newaxis, ...]
        if batch.ndim != 4:
            raise ValueError(f"Expected a 3D or 4D array, received shape {batch.shape}.")

        use_tta = self.tta if tta is None else bool(tta)
        if not use_tta:
            # Direct call avoids the tf.data / predict() overhead for small batches.
            return model(batch, training=False).numpy().reshape(-1)

        views = tta_views(batch, self.tta_views)
        total = None
        for view in views:
            probabilities = model(view, training=False).numpy().reshape(-1)
            total = probabilities if total is None else total + probabilities
        return total / float(len(views))

    @property
    def n_tta_views(self) -> int:
        """Number of forward passes used per prediction (1 when TTA is off)."""
        return max(1, min(self.tta_views, 4)) if self.tta else 1

    def predict_batch(self, canonical: np.ndarray) -> List[Dict[str, Any]]:
        """Run the network on a batch of canonical ``(N, H, W, 3)`` float32 images."""
        started = time.perf_counter()
        probabilities = self.probabilities(canonical)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        results: List[Dict[str, Any]] = []
        for probability in probabilities:
            pneumonia_probability = float(probability)
            normal_probability = 1.0 - pneumonia_probability
            is_pneumonia = pneumonia_probability >= self.threshold
            results.append(
                {
                    "label": self.class_names[1] if is_pneumonia else self.class_names[0],
                    "prediction": "PNEUMONIA" if is_pneumonia else "NORMAL",
                    "is_pneumonia": bool(is_pneumonia),
                    "confidence": (
                        pneumonia_probability if is_pneumonia else normal_probability
                    ),
                    "probability": pneumonia_probability,
                    "probabilities": {
                        self.class_names[0]: normal_probability,
                        self.class_names[1]: pneumonia_probability,
                    },
                    "threshold": self.threshold,
                    "tta_views": self.n_tta_views,
                    "inference_ms": round(elapsed_ms, 2),
                }
            )
        return results

    def predict_array(self, canonical: np.ndarray) -> Dict[str, Any]:
        """Predict a single canonical ``(H, W, 3)`` float32 image."""
        return self.predict_batch(to_batch(canonical))[0]

    def predict_bytes(self, data: bytes) -> Dict[str, Any]:
        """Full backend path: raw upload bytes -> prediction dictionary."""
        canonical = preprocess_bytes(data, self.img_size)
        result = self.predict_array(canonical)
        result["image_size"] = [int(canonical.shape[1]), int(canonical.shape[0])]
        result["model"] = self.info()
        return result

    # -- introspection --------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name", "pneumonia_classifier"),
            "backbone": self.backbone,
            "img_size": list(self.img_size),
            "input_range": self.metadata.get("input_range", [0.0, 255.0]),
            "class_names": self.class_names,
            "positive_class": self.metadata.get("positive_class", config.POSITIVE_CLASS),
            "threshold": self.threshold,
            "tta": self.tta,
            "tta_views": self.n_tta_views,
            "created_at": self.metadata.get("created_at"),
            "model_path": str(self.model_path),
        }

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready,
            "error": self._load_error,
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "backbone": self.backbone,
            "threshold": self.threshold,
            "img_size": list(self.img_size),
        }
