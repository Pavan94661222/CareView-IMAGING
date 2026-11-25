"""Prediction orchestration: validate -> preprocess -> infer -> format.

The module contains no HTTP concerns so it can be unit-tested on its own.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from pneumonia.inference import ModelNotAvailableError, PneumoniaPredictor

from .validation import image_metadata, validate_upload

DISCLAIMER = (
    "This result is produced by an automated research prototype and is not a "
    "medical diagnosis. Always consult a qualified radiologist or physician."
)


def severity_from_probability(probability: float) -> str:
    """Coarse, documented banding of P(PNEUMONIA) used for UI colour coding."""
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "moderate"
    return "high"


def run_prediction(
    predictor: PneumoniaPredictor,
    data: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the upload and return the full prediction payload."""
    started = time.perf_counter()

    payload = validate_upload(data, filename=filename, content_type=content_type)
    metadata = image_metadata(payload)

    if not predictor.is_ready:
        raise ModelNotAvailableError(
            predictor.health().get("error") or "The model is not loaded."
        )

    result = predictor.predict_bytes(payload)

    return {
        "request_id": uuid.uuid4().hex,
        "prediction": result["prediction"],
        "label": result["prediction"],
        "is_pneumonia": result["is_pneumonia"],
        "confidence": round(float(result["confidence"]), 4),
        "probability": round(float(result["probability"]), 4),
        "probabilities": {
            "NORMAL": round(float(result["probabilities"]["NORMAL"]), 4),
            "PNEUMONIA": round(float(result["probabilities"]["PNEUMONIA"]), 4),
        },
        "threshold": float(result["threshold"]),
        "severity": severity_from_probability(float(result["probability"])),
        "inference_ms": float(result["inference_ms"]),
        "processing_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "image": {
            "filename": filename,
            "width": metadata["width"],
            "height": metadata["height"],
            "mode": metadata["mode"],
            "format": metadata["format"],
            "size_bytes": len(payload),
        },
        "model": {
            "name": result["model"]["model_name"],
            "backbone": result["model"]["backbone"],
            "img_size": result["model"]["img_size"],
            "class_names": result["model"]["class_names"],
            "positive_class": result["model"]["positive_class"],
            "threshold": result["model"]["threshold"],
            "created_at": result["model"]["created_at"],
        },
        "disclaimer": DISCLAIMER,
    }
