"""Shared, lazily created service objects (dependency-injection helpers)."""

from __future__ import annotations

import threading
from typing import Optional

from pneumonia.inference import PneumoniaPredictor

_lock = threading.Lock()
_predictor: Optional[PneumoniaPredictor] = None


def get_predictor() -> PneumoniaPredictor:
    """Return the process-wide predictor, loading the model on first use."""
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                _predictor = PneumoniaPredictor()
    return _predictor


def set_predictor(predictor: PneumoniaPredictor) -> None:
    """Replace the singleton (used by the test-suite)."""
    global _predictor
    with _lock:
        _predictor = predictor


def reset_predictor() -> None:
    global _predictor
    with _lock:
        _predictor = None
