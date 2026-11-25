"""Pneumonia Detection - shared deep-learning core.

This package holds everything that must behave identically at training time and
at inference time:

* :mod:`pneumonia.config`       - paths, class order, image size, thresholds
* :mod:`pneumonia.preprocessing`- the single image preprocessing pipeline
* :mod:`pneumonia.data`         - KaggleHub download, indexing, balancing, tf.data
* :mod:`pneumonia.model`        - transfer-learning model factory
* :mod:`pneumonia.metrics`      - metrics, threshold search, plots
* :mod:`pneumonia.inference`    - artifact loading + prediction service
* :mod:`pneumonia.train`        - training CLI
* :mod:`pneumonia.evaluate`     - offline evaluation CLI
"""

from .config import (  # noqa: F401
    CLASS_NAMES,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    IMG_SIZE,
    POSITIVE_CLASS,
)

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_METADATA_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_THRESHOLD",
    "IMG_SIZE",
    "POSITIVE_CLASS",
]

__version__ = "1.0.0"
