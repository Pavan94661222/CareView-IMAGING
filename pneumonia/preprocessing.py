"""Single source of truth for image preprocessing.

The training pipeline, the offline evaluation script and the FastAPI backend
all call the functions in this module, which guarantees that an image is
transformed exactly the same way at train time and at inference time.

Canonical representation
------------------------
``float32`` tensor / array with shape ``(height, width, 3)`` in the **[0, 255]**
range, channels ordered RGB.  Model specific normalisation (ImageNet mean/std
subtraction for MobileNetV2 / VGG16, the built-in ``Rescaling`` layer of
EfficientNet) is baked *inside* the Keras model, so nothing outside the model
has to know about it.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import tensorflow as tf

from .config import IMG_SIZE

ImageLike = Union[str, bytes, bytearray, "np.ndarray"]


def resize_canonical(image: tf.Tensor, img_size: tuple[int, int] = IMG_SIZE) -> tf.Tensor:
    """Resize an RGB image tensor and cast it to ``float32`` in [0, 255].

    This is the *only* geometric/numeric transform applied outside the model,
    and it is shared by the tf.data training pipeline and the web backend.
    """
    resized = tf.image.resize(image, img_size, method="bilinear", antialias=True)
    return tf.cast(resized, tf.float32)


def preprocess_bytes(data: bytes, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Decode raw image bytes (jpeg/png/bmp/tiff) into the canonical array.

    Raises
    ------
    ValueError
        If the payload is not a decodable single-frame image.
    """
    if not data:
        raise ValueError("Empty image payload.")
    try:
        decoded = tf.io.decode_image(
            tf.constant(data, dtype=tf.string),
            channels=3,
            expand_animations=False,
        )
    except Exception as exc:  # pragma: no cover - depends on TF internals
        raise ValueError("Uploaded file is not a valid image.") from exc
    decoded.set_shape([None, None, 3])
    return resize_canonical(decoded, img_size).numpy()


def preprocess_file(path: str, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Read an image from disk and return the canonical array."""
    with open(path, "rb") as handle:
        return preprocess_bytes(handle.read(), img_size)


def preprocess_pil(image, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Convert a :class:`PIL.Image.Image` into the canonical array."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = tf.convert_to_tensor(array)
    return resize_canonical(tensor, img_size).numpy()


def preprocess_array(array: np.ndarray, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Convert an already decoded HWC numpy array (uint8 or float) to canonical."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    tensor = tf.convert_to_tensor(array)
    if tensor.dtype != tf.uint8:
        tensor = tf.cast(tf.clip_by_value(tensor, 0, 255), tf.uint8)
    return resize_canonical(tensor, img_size).numpy()


def to_batch(canonical: np.ndarray) -> np.ndarray:
    """Add the batch dimension expected by :meth:`tf.keras.Model.predict`."""
    array = np.asarray(canonical, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a HWC image, received shape {array.shape}.")
    return array[np.newaxis, ...]


def preprocess_upload(data: bytes, img_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Backend entry point: raw upload bytes -> ``(1, H, W, 3)`` float32 batch."""
    return to_batch(preprocess_bytes(data, img_size))
