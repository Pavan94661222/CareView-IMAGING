"""Upload validation for the prediction endpoint.

Everything that can go wrong with a user-supplied file is turned into a
``UploadValidationError`` carrying the HTTP status the API should return.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from pneumonia import config


class UploadValidationError(ValueError):
    """Raised when an uploaded file cannot be accepted."""

    def __init__(self, message: str, status_code: int = 400, code: str = "invalid_upload"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def _check_extension(filename: Optional[str]) -> None:
    if not filename:
        return
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in config.IMAGE_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file extension '{suffix}'. "
            f"Allowed formats: {', '.join(config.IMAGE_EXTENSIONS)}.",
            status_code=400,
            code="unsupported_extension",
        )


def _check_content_type(content_type: Optional[str]) -> None:
    if not content_type:
        return
    main_type = content_type.split(";")[0].strip().lower()
    if main_type not in config.ALLOWED_CONTENT_TYPES:
        raise UploadValidationError(
            f"Unsupported content type '{content_type}'. Please upload a JPEG/PNG radiograph.",
            status_code=400,
            code="unsupported_media_type",
        )


def validate_upload(
    data: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> bytes:
    """Validate a chest X-ray upload and return the raw bytes.

    Raises
    ------
    UploadValidationError
        400 for unusable payloads, 413 when the file is too large.
    """
    if not data:
        raise UploadValidationError("The uploaded file is empty.", code="empty_file")

    if len(data) > config.MAX_UPLOAD_BYTES:
        limit_mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
        raise UploadValidationError(
            f"File is too large ({len(data) / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {limit_mb:.1f} MB.",
            status_code=413,
            code="file_too_large",
        )

    _check_extension(filename)
    _check_content_type(content_type)

    # Pillow is the arbiter of "is this really an image".
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except UnidentifiedImageError as exc:
        raise UploadValidationError(
            "The file is not a recognised image. Upload a JPEG or PNG chest X-ray.",
            code="not_an_image",
        ) from exc
    except UploadValidationError:
        raise
    except Exception as exc:
        raise UploadValidationError(
            f"The image file appears to be corrupted or truncated ({exc}).",
            code="corrupt_image",
        ) from exc

    # Re-open (verify() leaves the file unusable) to inspect the real metadata.
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
    except Exception as exc:
        raise UploadValidationError(
            f"Could not read the image dimensions ({exc}).", code="corrupt_image"
        ) from exc

    if image_format and image_format not in config.ALLOWED_IMAGE_FORMATS:
        raise UploadValidationError(
            f"Unsupported image format '{image_format}'. Use JPEG, PNG, BMP or TIFF.",
            code="unsupported_format",
        )

    minimum = max(32, config.IMG_SIZE[0] // 8)
    if width < minimum or height < minimum:
        raise UploadValidationError(
            f"Image is too small ({width}x{height}). Minimum is {minimum}x{minimum} pixels.",
            code="image_too_small",
        )
    if width > 20000 or height > 20000:
        raise UploadValidationError(
            f"Image is unreasonably large ({width}x{height}).", code="image_too_large"
        )

    return data


def image_metadata(data: bytes) -> dict:
    """Return width/height/mode/format of a previously validated image."""
    with Image.open(io.BytesIO(data)) as image:
        return {
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "mode": image.mode,
            "format": (image.format or "").upper(),
        }
