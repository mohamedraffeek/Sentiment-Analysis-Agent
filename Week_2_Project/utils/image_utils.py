"""Utility functions for loading and preprocessing images for the sentiment tool.

Responsibilities:
- Load an image from a filesystem path.
- Load an image from a base64 data URI or raw base64 string.
- Provide a uniform Pillow Image object.
- (Optional) future: face detection, multi-face handling, cropping.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(image_input: str) -> Image.Image:
    """Load an image from a filesystem path, data URI, or base64 string.

    Strategy:
    1. Strip surrounding quotes/whitespace.
    2. If path exists -> load.
    3. If *looks like* a path (drive letter / slash / supported extension) but does not exist -> explicit 'not found'.
    4. If data URI -> decode.
    5. Else attempt raw base64 decode; on failure raise clear error.
    """
    image_input = image_input.strip().strip('"').strip("'")

    # Normalization: allow forward slashes on Windows
    if os.name == 'nt' and '/' in image_input:
        image_input = image_input.replace('/', '\\')

    if os.path.exists(image_input):
        return _load_from_path(image_input)

    if _looks_like_path(image_input):
        raise ValueError(f"Image path not found: {image_input}")

    if image_input.startswith("data:image"):
        return _load_from_data_uri(image_input)

    # Try raw base64
    try:
        return _load_from_base64(image_input)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Could not load image: input is neither an existing path, a valid data URI, nor valid base64."
        ) from exc


def _load_from_path(path: str) -> Image.Image:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {ext}")
    with Image.open(path) as im:
        return im.convert("RGB")


def _load_from_data_uri(data_uri: str) -> Image.Image:
    try:
        header, b64data = data_uri.split(",", 1)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError("Invalid data URI format") from exc
    return _decode_base64_to_image(b64data)


def _load_from_base64(b64data: str) -> Image.Image:
    return _decode_base64_to_image(b64data)


def _decode_base64_to_image(b64data: str) -> Image.Image:
    try:
        binary = base64.b64decode(b64data)
        bio = io.BytesIO(binary)
        with Image.open(bio) as im:
            return im.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 image data") from exc


def _looks_like_path(s: str) -> bool:
    lower = s.lower()
    if ':' in s[:4]:  # Windows drive letter
        return True
    if any(ch in s for ch in ('/', '\\')):
        return True
    return any(lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def resize_if_needed(image: Image.Image, max_side: int = 512) -> Image.Image:
    """Resize the image maintaining aspect ratio if the longest side exceeds max_side."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.BICUBIC) # type: ignore


__all__ = [
    "load_image",
    "resize_if_needed",
    "_looks_like_path",
]