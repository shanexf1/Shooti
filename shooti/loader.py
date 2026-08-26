"""Turn whatever the UI hands us into a BGR numpy array."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_bgr(data: bytes) -> np.ndarray:
    """Decode image bytes to BGR, honoring EXIF orientation.

    Phones write portrait shots as landscape plus an EXIF rotation flag. Without
    exif_transpose every framing measurement would be computed on a sideways
    image, so this is load-bearing, not a nicety.
    """
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
