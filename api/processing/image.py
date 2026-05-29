"""Image pre-processing utilities."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from config import IMGSZ, LETTERBOX_COLOR


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (IMGSZ, IMGSZ),
    color: tuple[int, int, int] = LETTERBOX_COLOR,
) -> np.ndarray:
    """
    Resize image to target size while preserving aspect ratio (YOLO letterbox).

    Adds padding so the result is exactly new_shape x new_shape.
    """
    shape = image.shape[:2]  # (height, width)
    if shape[0] == new_shape[0] and shape[1] == new_shape[1]:
        return image.copy()

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare an image for YOLOv8 inference.

    Workflow:
        1. Ensure BGR uint8 format
        2. Letterbox-resize to 640x640
        3. Normalize to [0, 1] for optional preview

    Returns:
        original_bgr, preprocessed_bgr (640x640), normalized_preview (float32)
    """
    if image.ndim == 2:
        original_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        original_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 3:
        # Uploaded via PIL is RGB; OpenCV expects BGR
        original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    preprocessed_bgr = letterbox(original_bgr, (IMGSZ, IMGSZ))
    normalized = preprocessed_bgr.astype(np.float32) / 255.0

    return original_bgr, preprocessed_bgr, normalized


def read_image_bytes(data: bytes) -> np.ndarray:
    """Read raw image bytes into an RGB numpy array."""
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(image)
    except Exception as exc:
        raise ValueError(f"Could not read the uploaded image: {exc}") from exc
