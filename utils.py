"""Pre-processing, inference, and post-processing utilities for FOD detection."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import IMGSZ, LETTERBOX_COLOR, resolve_model_path


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


def read_uploaded_image(uploaded_file) -> np.ndarray | None:
    """Read a Streamlit uploaded file into an RGB numpy array."""
    if uploaded_file is None:
        return None

    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        return np.array(image)
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        return None


def get_model_cache_key() -> str:
    """Build a cache key from the resolved weights path and file modification time."""
    model_path, _ = resolve_model_path()
    path = Path(model_path)
    if path.is_file():
        stat = path.stat()
        return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    return model_path


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(cache_key: str) -> tuple[YOLO, bool]:
    """
    Load YOLOv8 weights with Streamlit caching.

    Uses weights/best.pt when available; otherwise falls back to yolov8n.pt.
    cache_key ensures the model reloads when best.pt is added or replaced.
    """
    del cache_key  # used only for cache invalidation
    model_path, is_custom = resolve_model_path()
    model = YOLO(model_path)
    return model, is_custom


def run_inference(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    iou: float,
    classes: list[int] | None = None,
) -> Results:
    """
    Run YOLOv8 detection on the original image.

    Ultralytics handles letterbox resize, normalization, and NMS internally.
    The iou parameter controls Non-Maximum Suppression overlap threshold.
    """
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=IMGSZ,
        classes=classes,
        verbose=False,
    )
    return results[0]


def parse_detections(results: Results, class_names: dict[int, str]) -> list[dict[str, Any]]:
    """Extract structured detection records from YOLOv8 results."""
    detections: list[dict[str, Any]] = []

    if results.boxes is None or len(results.boxes) == 0:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        name = class_names.get(int(class_id), str(class_id))
        detections.append(
            {
                "class_name": name,
                "confidence": float(confidence),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            }
        )

    return detections


def postprocess_results(
    results: Results, class_names: dict[int, str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Build annotated output image and detection list.

    Uses results.plot() for bounding boxes, labels, and confidence scores.
    """
    annotated_bgr = results.plot()
    detections = parse_detections(results, class_names)
    return annotated_bgr, detections


def summarize_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute total and per-class detection counts for the UI."""
    per_class: dict[str, int] = {}
    for det in detections:
        name = det["class_name"]
        per_class[name] = per_class.get(name, 0) + 1

    return {
        "total": len(detections),
        "per_class": per_class,
    }


def get_fod_alert(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build FOD alert status from detection results.

    Matches the trained class label "FOD" (case-insensitive).
    """
    fod_items = [
        det for det in detections if det["class_name"].strip().upper() == "FOD"
    ]

    if not fod_items:
        return {
            "fod_present": False,
            "fod_count": 0,
            "max_confidence": 0.0,
            "message": "No Foreign Object Debris (FOD) detected in this image.",
        }

    max_conf = max(det["confidence"] for det in fod_items)
    count = len(fod_items)
    message = (
        f"ALERT: Foreign Object Debris (FOD) detected in this image! "
        f"{count} FOD object{'s' if count != 1 else ''} found "
        f"(highest confidence: {max_conf * 100:.1f}%)."
    )

    return {
        "fod_present": True,
        "fod_count": count,
        "max_confidence": max_conf,
        "message": message,
    }
