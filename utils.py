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

from config import (
    ALERT_CLASS_NAMES,
    CLASS_COLORS,
    IMGSZ,
    LETTERBOX_COLOR,
    PRESERVED_CLASS_NAMES,
    get_ensemble_model_paths,
    resolve_model_path,
)


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


def get_model_cache_key(model_path: str) -> str:
    """Build a cache key from the weights path and file modification time."""
    path = Path(model_path)
    if path.is_file():
        stat = path.stat()
        return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    return model_path


def get_ensemble_cache_key() -> str:
    """Cache key covering all ensemble weight files."""
    parts: list[str] = []
    for filename, path in get_ensemble_model_paths():
        stat = path.stat()
        parts.append(f"{filename}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts) if parts else "no-models"


@st.cache_resource(show_spinner="Loading YOLOv8 models...")
def load_ensemble_models(cache_key: str) -> list[tuple[str, YOLO]]:
    """Load all ensemble YOLOv8 checkpoints."""
    del cache_key
    models: list[tuple[str, YOLO]] = []
    for filename, path in get_ensemble_model_paths():
        models.append((filename, YOLO(str(path))))
    return models


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(cache_key: str, model_path: str) -> tuple[YOLO, bool]:
    """Load a single YOLOv8 weights file with Streamlit caching."""
    del cache_key
    _, is_custom = resolve_model_path(Path(model_path).name)
    model = YOLO(model_path)
    return model, is_custom


def normalize_class_name(class_name: str) -> str:
    """Remap obj and other non-priority labels to FOD for unified reporting."""
    if class_name in PRESERVED_CLASS_NAMES:
        return class_name
    return "FOD"


def normalize_detections(
    detections: list[dict[str, Any]],
    selected_filters: list[str],
) -> list[dict[str, Any]]:
    """Normalize class names and keep only selected priority classes."""
    normalized: list[dict[str, Any]] = []

    for det in detections:
        original = det["class_name"]
        display_name = normalize_class_name(original)
        if display_name not in selected_filters:
            continue

        normalized.append(
            {
                **det,
                "class_name": display_name,
                "original_class": original,
                "source_model": det.get("source_model", "unknown"),
            }
        )

    return normalized


def _box_iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    """Intersection-over-Union for two xyxy boxes."""
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms_detections(
    detections: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """Remove overlapping boxes per class, keeping highest-confidence detections."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []

    for candidate in sorted_dets:
        overlaps = False
        for existing in kept:
            if candidate["class_name"] != existing["class_name"]:
                continue
            if _box_iou(candidate, existing) >= iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(candidate)

    return kept


def draw_annotated_image(
    image: np.ndarray, detections: list[dict[str, Any]]
) -> np.ndarray:
    """Draw bounding boxes with normalized FOD / Animal labels."""
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
        label = det["class_name"]
        conf = det["confidence"]
        color = CLASS_COLORS.get(label, (0, 0, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(
            annotated, (x1, max(y1 - th - baseline - 6, 0)), (x1 + tw + 4, y1), color, -1
        )
        cv2.putText(
            annotated,
            caption,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def run_inference(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    iou: float,
) -> Results:
    """
    Run YOLOv8 detection on the enhanced input image with Test-Time Augmentation.

    TTA (`augment=True`) improves recall/confidence on small or low-contrast FOD.
    Sidebar `conf` and `iou` thresholds are passed through unchanged.
    """
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=IMGSZ,
        augment=True,
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
                "source_model": results.path if hasattr(results, "path") else "",
            }
        )

    return detections


def parse_detections_from_results(
    results: Results,
    class_names: dict[int, str],
    source_model: str,
) -> list[dict[str, Any]]:
    """Extract detections and tag them with the originating checkpoint."""
    detections = parse_detections(results, class_names)
    for det in detections:
        det["source_model"] = source_model
    return detections


def run_ensemble_inference(
    models: list[tuple[str, YOLO]],
    image: np.ndarray,
    conf: float,
    iou: float,
    selected_filters: list[str],
    original_bgr: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run all models, merge, normalize, deduplicate, and annotate."""
    combined_raw: list[dict[str, Any]] = []

    for model_name, model in models:
        results = run_inference(model, image, conf, iou)
        combined_raw.extend(
            parse_detections_from_results(results, model.names, model_name)
        )

    normalized = normalize_detections(combined_raw, selected_filters)
    detections = nms_detections(normalized, iou)
    annotated_bgr = draw_annotated_image(original_bgr, detections)
    return annotated_bgr, detections


def postprocess_results(
    results: Results,
    class_names: dict[int, str],
    selected_filters: list[str],
    original_bgr: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Normalize detections, filter priority classes, and draw annotations."""
    raw_detections = parse_detections(results, class_names)
    detections = normalize_detections(raw_detections, selected_filters)
    annotated_bgr = draw_annotated_image(original_bgr, detections)
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
    Build safety alert status from normalized detections.

    Alerts on FOD and Animal when present in the filtered results.
    (obj and other debris classes are normalized to FOD before alerting.)
    """
    alert_items = [det for det in detections if det["class_name"] in ALERT_CLASS_NAMES]

    if not alert_items:
        return {
            "fod_present": False,
            "fod_count": 0,
            "max_confidence": 0.0,
            "message": "No FOD or Animal detected in this image.",
            "alert_types": [],
        }

    max_conf = max(det["confidence"] for det in alert_items)
    count = len(alert_items)
    types = sorted({det["class_name"] for det in alert_items})

    has_fod = "FOD" in types
    has_animal = "Animal" in types

    if has_fod and has_animal:
        title = "FOD & ANIMAL DETECTED"
    elif has_animal:
        title = "ANIMAL DETECTED"
    else:
        title = "FOD DETECTED"

    type_label = ", ".join(types)
    message = (
        f"ALERT: {title}! {count} object{'s' if count != 1 else ''} found "
        f"({type_label}; highest confidence: {max_conf * 100:.1f}%)."
    )

    return {
        "fod_present": True,
        "fod_count": count,
        "max_confidence": max_conf,
        "message": message,
        "alert_types": types,
        "title": title,
    }
