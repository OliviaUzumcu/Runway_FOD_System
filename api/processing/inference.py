"""YOLOv8 inference and detection parsing."""

from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from config import IMGSZ


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
