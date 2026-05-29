"""Bounding-box visualization."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from config import CLASS_COLORS


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
