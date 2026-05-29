"""Data transfer objects for detection results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetectionResult:
    """Complete output from a detection pipeline run."""

    original_bgr: np.ndarray
    preprocessed_bgr: np.ndarray
    annotated_bgr: np.ndarray
    detections: list[dict[str, Any]]
    summary: dict[str, Any]
    fod_alert: dict[str, Any]
