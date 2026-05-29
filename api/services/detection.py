"""Detection pipeline service — single entry point for the web layer."""

from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from api.processing.alerts import get_fod_alert, summarize_detections
from api.processing.ensemble import merge_ensemble_detections, normalize_detections
from api.processing.image import preprocess_image
from api.processing.inference import (
    parse_detections,
    parse_detections_from_results,
    run_inference,
)
from api.processing.visualization import draw_annotated_image
from api.schemas import DetectionResult
from config import ENSEMBLE_MATCH_IOU


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
    detections = merge_ensemble_detections(
        normalized,
        match_iou=ENSEMBLE_MATCH_IOU,
        nms_iou=iou,
    )
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


class DetectionService:
    """Orchestrates the full FOD detection pipeline."""

    def __init__(self, models: list[tuple[str, YOLO]]) -> None:
        self._models = models

    def detect(
        self,
        image_rgb: np.ndarray,
        conf: float,
        iou: float,
        selected_filters: list[str],
    ) -> DetectionResult:
        """Run preprocessing, ensemble inference, and post-processing."""
        original_bgr, preprocessed_bgr, _ = preprocess_image(image_rgb)

        annotated_bgr, detections = run_ensemble_inference(
            models=self._models,
            image=original_bgr,
            conf=conf,
            iou=iou,
            selected_filters=selected_filters,
            original_bgr=original_bgr,
        )
        summary = summarize_detections(detections)
        fod_alert = get_fod_alert(detections)

        return DetectionResult(
            original_bgr=original_bgr,
            preprocessed_bgr=preprocessed_bgr,
            annotated_bgr=annotated_bgr,
            detections=detections,
            summary=summary,
            fod_alert=fod_alert,
        )
