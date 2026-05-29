"""Ensemble merge, normalization, and NMS."""

from __future__ import annotations

from typing import Any

from config import DEFAULT_IOU, ENSEMBLE_MATCH_IOU, PRESERVED_CLASS_NAMES


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


def fuse_confidence_scores(confidences: list[float]) -> float:
    """
    Boost confidence when multiple models agree (independent-detection formula).

    Single model: returns the original score unchanged.
    Two models: conf_final = 1 - (1 - conf1) * (1 - conf2)
    """
    fused = 0.0
    for conf in confidences:
        fused = 1.0 - (1.0 - fused) * (1.0 - conf)
    return min(fused, 1.0)


def _fuse_detection_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    """Fuse a cluster of cross-model detections into one weighted box."""
    confidences = [det["confidence"] for det in cluster]
    fused_conf = fuse_confidence_scores(confidences)
    weight_sum = sum(confidences) or 1.0

    fused_box = {
        axis: sum(det[axis] * det["confidence"] for det in cluster) / weight_sum
        for axis in ("x1", "y1", "x2", "y2")
    }

    best_det = max(cluster, key=lambda det: det["confidence"])
    sources = sorted({det["source_model"] for det in cluster})

    return {
        "class_name": best_det["class_name"],
        "original_class": best_det["original_class"],
        "confidence": fused_conf,
        **fused_box,
        "source_model": "+".join(sources),
        "model_agreement": len(sources),
    }


def merge_ensemble_detections(
    detections: list[dict[str, Any]],
    match_iou: float = ENSEMBLE_MATCH_IOU,
    nms_iou: float = DEFAULT_IOU,
) -> list[dict[str, Any]]:
    """
    IoU-based weighted box fusion for the dual-model ensemble.

    Detections from different checkpoints with the same normalized class and
    IoU >= match_iou are merged into one box. Agreement boosts confidence via
    fuse_confidence_scores(); single-model hits keep their original score.
    """
    if not detections:
        return []

    clusters: list[list[dict[str, Any]]] = []
    assigned = [False] * len(detections)
    order = sorted(
        range(len(detections)),
        key=lambda idx: detections[idx]["confidence"],
        reverse=True,
    )

    for idx in order:
        if assigned[idx]:
            continue

        seed = detections[idx]
        cluster = [seed]
        assigned[idx] = True

        for other_idx in order:
            if assigned[other_idx]:
                continue
            candidate = detections[other_idx]
            if candidate["class_name"] != seed["class_name"]:
                continue
            if any(
                _box_iou(candidate, member) >= match_iou for member in cluster
            ):
                cluster.append(candidate)
                assigned[other_idx] = True

        clusters.append(cluster)

    fused = [_fuse_detection_cluster(cluster) for cluster in clusters]
    return nms_detections(fused, nms_iou)


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
