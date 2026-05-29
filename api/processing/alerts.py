"""Detection summary and safety alert logic."""

from __future__ import annotations

from typing import Any

from config import ALERT_CLASS_NAMES


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
