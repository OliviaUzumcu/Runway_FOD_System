"""Web-layer detection pipeline — delegates to api and updates session state."""

from __future__ import annotations

import traceback

import streamlit as st

from api.processing.image import read_image_bytes
from api.services.detection import DetectionService


def run_detection_web(
    models: list[tuple[str, object]],
    uploaded_image,
    conf: float,
    iou: float,
    selected_filters: list[str],
) -> None:
    """Execute the full ensemble detection pipeline."""
    if not models:
        st.error("No model weights found in `weights/`.")
        return

    try:
        rgb_image = read_image_bytes(uploaded_image.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        result = DetectionService(models).detect(
            image_rgb=rgb_image,
            conf=conf,
            iou=iou,
            selected_filters=selected_filters,
        )

        st.session_state.original_bgr = result.original_bgr
        st.session_state.preprocessed_bgr = result.preprocessed_bgr
        st.session_state.annotated_image = result.annotated_bgr
        st.session_state.detections = result.detections
        st.session_state.summary = result.summary
        st.session_state.last_upload_name = uploaded_image.name
        st.session_state.fod_alert = result.fod_alert
        st.session_state.alert_dismissed = False

    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
