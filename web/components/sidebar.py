"""Sidebar controls for inference settings."""

from __future__ import annotations

import streamlit as st

from config import DEFAULT_CONF, DEFAULT_IOU, ENSEMBLE_MODELS, get_filter_options


def render_sidebar(models: list[tuple[str, object]]) -> tuple[float, float, list[str]]:
    """Render sidebar controls and return inference settings."""
    st.sidebar.title("Settings")

    if models:
        st.sidebar.success("Dual-model ensemble active")
    else:
        st.sidebar.error(
            f"Missing weights. Add {' and '.join(ENSEMBLE_MODELS)} under `weights/`."
        )

    conf = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONF,
        step=0.01,
        help="Minimum confidence score to keep a detection.",
    )
    iou = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_IOU,
        step=0.01,
        help="Intersection-over-Union threshold for Non-Maximum Suppression.",
    )

    filter_options = get_filter_options()

    selected_classes = st.sidebar.multiselect(
        "Filter Classes",
        options=filter_options,
        default=filter_options,
        help="FOD includes obj and all other debris classes remapped from the models.",
    )
    if not selected_classes:
        selected_classes = filter_options

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Workflow:** Upload image → Both models infer → Merge → "
        "Normalize (FOD / Animal) → Alert"
    )

    return conf, iou, selected_classes
