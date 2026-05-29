"""Streamlit session state initialization."""

from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    """Initialize Streamlit session state keys."""
    defaults = {
        "detections": [],
        "summary": {"total": 0, "per_class": {}},
        "annotated_image": None,
        "original_bgr": None,
        "preprocessed_bgr": None,
        "last_upload_name": None,
        "fod_alert": None,
        "alert_dismissed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
