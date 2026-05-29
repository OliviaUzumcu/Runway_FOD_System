"""Streamlit web application for Foreign Object Debris (FOD) detection."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit adds web/ to sys.path; ensure project root is importable for api/ + config.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from api.models.loader import get_ensemble_cache_key
from config import SUPPORTED_FORMATS
from web.cache import load_ensemble_models_cached
from web.components.results import render_results
from web.components.sidebar import render_sidebar
from web.pipeline import run_detection_web
from web.state import init_session_state


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="FOD Detection System",
        page_icon="🛫",
        layout="wide",
    )

    st.title("FOD Detection System")
    st.markdown(
        "Upload a static image (e.g., airport runway). **Both** trained models "
        "(`best.pt` + `best_v2.pt`) run automatically to detect **FOD** and **Animal**."
    )

    init_session_state()

    models = load_ensemble_models_cached(get_ensemble_cache_key())
    conf, iou, selected_filters = render_sidebar(models)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=SUPPORTED_FORMATS,
        help="Supported formats: JPG, PNG, BMP, WEBP",
    )

    run_clicked = st.button("Run Detection", type="primary", use_container_width=False)

    if uploaded_file is not None and run_clicked:
        with st.spinner("Running dual-model FOD detection..."):
            run_detection_web(models, uploaded_file, conf, iou, selected_filters)

    render_results()


if __name__ == "__main__":
    main()
