"""Streamlit-cached model loading."""

from __future__ import annotations

import streamlit as st
from ultralytics import YOLO

from api.models.loader import load_ensemble_models, load_model


@st.cache_resource(show_spinner="Loading YOLOv8 models...")
def load_ensemble_models_cached(cache_key: str) -> list[tuple[str, YOLO]]:
    """Load all ensemble YOLOv8 checkpoints with Streamlit caching."""
    del cache_key
    return load_ensemble_models()


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model_cached(cache_key: str, model_path: str) -> tuple[YOLO, bool]:
    """Load a single YOLOv8 weights file with Streamlit caching."""
    del cache_key
    return load_model(model_path)
