"""Detection results display."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from web.components.alerts import render_fod_alert


def render_results() -> None:
    """Display detection results from session state."""
    if st.session_state.original_bgr is None:
        st.info("Upload an image and click **Run Detection** to begin.")
        return

    render_fod_alert()

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.image(
            st.session_state.original_bgr,
            channels="BGR",
            caption=f"Original — {st.session_state.last_upload_name}",
            use_container_width=True,
        )
    with col_result:
        st.image(
            st.session_state.annotated_image,
            channels="BGR",
            caption="Detection Result",
            use_container_width=True,
        )

    with st.expander("Pre-processed Input (640×640 letterbox)"):
        st.image(
            st.session_state.preprocessed_bgr,
            channels="BGR",
            caption="Resized and padded for YOLOv8 input",
            use_container_width=True,
        )

    summary = st.session_state.summary
    st.subheader("Detection Summary")

    metric_cols = st.columns(min(len(summary["per_class"]) + 1, 4))
    metric_cols[0].metric("Total Detected", summary["total"])

    for idx, (class_name, count) in enumerate(summary["per_class"].items()):
        if idx + 1 < len(metric_cols):
            metric_cols[idx + 1].metric(class_name, count)

    if summary["per_class"] and len(summary["per_class"]) + 1 > len(metric_cols):
        st.caption(
            "Additional classes: "
            + ", ".join(
                f"{name} ({count})"
                for name, count in list(summary["per_class"].items())[len(metric_cols) - 1 :]
            )
        )

    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections)
        df["confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
        rename_map = {
            "class_name": "Class",
            "original_class": "Raw Model Class",
            "source_model": "Source Model",
            "model_agreement": "Models Agreed",
            "confidence": "Confidence",
            "x1": "X1",
            "y1": "Y1",
            "x2": "X2",
            "y2": "Y2",
        }
        df = df.rename(columns=rename_map)
        display_cols = [
            "Class",
            "Raw Model Class",
            "Source Model",
            "Models Agreed",
            "Confidence",
            "X1",
            "Y1",
            "X2",
            "Y2",
        ]
        df = df[[c for c in display_cols if c in df.columns]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No FOD or Animal detected at the current threshold settings.")
