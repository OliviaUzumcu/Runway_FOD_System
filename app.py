"""Streamlit web application for Foreign Object Debris (FOD) detection."""

from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from config import DEFAULT_CONF, DEFAULT_IOU, SUPPORTED_FORMATS
from config import CUSTOM_MODEL
from utils import (
    get_model_cache_key,
    load_model,
    postprocess_results,
    preprocess_image,
    read_uploaded_image,
    run_inference,
    summarize_detections,
)


def init_session_state() -> None:
    """Initialize Streamlit session state keys."""
    defaults = {
        "detections": [],
        "summary": {"total": 0, "per_class": {}},
        "annotated_image": None,
        "original_bgr": None,
        "preprocessed_bgr": None,
        "last_upload_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(model, is_custom: bool) -> tuple[float, float, list[int] | None]:
    """Render sidebar controls and return inference settings."""
    st.sidebar.title("Settings")

    if is_custom:
        st.sidebar.success("Custom FOD model loaded")
        st.sidebar.caption(f"`{CUSTOM_MODEL.name}` — {len(model.names)} classes")
        with st.sidebar.expander("Model classes"):
            for class_id, class_name in model.names.items():
                st.write(f"{class_id}: **{class_name}**")
    else:
        st.sidebar.warning(
            "Using placeholder model (`yolov8n.pt`). "
            "Place your trained weights at `weights/best.pt` for FOD-specific detection."
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

    class_names = list(model.names.values())
    selected_classes = st.sidebar.multiselect(
        "Filter Classes",
        options=class_names,
        default=class_names,
        help="Limit detections to selected classes.",
    )
    class_filter = [class_names.index(name) for name in selected_classes] or None

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Workflow:** Upload image → Pre-process (640×640) → "
        "YOLOv8 inference → NMS → Annotated output"
    )

    return conf, iou, class_filter


def run_detection_pipeline(
    model,
    uploaded_image,
    conf: float,
    iou: float,
    class_filter: list[int] | None,
) -> None:
    """Execute the full detection pipeline and store results in session state."""
    rgb_image = read_uploaded_image(uploaded_image)
    if rgb_image is None:
        return

    try:
        # Step 1: Pre-processing — resize to 640x640 and normalize for preview
        original_bgr, preprocessed_bgr, _ = preprocess_image(rgb_image)

        # Step 2: Inference — pass original image; YOLOv8 handles resize + NMS
        results = run_inference(
            model=model,
            image=original_bgr,
            conf=conf,
            iou=iou,
            classes=class_filter,
        )

        # Step 3: Post-processing — draw boxes, labels, confidence scores
        annotated_bgr, detections = postprocess_results(results, model.names)
        summary = summarize_detections(detections)

        st.session_state.original_bgr = original_bgr
        st.session_state.preprocessed_bgr = preprocessed_bgr
        st.session_state.annotated_image = annotated_bgr
        st.session_state.detections = detections
        st.session_state.summary = summary
        st.session_state.last_upload_name = uploaded_image.name

    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def render_results() -> None:
    """Display detection results from session state."""
    if st.session_state.original_bgr is None:
        st.info("Upload an image and click **Run Detection** to begin.")
        return

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
    metric_cols[0].metric("Total FOD Detected", summary["total"])

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
        df = df.rename(
            columns={
                "class_name": "Class",
                "confidence": "Confidence",
                "x1": "X1",
                "y1": "Y1",
                "x2": "X2",
                "y2": "Y2",
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No foreign object debris detected at the current threshold settings.")


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="FOD Detection System",
        page_icon="🛫",
        layout="wide",
    )

    st.title("FOD Detection System")
    st.markdown(
        "Upload a static image (e.g., airport runway) to detect **Foreign Object Debris** "
        "using **Ultralytics YOLOv8**."
    )

    init_session_state()

    model, is_custom = load_model(get_model_cache_key())
    conf, iou, class_filter = render_sidebar(model, is_custom)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=SUPPORTED_FORMATS,
        help="Supported formats: JPG, PNG, BMP, WEBP",
    )

    run_clicked = st.button("Run Detection", type="primary", use_container_width=False)

    if uploaded_file is not None and run_clicked:
        with st.spinner("Running FOD detection..."):
            run_detection_pipeline(model, uploaded_file, conf, iou, class_filter)

    render_results()


if __name__ == "__main__":
    main()
