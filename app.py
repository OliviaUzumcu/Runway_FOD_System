"""Streamlit web application for Foreign Object Debris (FOD) detection."""

from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    ENSEMBLE_MODELS,
    SUPPORTED_FORMATS,
    get_filter_options,
)
from utils import (
    get_ensemble_cache_key,
    get_fod_alert,
    load_ensemble_models,
    preprocess_image,
    read_uploaded_image,
    run_ensemble_inference,
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
        "fod_alert": None,
        "alert_dismissed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


def run_detection_pipeline(
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

    rgb_image = read_uploaded_image(uploaded_image)
    if rgb_image is None:
        return

    try:
        original_bgr, preprocessed_bgr, _ = preprocess_image(rgb_image)

        annotated_bgr, detections = run_ensemble_inference(
            models=models,
            image=original_bgr,
            conf=conf,
            iou=iou,
            selected_filters=selected_filters,
            original_bgr=original_bgr,
        )
        summary = summarize_detections(detections)

        st.session_state.original_bgr = original_bgr
        st.session_state.preprocessed_bgr = preprocessed_bgr
        st.session_state.annotated_image = annotated_bgr
        st.session_state.detections = detections
        st.session_state.summary = summary
        st.session_state.last_upload_name = uploaded_image.name
        st.session_state.fod_alert = get_fod_alert(detections)
        st.session_state.alert_dismissed = False

    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def show_fod_popup(alert: dict) -> None:
    """Display a flashing safety alert pop-up dialog."""
    count = alert["fod_count"]
    conf = alert["max_confidence"] * 100
    title = alert.get("title", "FOD DETECTED")
    types = ", ".join(alert.get("alert_types", ["FOD"]))

    st.markdown(
        f"""
        <style>
        @keyframes fodFlash {{
            0%, 100% {{
                background-color: #dc2626;
                border-color: #fecaca;
                box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.8);
            }}
            50% {{
                background-color: #991b1b;
                border-color: #ffffff;
                box-shadow: 0 0 35px 10px rgba(255, 0, 0, 0.5);
            }}
        }}
        @keyframes fodPulseIcon {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.3; transform: scale(1.15); }}
        }}
        .fod-popup-box {{
            animation: fodFlash 0.7s ease-in-out infinite;
            border: 3px solid #fecaca;
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            color: #ffffff;
            margin-bottom: 10px;
            max-width: 300px;
            margin-left: auto;
            margin-right: auto;
        }}
        .fod-popup-icon {{
            font-size: 28px;
            animation: fodPulseIcon 0.7s ease-in-out infinite;
        }}
        .fod-popup-title {{
            font-size: 17px;
            font-weight: 800;
            letter-spacing: 0.5px;
            margin: 4px 0;
        }}
        .fod-popup-text {{
            font-size: 13px;
            line-height: 1.4;
            margin: 0;
        }}
        </style>
        <div class="fod-popup-box">
            <div class="fod-popup-icon">🚨</div>
            <div class="fod-popup-title">{title}</div>
            <p class="fod-popup-text">
                Runway safety issue found in this image.<br>
                <strong>{count}</strong> object{"s" if count != 1 else ""} detected
                ({types}; highest confidence: <strong>{conf:.1f}%</strong>).
            </p>
            <p class="fod-popup-text" style="margin-top:6px; font-size:11px; opacity:0.9;">
                Inspect the marked region(s) before operations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Dismiss Alert", type="primary"):
        st.session_state.alert_dismissed = True
        st.rerun()


@st.dialog("⚠️ Runway Safety Alert", width="small")
def open_fod_alert_dialog(alert: dict) -> None:
    show_fod_popup(alert)


def render_fod_alert() -> None:
    """Show a flashing pop-up alert when FOD is detected."""
    alert = st.session_state.get("fod_alert")
    if alert is None:
        return

    if alert["fod_present"] and not st.session_state.alert_dismissed:
        open_fod_alert_dialog(alert)
    elif st.session_state.original_bgr is not None and not alert["fod_present"]:
        st.success(f"✅ {alert['message']}")


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

    models = load_ensemble_models(get_ensemble_cache_key())
    conf, iou, selected_filters = render_sidebar(models)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=SUPPORTED_FORMATS,
        help="Supported formats: JPG, PNG, BMP, WEBP",
    )

    run_clicked = st.button("Run Detection", type="primary", use_container_width=False)

    if uploaded_file is not None and run_clicked:
        with st.spinner("Running dual-model FOD detection..."):
            run_detection_pipeline(models, uploaded_file, conf, iou, selected_filters)

    render_results()


if __name__ == "__main__":
    main()
