"""FOD safety alert UI components."""

from __future__ import annotations

import streamlit as st


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
