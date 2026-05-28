"""Central configuration for the FOD Detection System."""

from pathlib import Path

# Project root (directory containing this file)
ROOT = Path(__file__).resolve().parent

# Model paths
WEIGHTS_DIR = ROOT / "weights"
CUSTOM_MODEL = WEIGHTS_DIR / "best.pt"
FALLBACK_MODEL = WEIGHTS_DIR / "yolov8n.pt"

# Inference defaults (aligned with Ultralytics Streamlit reference)
IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# Letterbox padding color (BGR) — YOLO convention
LETTERBOX_COLOR = (114, 114, 114)

# Supported upload formats
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]

# Optional sample assets
ASSETS_DIR = ROOT / "assets"
SAMPLE_IMAGE = ASSETS_DIR / "sample_runway.jpg"


def resolve_model_path() -> tuple[str, bool]:
    """
    Resolve which weights file to load.

    Returns:
        Tuple of (model_path, is_custom) where is_custom is True when
        weights/best.pt exists.
    """
    if CUSTOM_MODEL.exists():
        return str(CUSTOM_MODEL), True
    # Ultralytics auto-downloads yolov8n.pt on first use if not present locally
    return str(FALLBACK_MODEL) if FALLBACK_MODEL.exists() else "yolov8n.pt", False
