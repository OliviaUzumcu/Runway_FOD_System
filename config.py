"""Central configuration for the FOD Detection System."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

WEIGHTS_DIR = ROOT / "weights"
CUSTOM_MODEL = WEIGHTS_DIR / "best.pt"
CUSTOM_MODEL_V2 = WEIGHTS_DIR / "best_v2.pt"
FALLBACK_MODEL = WEIGHTS_DIR / "yolov8n.pt"

# Both checkpoints are always loaded and run together (ensemble)
ENSEMBLE_MODELS = ["best.pt", "best_v2.pt"]

# Main filter classes shown in the UI (obj is merged into FOD)
PRIORITY_FILTER_CLASSES = ["FOD", "Animal"]

# Classes that keep their original label after inference
PRESERVED_CLASS_NAMES = {"FOD", "Animal"}

# Any other model output (including obj) is remapped to FOD
REMAP_UNKNOWN_TO_FOD = True

# Classes that trigger the safety alert pop-up
ALERT_CLASS_NAMES = {"FOD", "Animal"}

IMGSZ = 640
DEFAULT_CONF = 0.34
DEFAULT_IOU = 0.45
# Cross-model box matching threshold for ensemble fusion (WBF-style)
ENSEMBLE_MATCH_IOU = 0.5
LETTERBOX_COLOR = (114, 114, 114)
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]

ASSETS_DIR = ROOT / "assets"
SAMPLE_IMAGE = ASSETS_DIR / "sample_runway.jpg"

# BGR colors for custom bounding-box drawing
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "FOD": (0, 0, 255),
    "Animal": (0, 255, 255),
}


def resolve_model_path(selected_file: str | None = None) -> tuple[str, bool]:
    if selected_file:
        candidate = WEIGHTS_DIR / selected_file
        if candidate.exists():
            return str(candidate), True

    if CUSTOM_MODEL.exists():
        return str(CUSTOM_MODEL), True

    return str(FALLBACK_MODEL) if FALLBACK_MODEL.exists() else "yolov8n.pt", False


def get_ensemble_model_paths() -> list[tuple[str, Path]]:
    """Return available ensemble model filenames and paths."""
    available: list[tuple[str, Path]] = []
    for filename in ENSEMBLE_MODELS:
        path = WEIGHTS_DIR / filename
        if path.exists():
            available.append((filename, path))
    return available


def get_filter_options() -> list[str]:
    """Return the three main Filter Classes options for the UI."""
    return list(PRIORITY_FILTER_CLASSES)
