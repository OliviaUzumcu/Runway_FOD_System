"""YOLOv8 model loading without framework dependencies."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from config import get_ensemble_model_paths, resolve_model_path


def get_model_cache_key(model_path: str) -> str:
    """Build a cache key from the weights path and file modification time."""
    path = Path(model_path)
    if path.is_file():
        stat = path.stat()
        return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    return model_path


def get_ensemble_cache_key() -> str:
    """Cache key covering all ensemble weight files."""
    parts: list[str] = []
    for filename, path in get_ensemble_model_paths():
        stat = path.stat()
        parts.append(f"{filename}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts) if parts else "no-models"


def load_ensemble_models() -> list[tuple[str, YOLO]]:
    """Load all ensemble YOLOv8 checkpoints."""
    models: list[tuple[str, YOLO]] = []
    for filename, path in get_ensemble_model_paths():
        models.append((filename, YOLO(str(path))))
    return models


def load_model(model_path: str) -> tuple[YOLO, bool]:
    """Load a single YOLOv8 weights file."""
    _, is_custom = resolve_model_path(Path(model_path).name)
    model = YOLO(model_path)
    return model, is_custom
