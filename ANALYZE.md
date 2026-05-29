# ANALYZE.md — FOD Detection System: Complete Code Analysis

## What This Project Does

This is a **Foreign Object Debris (FOD) Detection** web application for airport runway safety. It accepts a static image (e.g., a runway photo), runs it through **two custom YOLOv8 models simultaneously** (`best.pt` and `best_v2.pt`), merges their results via a weighted ensemble algorithm, and displays annotated bounding boxes with a safety alert pop-up if any FOD or animals are found.

---

## Project Tree Overview

```
grad_project_oli/
├── app.py                       ← Root entry point (just calls web/app.py:main)
├── config.py                    ← All global constants, paths, thresholds
├── requirements.txt             ← Python package dependencies
├── .gitignore                   ← Files excluded from version control
├── .streamlit/
│   └── config.toml              ← Streamlit theme (dark mode, purple)
├── data/
│   └── fod.yaml                 ← YOLO training dataset config
├── weights/
│   ├── best.pt                  ← Primary trained model (FOD, obj, debris)
│   ├── best_v2.pt               ← Secondary trained model (FOD, Animal)
│   └── yolov8n.pt               ← Fallback vanilla YOLOv8 nano model
├── assets/                      ← (empty) for optional sample images
├── web/                         ← Streamlit UI layer
│   ├── app.py                   ← Main Streamlit app function
│   ├── cache.py                 ← Cached model loading wrappers
│   ├── pipeline.py              ← Glue: reads upload, calls API, saves to state
│   ├── state.py                 ← Session state initialization
│   └── components/
│       ├── sidebar.py           ← Sidebar sliders and filter controls
│       ├── results.py           ← Images, metrics, detection table display
│       └── alerts.py            ← Flashing safety alert pop-up
└── api/                         ← Framework-agnostic backend logic
    ├── schemas.py               ← DetectionResult dataclass (output container)
    ├── models/
    │   └── loader.py            ← Loads YOLOv8 model files
    ├── services/
    │   └── detection.py         ← DetectionService: orchestrates the pipeline
    └── processing/
        ├── image.py             ← Letterbox resize + image format conversion
        ├── inference.py         ← Runs YOLO predict, parses box outputs
        ├── ensemble.py          ← Merges two models' results, NMS, confidence fusion
        ├── alerts.py            ← Builds the safety alert dict
        └── visualization.py     ← Draws colored bounding boxes with OpenCV
```

---

## Root Files

---

### `app.py`

**Purpose:** The single-line entry point for the entire application.

```python
from web.app import main
main()
```

**How it works:** Running `streamlit run app.py` imports and calls `main()` from `web/app.py`. This file exists purely so users can launch the app from the project root without specifying the full path to `web/app.py`. It does nothing else.

---

### `config.py`

**Purpose:** Central configuration file — all magic numbers, paths, and thresholds live here so the rest of the codebase imports constants instead of hardcoding values.

**Where values come from:**

| Constant | Value | Origin |
|---|---|---|
| `ROOT` | Project root directory | Resolved from `__file__` at runtime |
| `CUSTOM_MODEL` | `weights/best.pt` | Hardcoded path relative to ROOT |
| `CUSTOM_MODEL_V2` | `weights/best_v2.pt` | Hardcoded path relative to ROOT |
| `FALLBACK_MODEL` | `weights/yolov8n.pt` | Vanilla YOLOv8 nano, used if custom models missing |
| `ENSEMBLE_MODELS` | `["best.pt", "best_v2.pt"]` | Hardcoded list — both always run |
| `IMGSZ` | `640` | Standard YOLOv8 input size |
| `DEFAULT_CONF` | `0.34` | Tuned confidence threshold (lower than default 0.5 to catch small debris) |
| `DEFAULT_IOU` | `0.45` | NMS IoU threshold |
| `ENSEMBLE_MATCH_IOU` | `0.5` | Minimum IoU for two boxes to be considered the same object across models |
| `LETTERBOX_COLOR` | `(114, 114, 114)` | Standard YOLO gray padding color |
| `CLASS_COLORS` | FOD=red, Animal=cyan | BGR values for bounding boxes in OpenCV |

**Key functions:**

- **`resolve_model_path(selected_file)`** — Given an optional filename, checks if it exists in `weights/`. Falls back to `best.pt`, then `yolov8n.pt`. Returns `(path_string, is_custom_bool)`. Used if you ever want single-model mode.

- **`get_ensemble_model_paths()`** — Iterates `ENSEMBLE_MODELS = ["best.pt", "best_v2.pt"]`, checks each file exists in `weights/`, and returns a list of `(filename, Path)` tuples. This is what the loader uses to know which models to load.

- **`get_filter_options()`** — Returns `["FOD", "Animal"]` — the two classes the sidebar multiselect widget shows.

---

### `requirements.txt`

**Purpose:** Python dependency list for `pip install -r requirements.txt`.

| Package | Purpose |
|---|---|
| `streamlit>=1.29.0` | Web UI framework |
| `ultralytics>=8.0.0` | YOLOv8 model loading and inference |
| `opencv-python-headless>=4.8.0` | Image processing, drawing bounding boxes |
| `numpy>=1.24.0` | Array operations on image data |
| `Pillow>=10.0.0` | Reading uploaded image bytes |
| `pandas>=2.0.0` | Detection results table in UI |
| `torch>=2.0.0` | PyTorch — the deep learning backend for YOLOv8 |

`opencv-python-headless` is used instead of `opencv-python` because there is no display/GUI needed (Streamlit handles all rendering).

---

### `.gitignore`

Excludes `.venv/`, `__pycache__/`, compiled Python files (`*.pyc`), `.DS_Store` (macOS metadata), Streamlit secrets, training run outputs (`runs/`), and log files. The model weight files (`weights/*.pt`) are **not** gitignored and are tracked in version control.

---

## `.streamlit/` Directory

---

### `.streamlit/config.toml`

**Purpose:** Sets the Streamlit visual theme.

```toml
[theme]
base = "dark"
primaryColor = "#A855F7"
```

`base = "dark"` activates dark mode. `primaryColor = "#A855F7"` is a purple color applied to buttons, sliders, and interactive elements. This file is read automatically by Streamlit before the app launches — no code in the Python files references it directly.

---

## `data/` Directory

---

### `data/fod.yaml`

**Purpose:** YOLO dataset configuration used during model **training** (not inference). This file is not read by the running app — it was used with the `yolo detect train` command to train `best.pt`.

```yaml
path: ./data/fod-a
train: images/train
val: images/val

names:
  0: Batterytrack_idkeyframe
  1: FOD
  2: Hole
  3: Woodtrack_idkeyframe
  4: obj
```

**What it tells us:**
- The training dataset was stored at `./data/fod-a/` with `images/train` and `images/val` subdirectories.
- The model learned 5 classes: `Batterytrack_idkeyframe`, `FOD`, `Hole`, `Woodtrack_idkeyframe`, and `obj`.
- At inference time, the app normalizes `Hole`, `Batterytrack_idkeyframe`, `Woodtrack_idkeyframe`, and `obj` all into `FOD` (see `ensemble.py`).
- `best_v2.pt` was trained with a different dataset (includes `Animal`) but no separate YAML is present for it.

---

## `weights/` Directory

Contains the actual trained neural network checkpoints. These are PyTorch `.pt` files loaded by `ultralytics.YOLO`.

| File | Size | Classes | Notes |
|---|---|---|---|
| `best.pt` | ~6.0 MB | Batterytrack_idkeyframe, FOD, Hole, Woodtrack_idkeyframe, obj | Primary model, trained on `fod.yaml` dataset |
| `best_v2.pt` | ~6.0 MB | Animal, FOD, Hole | Secondary model, adds animal detection |
| `yolov8n.pt` | ~6.3 MB | 80 COCO classes | Fallback vanilla YOLOv8 nano (used only if custom models missing) |

The `.pt` files are the result of `yolo detect train ...` and contain the full YOLOv8 nano architecture weights. They are loaded at startup via `ultralytics.YOLO(path)` and cached by Streamlit so they don't reload on every page interaction.

---

## `api/` Directory — Backend Logic

The `api/` package is designed to be **framework-agnostic** — it has no Streamlit imports. This means the detection logic could be reused with Flask, FastAPI, or any other framework.

---

### `api/__init__.py`

Empty package marker. Makes `api` importable as a Python package.

---

### `api/schemas.py`

**Purpose:** Defines the data transfer object that the backend returns to the frontend.

```python
@dataclass
class DetectionResult:
    original_bgr: np.ndarray       # The uploaded image as-is (BGR)
    preprocessed_bgr: np.ndarray   # The letterboxed 640×640 version
    annotated_bgr: np.ndarray      # Image with bounding boxes drawn
    detections: list[dict]         # List of detection records
    summary: dict                  # {"total": N, "per_class": {"FOD": N, ...}}
    fod_alert: dict                # Alert payload (fod_present, count, confidence)
```

**Why a dataclass:** Using a dataclass instead of a plain dict gives named attributes, type hints, and prevents key-name typos. The `web/pipeline.py` unpacks each field into `st.session_state`.

---

### `api/models/` — Model Loading

---

#### `api/models/loader.py`

**Purpose:** Handles all YOLOv8 model file loading. Does NOT touch Streamlit — caching is handled separately by `web/cache.py`.

**`get_model_cache_key(model_path)`**
- Builds a cache key string from the file's absolute path, modification time (`st_mtime_ns`), and size (`st_size`).
- **Why:** Streamlit's `@st.cache_resource` uses function arguments as cache keys. If the `.pt` file on disk is replaced (e.g., after retraining), the modification time changes, the key changes, and Streamlit reloads the model instead of using the stale cached version.

**`get_ensemble_cache_key()`**
- Calls `config.get_ensemble_model_paths()` to get `[(filename, Path), ...]` for both `best.pt` and `best_v2.pt`.
- Concatenates each file's mtime and size into one string like `best.pt:17...:6261994|best_v2.pt:17...:6261162`.
- **Data source:** File system metadata of files in `weights/`.

**`load_ensemble_models()`**
- Loops over `get_ensemble_model_paths()` and calls `YOLO(str(path))` for each.
- Returns `[("best.pt", YOLO_object), ("best_v2.pt", YOLO_object)]`.
- **Data source:** `weights/best.pt` and `weights/best_v2.pt` on disk.

**`load_model(model_path)`**
- Loads a single model file, also calls `resolve_model_path` to determine if it's a custom model.
- Returns `(YOLO_object, is_custom_bool)`.

---

### `api/services/` — Orchestration

---

#### `api/services/detection.py`

**Purpose:** The main pipeline orchestrator. Ties together preprocessing, ensemble inference, post-processing, and alert generation into one clean `detect()` call.

**`run_ensemble_inference(models, image, conf, iou, selected_filters, original_bgr)`**

This function runs the actual dual-model detection:

1. Loops over `models` — `[("best.pt", YOLO), ("best_v2.pt", YOLO)]`
2. For each model: calls `run_inference(model, image, conf, iou)` → gets raw YOLOv8 `Results` object
3. Calls `parse_detections_from_results(results, model.names, model_name)` — extracts box coordinates, confidence scores, and class names, and tags each detection with which model produced it
4. All detections from both models are combined into `combined_raw`
5. `normalize_detections(combined_raw, selected_filters)` — remaps `obj`, `Hole`, etc. to `FOD`
6. `merge_ensemble_detections(normalized, match_iou=0.5, nms_iou=iou)` — fuses boxes from both models that overlap the same object
7. `draw_annotated_image(original_bgr, detections)` — draws final bounding boxes

**`postprocess_results(results, class_names, selected_filters, original_bgr)`**

A simpler single-model version (not currently used in the main flow but kept for potential future single-model mode).

**`DetectionService` class**

The public API surface. The web layer creates `DetectionService(models)` and calls `.detect(image_rgb, conf, iou, selected_filters)`.

**`.detect()` method:**

1. Calls `preprocess_image(image_rgb)` → returns `(original_bgr, preprocessed_bgr, _)`. The `original_bgr` is what inference runs on (full resolution) and what bounding boxes are drawn on.
2. Calls `run_ensemble_inference(...)` → returns `(annotated_bgr, detections)`
3. Calls `summarize_detections(detections)` → `{"total": N, "per_class": {...}}`
4. Calls `get_fod_alert(detections)` → alert dict
5. Packages everything into a `DetectionResult` dataclass and returns it

---

### `api/processing/` — Core Processing Modules

---

#### `api/processing/image.py`

**Purpose:** Converts uploaded images into the format YOLOv8 expects.

**`letterbox(image, new_shape=(640,640), color=(114,114,114))`**

Resizes an image to exactly 640×640 without distorting the aspect ratio:

1. Calculates the scale ratio: `ratio = min(640 / height, 640 / width)` — uses the smaller ratio so the image fits inside the box
2. Resizes the image to `(width*ratio, height*ratio)` using `cv2.INTER_LINEAR` (bilinear interpolation)
3. Calculates remaining padding: `dw = 640 - new_width`, `dh = 640 - new_height`
4. Splits padding equally on both sides (top/bottom, left/right) with gray `(114,114,114)` fill
5. Returns the padded image with `cv2.copyMakeBorder`

**Why `(114,114,114)`:** This is the standard YOLO padding color — it's a neutral gray that the model was trained with, so using the same color at inference avoids distribution shift.

**Data source:** The input `image` numpy array (from `read_image_bytes`).

**`preprocess_image(image)`**

Handles color space conversion:
- Grayscale → BGR (`cv2.COLOR_GRAY2BGR`)
- RGBA → BGR (`cv2.COLOR_RGBA2BGR`, drops alpha channel)
- RGB → BGR (`cv2.COLOR_RGB2BGR`) — PIL loads images as RGB, OpenCV uses BGR

Then calls `letterbox()` to get the 640×640 version, and divides by 255.0 for the float32 normalized preview.

Returns three images: `(original_bgr, preprocessed_bgr, normalized_preview)`.

**`read_image_bytes(data: bytes)`**

Takes raw file bytes (from `uploaded_file.getvalue()` in Streamlit), opens them with PIL's `Image.open()`, converts to RGB, and returns a numpy array. PIL handles all image formats (JPG, PNG, BMP, WEBP) uniformly.

**Data source:** Raw bytes from the user's uploaded file via Streamlit's `st.file_uploader`.

---

#### `api/processing/inference.py`

**Purpose:** Runs YOLOv8 prediction and extracts structured data from the results.

**`run_inference(model, image, conf, iou)`**

Calls `model.predict(...)` with:
- `source=image` — the BGR numpy array (full-resolution original)
- `conf=conf` — confidence threshold from the sidebar slider (default 0.34)
- `iou=iou` — NMS IoU threshold from the sidebar slider (default 0.45)
- `imgsz=640` — from `config.IMGSZ`, tells YOLOv8 to internally resize to 640 (even if we pass full-res image)
- `augment=True` — **Test-Time Augmentation (TTA)**: runs inference on the image at multiple scales and flips, then merges. Improves recall for small or low-contrast FOD at the cost of ~3× inference time
- `verbose=False` — suppresses per-inference console output

Returns `results[0]` — a single `Results` object (not a list).

**Data source:** The `model` YOLO object loaded from `weights/best.pt` or `weights/best_v2.pt`. The `conf` and `iou` values come from the sidebar sliders in `web/components/sidebar.py`.

**`parse_detections(results, class_names)`**

Extracts raw box data from the YOLOv8 `Results` object:
- `results.boxes.xyxy` — bounding box coordinates as `[x1, y1, x2, y2]` in pixel coordinates
- `results.boxes.conf` — confidence scores (0.0 to 1.0)
- `results.boxes.cls` — integer class IDs

Converts them from PyTorch tensors to numpy (`.cpu().numpy()`), then looks up the class name using `class_names` — the model's `names` dict (e.g., `{0: "Batterytrack_idkeyframe", 1: "FOD", ...}`).

Returns a list of dicts like:
```python
{
    "class_name": "FOD",
    "confidence": 0.87,
    "x1": 120.3, "y1": 45.1, "x2": 200.8, "y2": 95.6,
    "source_model": ""
}
```

**Data source:** The `results` object from YOLOv8's `.predict()` call. Class names come from `model.names` which is embedded in the `.pt` file at training time (derived from `fod.yaml`'s `names` section).

**`parse_detections_from_results(results, class_names, source_model)`**

Calls `parse_detections()` then overwrites `source_model` on each detection with the model filename (e.g., `"best.pt"` or `"best_v2.pt"`). This tagging is needed by the ensemble merger to know which model each box came from.

---

#### `api/processing/ensemble.py`

**Purpose:** The most algorithmically complex module. Merges detections from two models, fuses confidence scores, deduplicates overlapping boxes.

**`normalize_class_name(class_name)`**

If the class name is in `PRESERVED_CLASS_NAMES = {"FOD", "Animal"}`, keep it as-is. Otherwise (for `obj`, `Hole`, `Batterytrack_idkeyframe`, `Woodtrack_idkeyframe`), return `"FOD"`.

**Data source:** `config.PRESERVED_CLASS_NAMES`.

**`normalize_detections(detections, selected_filters)`**

Loops over all raw detections:
1. Saves the original class name as `original_class`
2. Calls `normalize_class_name()` to get the display name
3. Drops the detection if the display name is not in `selected_filters` (the user's sidebar multiselect)
4. Adds `original_class` and `source_model` fields

This is why the results table shows both "Class" (e.g., `FOD`) and "Raw Model Class" (e.g., `obj`).

**`_box_iou(box_a, box_b)`**

Computes **Intersection-over-Union** for two bounding boxes in xyxy format:
1. Finds the intersection rectangle: `x1=max(a.x1,b.x1)`, `y1=max(a.y1,b.y1)`, `x2=min(a.x2,b.x2)`, `y2=min(a.y2,b.y2)`
2. Computes intersection area (0 if no overlap)
3. Computes union area = `area_a + area_b - intersection`
4. Returns `intersection / union`

Used to determine if two boxes are "the same object" (IoU ≥ 0.5) or "different objects" (IoU < 0.5).

**`fuse_confidence_scores(confidences)`**

Implements the **independent event probability formula** to boost confidence when multiple models agree on an object:

```
fused = 1 - (1 - conf1) * (1 - conf2) * ...
```

- Single model: returns the original score unchanged (math works out: `1 - (1-c) = c`)
- Two models at 0.8 each: `1 - (1-0.8)*(1-0.8) = 1 - 0.04 = 0.96`
- This models confidence as probability: two independent detectors both seeing the object is stronger evidence than one

**`_fuse_detection_cluster(cluster)`**

Takes a group (cluster) of detections that all correspond to the same physical object and fuses them into one:
1. **Confidence:** Uses `fuse_confidence_scores()` — boosted if multiple models agree
2. **Box coordinates:** Weighted average of all boxes, weighted by each detection's confidence. Higher-confidence detections pull the final box position more.
3. **Source model:** Joins all model names with `+` (e.g., `"best.pt+best_v2.pt"`) — this becomes the "Source Model" column in the UI
4. **`model_agreement`:** Count of distinct source models (1 or 2) — becomes "Models Agreed" column in the UI

**`merge_ensemble_detections(detections, match_iou=0.5, nms_iou=0.45)`**

The main ensemble fusion algorithm:

1. Sorts all detections by confidence (highest first)
2. Greedy clustering: takes the highest-confidence detection as a "seed", finds all other detections of the **same class** that overlap it by IoU ≥ `match_iou` (0.5), groups them into a cluster
3. Each cluster (1 or more detections) is fused via `_fuse_detection_cluster()`
4. After fusion, runs `nms_detections()` to remove any remaining overlapping fused boxes

**Why this matters:** Without this, running two models would show duplicate bounding boxes for the same object. This algorithm collapses them into one confident detection.

**Data source:** The `ENSEMBLE_MATCH_IOU = 0.5` and `DEFAULT_IOU = 0.45` constants from `config.py`. The detections list comes from both models combined by `detection.py`.

**`nms_detections(detections, iou_threshold)`**

Standard **Non-Maximum Suppression**: sorts by confidence, iterates through boxes, keeps a box only if it doesn't heavily overlap (IoU ≥ threshold) any already-kept box of the same class. This is applied after ensemble fusion as a final cleanup pass.

---

#### `api/processing/visualization.py`

**Purpose:** Draws bounding boxes on the image using OpenCV.

**`draw_annotated_image(image, detections)`**

For each detection:
1. Converts float coordinates to int pixel positions
2. Looks up color from `config.CLASS_COLORS`: FOD → red `(0,0,255)` BGR, Animal → cyan `(0,255,255)` BGR
3. Draws the rectangle with `cv2.rectangle(..., thickness=2)`
4. Computes label text: `"FOD 0.87"` (class name + confidence to 2 decimal places)
5. Measures text size with `cv2.getTextSize` to draw a filled background rectangle behind the label (so text is readable on any background)
6. Draws white text with `cv2.putText` using anti-aliasing (`cv2.LINE_AA`)

Returns a copy of the image with all boxes drawn. The copy is important — the original image array is preserved untouched.

**Data source:** Colors from `config.CLASS_COLORS`. Detection coordinates from the ensemble merger output.

---

#### `api/processing/alerts.py`

**Purpose:** Computes the safety alert status dict from detections. No UI code here — this is pure logic.

**`summarize_detections(detections)`**

Counts total detections and per-class counts. Returns:
```python
{"total": 3, "per_class": {"FOD": 2, "Animal": 1}}
```

**`get_fod_alert(detections)`**

Filters detections to only those in `ALERT_CLASS_NAMES = {"FOD", "Animal"}` (from `config.py`).

If none found → returns a "clean" dict with `fod_present: False`.

If found:
- Finds the maximum confidence score across all alert detections
- Collects the unique classes detected (e.g., `["Animal", "FOD"]`)
- Builds a title string: `"FOD DETECTED"`, `"ANIMAL DETECTED"`, or `"FOD & ANIMAL DETECTED"` depending on what was found
- Returns:
```python
{
    "fod_present": True,
    "fod_count": 3,
    "max_confidence": 0.94,
    "message": "ALERT: FOD DETECTED! 2 objects found ...",
    "alert_types": ["FOD"],
    "title": "FOD DETECTED"
}
```

**Data source:** `ALERT_CLASS_NAMES` from `config.py`. The detections list with normalized class names from the ensemble pipeline.

---

## `web/` Directory — Streamlit UI Layer

---

### `web/__init__.py`

Minimal package marker. Contains a `sys.path` fix ensuring the project root is importable from within the `web/` package context.

---

### `web/app.py`

**Purpose:** The Streamlit application entry point. Renders the page, wires all components together.

**`main()`**

1. **Path fix:** Adds the project root to `sys.path` so `api/` and `config.py` can be imported when Streamlit runs `web/app.py` directly (Streamlit adds `web/` to the path by default, which would break `from config import ...`).

2. **`st.set_page_config(...)`** — Sets browser tab title to "FOD Detection System", page icon to ✈️, and layout to "wide" (full browser width instead of centered column).

3. **`init_session_state()`** — Initializes all session state keys to defaults.

4. **`load_ensemble_models_cached(get_ensemble_cache_key())`** — Loads both `.pt` models into memory. The cache key (mtime+size of both files) ensures models reload if weights are replaced on disk but not otherwise (expensive operation).

5. **`render_sidebar(models)`** — Draws the sidebar, returns `(conf, iou, selected_filters)`.

6. **`st.file_uploader(...)`** — Shows the upload widget. Accepts JPG, PNG, BMP, WEBP (from `config.SUPPORTED_FORMATS`). Returns a file-like object or `None`.

7. **`st.button("Run Detection", ...)`** — Primary action button. Detection only runs when both a file is uploaded AND the button is clicked (not on every rerun).

8. **`run_detection_web(models, uploaded_file, conf, iou, selected_filters)`** — Runs the full pipeline, results go into `st.session_state`.

9. **`render_results()`** — Displays whatever is in session state (images, table, alerts).

---

### `web/state.py`

**Purpose:** Ensures all session state keys exist before any component tries to read them.

**`init_session_state()`**

Defines a `defaults` dict and sets each key in `st.session_state` only if it doesn't already exist (`if key not in st.session_state`). This is the standard Streamlit pattern to avoid resetting state on page reruns.

| Key | Default | Holds |
|---|---|---|
| `detections` | `[]` | List of detection dicts from ensemble |
| `summary` | `{"total": 0, "per_class": {}}` | Count summary |
| `annotated_image` | `None` | BGR numpy array with drawn boxes |
| `original_bgr` | `None` | BGR numpy array, original upload |
| `preprocessed_bgr` | `None` | BGR numpy array, 640×640 letterboxed |
| `last_upload_name` | `None` | Original filename string |
| `fod_alert` | `None` | Alert dict from `get_fod_alert()` |
| `alert_dismissed` | `False` | Whether the user clicked "Dismiss Alert" |

---

### `web/cache.py`

**Purpose:** Wraps model loading with Streamlit's `@st.cache_resource` decorator so models are loaded once per process, not on every page interaction.

**`load_ensemble_models_cached(cache_key)`**

- Decorated with `@st.cache_resource(show_spinner="Loading YOLOv8 models...")` — Streamlit caches the return value keyed by `cache_key`.
- `del cache_key` — The parameter is consumed as a cache key signal. The actual loading is done by `api/models/loader.py:load_ensemble_models()` which reads from `weights/`.
- The `cache_key` string encodes the mtime+size of both `.pt` files. If you replace `best.pt` with a retrained version, the mtime changes → new cache key → Streamlit reloads the model. Otherwise the YOLO objects stay in memory.

**`load_model_cached(cache_key, model_path)`**

Single-model version. Not currently used in the main ensemble flow but available for single-model mode.

---

### `web/pipeline.py`

**Purpose:** Glue layer between the Streamlit upload event and the backend API. Reads the uploaded file bytes, calls `DetectionService`, and stores results in `st.session_state`.

**`run_detection_web(models, uploaded_image, conf, iou, selected_filters)`**

1. Checks `if not models` — if no `.pt` files were found, shows an error and returns.
2. `read_image_bytes(uploaded_image.getvalue())` — converts the Streamlit `UploadedFile` bytes to an RGB numpy array. If this fails (corrupt file), shows the error message and returns.
3. `DetectionService(models).detect(...)` — runs the full ensemble pipeline.
4. Unpacks the returned `DetectionResult` into `st.session_state` fields: `original_bgr`, `preprocessed_bgr`, `annotated_image`, `detections`, `summary`, `last_upload_name`, `fod_alert`.
5. Resets `alert_dismissed = False` so the alert fires again for each new detection.
6. Any unexpected exception is caught and displayed with a full traceback in an expander.

**Data sources:** `uploaded_image.getvalue()` — raw bytes from Streamlit's file uploader. `conf` and `iou` from sidebar sliders. `selected_filters` from sidebar multiselect.

---

### `web/components/` — UI Components

---

#### `web/components/sidebar.py`

**Purpose:** Renders all sidebar controls and returns the user's chosen inference settings.

**`render_sidebar(models)`**

1. Shows `"Dual-model ensemble active"` (green) or a missing weights error (red) depending on whether `models` is non-empty.

2. **Confidence slider:** `st.sidebar.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONF=0.34, step=0.01)` — values from `config.DEFAULT_CONF`. The slider value is passed directly to `model.predict(conf=...)`.

3. **IoU slider:** `st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, DEFAULT_IOU=0.45, step=0.01)` — controls Non-Maximum Suppression aggressiveness.

4. **Filter Classes multiselect:** `st.sidebar.multiselect(...)` with options `["FOD", "Animal"]` from `config.get_filter_options()`. All selected by default. If the user deselects everything, it resets to all-selected (prevents accidentally showing zero detections).

5. Returns `(conf, iou, selected_classes)` — these flow directly into `run_detection_web()`.

**Data sources:** Default values from `config.py`. Ensemble model names from `config.ENSEMBLE_MODELS`. Filter options from `config.get_filter_options()`.

---

#### `web/components/results.py`

**Purpose:** Renders the detection results section of the main page.

**`render_results()`**

1. If `original_bgr is None` (no detection run yet), shows a placeholder info message and returns.

2. **`render_fod_alert()`** — triggers the safety alert pop-up if needed.

3. **Two-column image display:**
   - Left column: original uploaded image (from `st.session_state.original_bgr`, displayed as BGR)
   - Right column: annotated image with bounding boxes (from `st.session_state.annotated_image`)

4. **Expander "Pre-processed Input (640×640 letterbox)":** Shows the letterboxed version so users can see what the model actually received as input.

5. **Detection summary metrics:** Creates up to 4 metric columns (Streamlit limit). First column shows total count. Remaining columns show per-class counts (`FOD: 3`, `Animal: 1`). If there are more classes than columns, extras are shown as a caption text.

6. **Detection table:** Uses `pandas.DataFrame` from `st.session_state.detections`. Formats confidence as a percentage string. Renames technical column names to human-readable headers. Shows: Class, Raw Model Class, Source Model, Models Agreed, Confidence, X1, Y1, X2, Y2.

**Data sources:** All data comes from `st.session_state`, which was populated by `web/pipeline.py` after the detection run.

---

#### `web/components/alerts.py`

**Purpose:** Handles the visual safety alert UI elements.

**`show_fod_popup(alert)`**

Generates a flashing red animated HTML/CSS box injected into the Streamlit page with `st.markdown(..., unsafe_allow_html=True)`.

CSS animations:
- `@keyframes fodFlash` — alternates between `#dc2626` (red) and `#991b1b` (dark red) with a growing box-shadow every 0.7 seconds
- `@keyframes fodPulseIcon` — pulses the 🚨 icon between full opacity/size and 30% opacity/115% size

The popup box shows:
- The `title` (e.g., `"FOD DETECTED"`, from `get_fod_alert()`)
- Count of detected objects
- Which types were found (e.g., `"FOD, Animal"`)
- The highest confidence score as a percentage
- A "Dismiss Alert" button — clicking it sets `st.session_state.alert_dismissed = True` and calls `st.rerun()` to close the dialog

**Data sources:** `alert["fod_count"]`, `alert["max_confidence"]`, `alert["title"]`, `alert["alert_types"]` — all from `api/processing/alerts.py:get_fod_alert()`.

**`open_fod_alert_dialog(alert)`**

Decorated with `@st.dialog("⚠️ Runway Safety Alert", width="small")` — wraps the popup in a Streamlit modal dialog window.

**`render_fod_alert()`**

The entry point called by `results.py`:
- If `fod_alert` is `None` (no detection run), does nothing
- If `fod_alert["fod_present"]` is `True` and alert hasn't been dismissed → opens the modal dialog
- If detection ran but nothing was found → shows a green `✅ No FOD or Animal detected` success message

---

## Complete Data Flow Diagram

```
User uploads image
        │
        ▼
web/app.py:main()
  → st.file_uploader() ──────────────────────────────────────┐
  → st.button("Run Detection")                               │
        │ (both clicked)                                      │
        ▼                                                     │
web/pipeline.py:run_detection_web()                          │
  → api/processing/image.py:read_image_bytes()               │
        │ raw bytes from UploadedFile ◄───────────────────────┘
        │ PIL.Image.open() → RGB numpy array
        ▼
api/services/detection.py:DetectionService.detect()
  → api/processing/image.py:preprocess_image()
        │ RGB → BGR, letterbox to 640×640
        ▼
  → api/services/detection.py:run_ensemble_inference()
        │
        ├── For "best.pt" (primary model):
        │     api/processing/inference.py:run_inference()
        │       → model.predict(augment=True, conf=0.34, iou=0.45, imgsz=640)
        │       Data source: weights/best.pt
        │     → parse_detections_from_results()
        │       → boxes.xyxy, boxes.conf, boxes.cls from YOLO Results
        │       → model.names from best.pt embedded class list
        │       → tags each det with source_model="best.pt"
        │
        └── For "best_v2.pt" (secondary model):
              api/processing/inference.py:run_inference()
                → model.predict(augment=True, conf=0.34, iou=0.45, imgsz=640)
                Data source: weights/best_v2.pt
              → parse_detections_from_results()
                → model.names from best_v2.pt embedded class list
                → tags each det with source_model="best_v2.pt"
        │
        ▼  combined_raw = all detections from both models
  → api/processing/ensemble.py:normalize_detections()
        │ obj → FOD, Hole → FOD, Batterytrack → FOD, etc.
        │ drops detections not in selected_filters
        ▼
  → api/processing/ensemble.py:merge_ensemble_detections()
        │ IoU clustering: boxes from different models that overlap ≥ 0.5
        │   are grouped → fused into one weighted box
        │ fuse_confidence_scores(): 1-(1-c1)(1-c2) formula
        │ nms_detections(): remove remaining duplicates
        ▼
  → api/processing/visualization.py:draw_annotated_image()
        │ CLASS_COLORS from config.py (FOD=red, Animal=cyan)
        │ cv2.rectangle + cv2.putText for each detection
        ▼
  → api/processing/alerts.py:summarize_detections()  → {"total": N, "per_class": {...}}
  → api/processing/alerts.py:get_fod_alert()         → {"fod_present": True, ...}
        ▼
DetectionResult dataclass
  original_bgr, preprocessed_bgr, annotated_bgr, detections, summary, fod_alert
        │
        ▼
web/pipeline.py → unpacks into st.session_state
        │
        ▼
web/components/results.py:render_results()
  → shows original vs annotated images
  → metric tiles (total, per-class counts)
  → pandas DataFrame table

web/components/alerts.py:render_fod_alert()
  → if fod_present: open_fod_alert_dialog() → flashing CSS modal
  → if clean: green success message
```

---

## Key Design Decisions

### Why Two Models?
`best.pt` was trained on general FOD (nails, small debris, battery fragments, wood). `best_v2.pt` adds `Animal` detection. Running both catches everything either model knows about, and the ensemble merging boosts confidence when both agree.

### Why `normalize_class_name()` Maps `obj` → `FOD`?
The training dataset (`data/fod.yaml`) has a class called `obj` which represents generic/unclassified debris. Rather than showing this technical label to end users, the app maps it to the cleaner `FOD` category. The original model output is preserved in the `original_class` field for inspection.

### Why `augment=True` in Inference?
Test-Time Augmentation (TTA) runs each image through the model at multiple scales and horizontal flips, then merges the resulting boxes. This significantly improves recall on small runway objects (bolts, nuts, small debris) at the cost of ~3× slower inference. For a safety-critical application this tradeoff makes sense.

### Why Streamlit Session State?
Streamlit reruns the entire Python script from top to bottom on every user interaction (slider move, button click, file upload). Session state (`st.session_state`) is a persistent dict that survives these reruns within a session, allowing detection results from a previous "Run Detection" click to still be displayed while the user adjusts a slider.

### Why `@st.cache_resource` for Models?
Loading a 6MB `.pt` checkpoint and initializing PyTorch tensors takes ~1-2 seconds. `@st.cache_resource` keeps the loaded YOLO objects in memory across reruns and even across multiple browser sessions. The cache key (mtime+size) ensures stale models are replaced when weights files are updated on disk.

### Why Separate `api/` and `web/` Packages?
The `api/` package has zero Streamlit imports, making it independently testable and reusable. The web layer (`web/`) acts as a thin adapter that reads Streamlit state and delegates all logic to `api/`. This architecture means you could add a REST API endpoint (`/detect`) with ~10 lines of code without touching any detection logic.
