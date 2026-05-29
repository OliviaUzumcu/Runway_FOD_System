# FOD Detection System

Web-based **Foreign Object Debris (FOD) Detection** using **Ultralytics YOLOv8** and **Streamlit**. Upload static images (e.g., airport runways), run dual-model inference, and view annotated results with safety alerts.

## Model Weights (`weights/`)

The app loads **both** checkpoints automatically and merges their results — no model selection needed.

| File | Raw classes | Role |
|------|-------------|------|
| **`best.pt`** | Batterytrack_idkeyframe, **FOD**, Hole, Woodtrack_idkeyframe, **obj** | Primary — nails, small debris, general FOD |
| **`best_v2.pt`** | **Animal**, **FOD**, Hole | Secondary — animal detection on runway |

### Unified output (UI)

All detections are normalized to two priority classes:

| Display class | Source |
|---------------|--------|
| **FOD** | Native FOD, **obj**, Hole, Batterytrack, Woodtrack, and all other debris labels |
| **Animal** | Native Animal (from `best_v2.pt`) |

The results table shows both **Class** (normalized) and **Raw Model Class** (original model label).

## Project Structure

```
grad_project_oli/
├── config.py               # Shared paths, ensemble config, defaults
├── web/                    # Streamlit UI
│   ├── app.py              # Entry point: streamlit run web/app.py
│   ├── cache.py            # Streamlit model caching
│   ├── pipeline.py         # Upload → api service → session state
│   ├── state.py            # Session state initialization
│   └── components/
│       ├── sidebar.py      # Confidence / IoU / filter controls
│       ├── alerts.py       # FOD safety alert dialog
│       └── results.py      # Images, metrics, detection table
├── api/                    # Framework-agnostic detection logic
│   ├── schemas.py          # DetectionResult dataclass
│   ├── models/
│   │   └── loader.py       # YOLOv8 ensemble loading
│   ├── services/
│   │   └── detection.py    # DetectionService entry point
│   └── processing/
│       ├── image.py        # Preprocess, letterbox, read_image_bytes
│       ├── inference.py    # YOLO predict + parse
│       ├── ensemble.py     # Merge, NMS, class normalization
│       ├── visualization.py
│       └── alerts.py       # Summary + safety alert logic
├── requirements.txt
├── weights/
│   ├── best.pt             # Primary FOD + obj model
│   └── best_v2.pt          # Secondary FOD + Animal model
├── assets/                 # Optional sample images
└── data/
    └── fod.yaml            # YOLO dataset config template
```

## Setup

```bash
cd /Users/olivia/Desktop/grad_project_oli
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
source .venv/bin/activate
streamlit run web/app.py
# or: streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Usage

1. Upload a JPG, PNG, BMP, or WEBP image (runway / taxiway photo).
2. Both **`best.pt`** and **`best_v2.pt`** run automatically (dual-model ensemble).
3. Adjust sidebar settings if needed:
   - **Confidence Threshold** — default `0.34`
   - **IoU Threshold (NMS)** — default `0.45`
   - **Filter Classes** — **FOD** and **Animal** (both selected by default)
4. Click **Run Detection**.
5. View original vs annotated image, detection summary, and results table.

## Alerts

A flashing pop-up appears when any of the following are detected:

- **FOD** (includes obj and remapped debris)
- **Animal**

If nothing is detected → green success message.

## Replace Weights

After retraining, copy weights into `weights/`:

```bash
cp runs/detect/train/weights/best.pt weights/best.pt
cp runs/detect/train/weights/best.pt weights/best_v2.pt   # if retraining v2
```

Restart Streamlit. Models reload when file contents change.

Training command (update `data/fod.yaml` paths first):

```bash
yolo detect train data=data/fod.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16
```

## Workflow

```
Upload image → best.pt + best_v2.pt inference → Merge & NMS
→ Normalize (obj → FOD) → Annotated output → Alert if FOD/Animal found
```
