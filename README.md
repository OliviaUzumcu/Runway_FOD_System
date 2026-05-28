# FOD Detection System

Web-based **Foreign Object Debris (FOD) Detection** using **Ultralytics YOLOv8** and **Streamlit**. Upload static images (e.g., airport runways), run inference, and view annotated results with detection counts.

## Active Model

The app uses your custom weights at **`weights/best.pt`** with these classes:

| ID | Class |
|----|-------|
| 0 | Batterytrack_idkeyframe |
| 1 | FOD |
| 2 | Hole |
| 3 | Woodtrack_idkeyframe |
| 4 | obj |

## Project Structure

```
grad_project_oli/
├── app.py              # Streamlit web UI
├── config.py           # Paths and inference defaults
├── utils.py            # Pre-process, inference, post-process
├── requirements.txt
├── weights/
│   ├── best.pt         # Custom FOD weights (active)
│   └── yolov8n.pt      # Fallback placeholder
├── assets/             # Optional sample images
└── data/
    └── fod.yaml        # Dataset config matching best.pt classes
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
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Usage

1. Upload a JPG, PNG, BMP, or WEBP image (runway / taxiway photo).
2. Adjust **Confidence Threshold** and **IoU (NMS)** in the sidebar if needed.
3. Optionally filter by class (FOD, Hole, obj, etc.).
4. Click **Run Detection**.
5. View the original image, annotated result, detection counts, and bounding-box table.

## Workflow

1. **Upload** — Select a static image.
2. **Pre-process** — Letterbox resize to 640×640 (preview in expander).
3. **Inference** — YOLOv8 runs on `weights/best.pt` with NMS.
4. **Post-process** — Bounding boxes, labels, and confidence scores drawn on output.
5. **Results** — Metrics and per-detection table in the UI.

## Sidebar Settings

| Setting | Description |
|---------|-------------|
| Confidence Threshold | Minimum score to keep a detection (default 0.25) |
| IoU Threshold (NMS) | Overlap threshold for Non-Maximum Suppression (default 0.45) |
| Filter Classes | Limit output to selected classes from your model |

## Replace or Retrain the Model

To update weights after retraining:

```bash
cp runs/detect/train/weights/best.pt weights/best.pt
```

Restart Streamlit. The app reloads automatically when `best.pt` changes.

Training command (update `data/fod.yaml` paths first):

```bash
yolo detect train data=data/fod.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16
```

## YOLOv8 API

- `YOLO("weights/best.pt")` — load custom weights
- `model.predict(conf=..., iou=..., imgsz=640)` — inference with built-in NMS
- `model.names` — class labels from the trained checkpoint
