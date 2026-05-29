# FOD Detection System

Web-based **Foreign Object Debris (FOD) Detection** using **Ultralytics YOLOv8** and **Streamlit**. Upload static images (e.g., airport runways), run inference, and view annotated results with detection counts.

## Active Model

The app uses your custom weights at **`weights/best.pt`**:

| ID | Class  | Filter UI |
|----|--------|-----------|
| 0  | Animal | Yes       |
| 1  | FOD    | Yes       |
| 2  | Hole   | No (excluded from Filter Classes) |

**Filter Classes** in the sidebar only shows **FOD** and **Animal** by default.

## Project Structure

```
grad_project_oli/
├── app.py              # Streamlit web UI
├── config.py           # Paths and inference defaults
├── utils.py            # Pre-process, inference, post-process
├── requirements.txt
├── weights/
│   └── best.pt         # Custom FOD weights (active)
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
3. Use **Filter Classes** to detect **FOD**, **Animal**, or both.
4. Click **Run Detection**.
5. If FOD is found, a flashing pop-up alert appears.

## Alerts

- **FOD detected** → flashing red pop-up warning
- **No FOD** → green success message
- **Animal** detections appear in results but do not trigger the FOD runway alert

## Replace the Model

After retraining, copy new weights:

```bash
cp runs/detect/train/weights/best.pt weights/best.pt
```

Restart Streamlit. The app reloads automatically when `best.pt` changes.

Training command:

```bash
yolo detect train data=data/fod.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16
```
