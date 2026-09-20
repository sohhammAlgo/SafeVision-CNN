# SafeVision — Real-Time PPE & Face Safety Detection System

SafeVision is a deep-learning-driven workplace safety monitoring system.
It detects people and their PPE (helmet, mask, vest) in images, video,
and live webcam feeds, and reports per-person compliance status in
real time through a Streamlit application.

> **Honesty note:** This repository ships a complete, working
> implementation of every pipeline stage. It does **not** ship a
> pretrained PPE dataset or pretrained weights, and it does not claim
> any accuracy/mAP numbers — those only exist once you train on real
> data with `scripts/train_classifier.py` / `scripts/train_detector.py`
> and evaluate with the corresponding evaluation scripts. See
> [Limitations](#limitations) and the in-app **About** page.

---

## 1. Features

- **Face Mask Classification** — custom CNN (from scratch) and
  MobileNetV2 transfer learning + fine-tuning, with Grad-CAM
  explainability.
- **PPE Object Detection** — YOLO (Ultralytics) detector for
  `person / helmet / mask / vest` (+ optional negative classes).
- **Compliance Engine** — spatial association of PPE to each detected
  person, producing COMPLIANT / VIOLATION status.
- **Real-Time Webcam & Video** — OpenCV-based inference with optional
  lightweight, session-scoped tracking (ByteTrack via Ultralytics).
- **Alert Manager** — per-person/violation cooldown to avoid alert spam.
- **Model Comparison & Evaluation** — accuracy/precision/recall/F1/
  ROC-AUC/confusion-matrix, computed from real predictions.
- **Grad-CAM** — visual explanation of classifier predictions.
- **SQLite persistence**, **Docker** support, **pytest** test suite.

## 2. Architecture

```
SafeVision/
├── app/            Streamlit application (pages, components, config)
├── src/
│   ├── data/        dataset loading, preprocessing, augmentation, splitting
│   ├── models/      custom CNN, MobileNetV2, fine-tuning, YOLO wrapper
│   ├── training/     callbacks, trainer, experiment tracker
│   ├── evaluation/   classification metrics, confusion matrix, Grad-CAM, detection metrics
│   ├── detection/    real-time inference, compliance engine, tracking, alerts
│   ├── db/           SQLite persistence layer
│   └── utils/        logging, file, image, video helpers
├── notebooks/        EDA, custom CNN, transfer learning, evaluation, YOLO analysis
├── scripts/          dataset prep, training, evaluation, CLI inference
├── configs/          classification.yaml, detection.yaml
├── tests/            pytest suite (26 tests, no GPU/dataset required)
├── models/, data/, outputs/, logs/
├── Dockerfile, docker-compose.yml
└── requirements.txt
```

## 3. Deep Learning Concepts Implemented

CNN fundamentals (Conv2D, BatchNorm, ReLU, MaxPool, GAP) · Data
augmentation · Transfer learning (MobileNetV2) · Fine-tuning (partial
unfreeze + low LR) · Class-imbalance handling (class weights, optional
focal loss) · Callbacks (EarlyStopping, ModelCheckpoint,
ReduceLROnPlateau, CSVLogger) · Confusion matrix / precision / recall /
F1 / ROC-AUC · Grad-CAM explainability · YOLO object detection · IoU ·
Non-Maximum Suppression · mAP@50 / mAP@50:95 · Real-time OpenCV inference
· Spatial compliance association · Lightweight multi-object tracking ·
Model comparison · Experiment tracking.

## 4. Model Architectures

**Custom CNN:** `Input → Rescaling → [Conv2D → BatchNorm → ReLU →
MaxPool] × 3 → GlobalAveragePooling2D → Dense → Dropout → Dense(softmax)`

**MobileNetV2 transfer learning:** `Input → preprocess_input →
MobileNetV2(frozen, ImageNet weights) → GlobalAveragePooling2D →
Dropout → Dense(softmax)`, later fine-tuned by unfreezing the top N base
layers and retraining at a 100x lower learning rate.

**YOLO PPE detector:** Ultralytics YOLOv8n as the base checkpoint,
retrained on a PPE-labeled dataset via `src/models/yolo_detector.py`.

## 5. Dataset

No dataset is bundled. See `scripts/download_dataset.py` for:
- A recommended **face mask classification** dataset and its expected
  layout at `data/raw/mask_classification/{mask,no_mask}/`.
- A recommended **PPE detection** dataset (YOLO format) and its
  expected layout at `data/raw/ppe_detection/` with a `data.yaml`.

For pipeline smoke-testing without a real dataset, run:
```
python scripts/prepare_dataset.py --synthetic
```
This generates a tiny, clearly-labeled **synthetic** image set (random
colored shapes, not real faces) so the training/evaluation code paths
can be exercised end-to-end. Results from this mode carry no real-world
meaning and must never be reported as model performance.

## 6. Installation

See [QUICKSTART.md](QUICKSTART.md) for copy-pasteable commands.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Environment Setup

Copy `.env.example` to `.env` and adjust as needed. No secrets are
required for local/offline use; `.env` is git-ignored.

## 8. Training

```bash
python scripts/download_dataset.py --check
python scripts/prepare_dataset.py
python scripts/train_classifier.py --model custom_cnn
python scripts/train_classifier.py --model mobilenet
python scripts/train_detector.py        # requires a PPE detection dataset
```

## 9. Evaluation

```bash
python scripts/evaluate_classifier.py --model custom_cnn
python scripts/evaluate_classifier.py --model mobilenet
```
YOLO detection metrics (precision/recall/mAP50/mAP50-95) are produced by
`YOLODetector.validate()` (see notebook `05_YOLO_Analysis.ipynb`) or the
Ultralytics CLI (`yolo val model=<best.pt> data=<data.yaml>`).

## 10. Running Streamlit

```bash
streamlit run app/streamlit_app.py
```
Then open the printed local URL (default `http://localhost:8501`).

## 11. Webcam Usage

Open the **Live Webcam** page, click **START CAMERA**. Requires the
machine running Streamlit to have a physical webcam accessible to
OpenCV (`cv2.VideoCapture(0)`); this will not work on a headless
server/container without a camera device.

## 12. Video Usage

Open **Video Detection**, upload an MP4/AVI/MOV file. The app processes
frame-by-frame, previews progress, and offers the annotated video for
download.

## 13. Model Comparison

The **Model Comparison** page reads real evaluation JSON files produced
by `scripts/evaluate_classifier.py` — rows only appear for models that
have actually been trained and evaluated.

## 14. Grad-CAM

The **Grad-CAM** page loads a trained classifier, lets you pick a class,
and renders the original image, heatmap, and overlay.

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "Model weights not found" | Model hasn't been trained yet | Run the relevant training script |
| Webcam page shows a device error | No camera accessible in this environment | Run Streamlit locally on a machine with a webcam |
| `DatasetNotFoundError` | Dataset not placed / prepared | Run `download_dataset.py --check` then `prepare_dataset.py` |
| YOLO detects `person` but not PPE classes | Using the generic pretrained fallback (DEMO MODE) | Train a real detector: `scripts/train_detector.py` |
| Slow inference on CPU | No GPU available | Lower resolution, raise frame-skip, use `yolov8n` (already default) |

## 16. Limitations

- PPE-to-person spatial association is heuristic (bounding-box regions),
  not a learned model; dense crowds/heavy occlusion may misassociate PPE.
- No PPE detection dataset ships with the repo — detection quality is
  entirely dependent on the dataset you train on.
- Real-time performance depends on hardware; CPU-only machines will see
  lower FPS than GPU machines.

## 17. Privacy

SafeVision performs **PPE compliance detection**, not facial
recognition or identity inference. Tracking IDs during video/webcam
sessions are temporary and reset every run. See the in-app **About →
Privacy** tab.

## 18. Future Scope

Multi-camera dashboards, model quantization for edge deployment,
active-learning re-annotation loops, per-zone configurable PPE rules.

## 19. Screenshots

_Add screenshots to `assets/screenshots/` after running the app against
real data, and reference them here._

## 20. Project Structure

See section 2 above, or run `tree SafeVision -L 3` after extracting the
archive.
