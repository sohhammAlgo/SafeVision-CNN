from ultralytics import YOLO


# ============================================
# Configuration
# ============================================

DATASET_CONFIG = "data/raw/ppe_detection/data.yaml"

MODEL_NAME = "yolov8n.pt"

OUTPUT_DIR = "models/ppe"


# ============================================
# Load pretrained YOLO
# ============================================

print("Loading YOLOv8n...")

model = YOLO(MODEL_NAME)


# ============================================
# Train
# ============================================

print("Starting PPE training...")

results = model.train(
    data=DATASET_CONFIG,

    epochs=30,

    imgsz=640,

    batch=8,

    workers=2,

    patience=5,

    project=OUTPUT_DIR,

    name="ppe_yolov8n",

    pretrained=True,

    verbose=True
)


print("Training complete.")