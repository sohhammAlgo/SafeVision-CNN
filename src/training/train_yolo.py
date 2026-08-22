from pathlib import Path
from ultralytics import YOLO


DATA_YAML = Path(
    "data/raw/ppe_detection/data.yaml"
)

model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=640,
    batch=8,
    workers=2,
    patience=5,
    project="outputs/yolo",
    name="ppe_yolov8n",
    pretrained=True,
    verbose=True,
)

print("\nYOLO training complete.")