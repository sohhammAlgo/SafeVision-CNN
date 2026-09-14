from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = Path(
    "runs/detect/models/ppe/ppe_yolov8n/weights/best.pt"
)

DATA_YAML = Path(
    "data/raw/ppe_detection/data.yaml"
)


def main():
    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=8,
        workers=2,
        plots=True,
    )

    print("\nYOLO Evaluation Complete")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()