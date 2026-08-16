from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data/raw/ppe_detection/data.yaml",
    epochs=1,
    imgsz=640,
    batch=4,
    project="outputs/ppe_test",
    name="smoke_test"
)

print("PPE training smoke test completed.")