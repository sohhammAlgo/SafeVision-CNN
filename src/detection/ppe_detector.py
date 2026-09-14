from pathlib import Path

from ultralytics import YOLO

from .types import Detection


class PPEDetector:
    """
    Detects PPE items using the trained YOLOv8 model.

    Classes:
        helmet
        mask
        vest
    """

    def __init__(
        self,
        model_path="runs/detect/models/ppe/ppe_yolov8n/weights/best.pt"
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"PPE model not found: {model_path}"
            )

        self.model = YOLO(str(model_path))

    def detect(self, frame, confidence=0.5):
        """
        Detect PPE objects in an image/frame.

        Returns:
            List[Detection]
        """

        results = self.model(
            frame,
            conf=confidence,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = result.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection(
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox=(
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    )
                )

                detections.append(detection)

        return detections