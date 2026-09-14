from ultralytics import YOLO


class PersonDetector:
    """
    Detects persons using a pretrained YOLO model.
    COCO class 0 = person.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, frame, confidence=0.5):
        """
        Detect persons in a frame.

        Returns:
            list of dictionaries containing:
            - bbox
            - confidence
            - class_name
        """

        results = self.model(
            frame,
            conf=confidence,
            verbose=False
        )

        persons = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id != self.PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                persons.append({
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "confidence": float(box.conf[0]),
                    "class_name": "person"
                })

        return persons