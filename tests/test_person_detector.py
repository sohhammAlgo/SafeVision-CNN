from src.detection.person_detector import PersonDetector


def test_person_detector_initialization():
    detector = PersonDetector("yolov8n.pt")

    assert detector.model is not None
    assert detector.PERSON_CLASS_ID == 0