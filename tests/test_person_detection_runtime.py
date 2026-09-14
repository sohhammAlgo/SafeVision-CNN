from pathlib import Path

from src.detection.person_detector import PersonDetector


def test_person_detection_on_image():
    image_path = Path("data/raw/ppe_detection/test/images")

    images = list(image_path.glob("*.jpg"))

    if not images:
        images = list(image_path.glob("*.png"))

    assert images, "No test images found."

    detector = PersonDetector("yolov8n.pt")

    persons = detector.detect(
        str(images[0]),
        confidence=0.5
    )

    print(f"\nImage: {images[0]}")
    print(f"Persons detected: {len(persons)}")

    for person in persons:
        print(person)

    assert isinstance(persons, list)