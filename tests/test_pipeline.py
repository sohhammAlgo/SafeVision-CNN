from pathlib import Path

from src.detection.pipeline import SafetyPipeline


def test_safety_pipeline_on_image():

    image_dir = Path(
        "data/raw/ppe_detection/test/images"
    )

    images = list(image_dir.glob("*.jpg"))

    if not images:
        images = list(image_dir.glob("*.png"))

    assert images, "No test images found."

    print(f"\nTesting image: {images[0]}")

    pipeline = SafetyPipeline()

    results = pipeline.process(
        str(images[0])
    )

    print(f"Persons processed: {len(results)}")

    for result in results:
        print(
            f"\nPerson ID: {result.track_id}"
        )

        print(
            f"Helmet: {result.helmet}"
        )

        print(
            f"Mask: {result.mask}"
        )

        print(
            f"Vest: {result.vest}"
        )

        print(
            f"Compliant: {result.compliant}"
        )

        print(
            f"Missing PPE: {result.missing_ppe}"
        )

        print(
            f"Severity: {result.severity}"
        )

    assert isinstance(results, list)