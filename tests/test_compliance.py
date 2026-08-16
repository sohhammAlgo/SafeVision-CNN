from src.detection.types import Detection
from src.detection.compliance import evaluate_compliance


def test_compliant_person():

    detections = {
        "helmet": Detection(
            "helmet",
            0.9,
            (10, 10, 50, 50)
        ),

        "mask": Detection(
            "mask",
            0.9,
            (20, 20, 50, 50)
        ),

        "vest": Detection(
            "vest",
            0.9,
            (20, 50, 80, 150)
        )
    }

    result = evaluate_compliance(
        track_id=1,
        associated_ppe=detections
    )

    assert result.compliant is True

def test_missing_mask():

    detections = {
        "helmet": Detection(
            "helmet",
            0.9,
            (10, 10, 50, 50)
        ),

        "mask": None,

        "vest": Detection(
            "vest",
            0.9,
            (20, 50, 80, 150)
        )
    }

    result = evaluate_compliance(
        track_id=1,
        associated_ppe=detections
    )

    assert result.compliant is False
    assert result.mask is False