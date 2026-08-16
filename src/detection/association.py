from .types import Detection
from .bbox import center


PPE_CLASSES = {
    "helmet",
    "mask",
    "vest"
}


def associate_ppe_to_person(
    person,
    detections
):
    """
    Associate PPE detections with one person.

    Currently uses the PPE bounding-box center.
    """

    result = {
        "helmet": None,
        "mask": None,
        "vest": None
    }

    px1, py1, px2, py2 = person.bbox

    for detection in detections:

        if detection.class_name not in PPE_CLASSES:
            continue

        cx, cy = center(
            detection.bbox
        )

        if (
            px1 <= cx <= px2
            and
            py1 <= cy <= py2
        ):

            current = result[
                detection.class_name
            ]

            if (
                current is None
                or
                detection.confidence
                > current.confidence
            ):
                result[
                    detection.class_name
                ] = detection

    return result