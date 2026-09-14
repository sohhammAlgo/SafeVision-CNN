import cv2


def draw_detections(frame, pipeline_output):
    """
    Draw person and PPE detections with
    compliance information.
    """

    output = frame.copy()

    # ========================================
    # Draw PPE detections
    # ========================================

    for detection in pipeline_output["ppe_detections"]:

        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 165, 0),
            2
        )

        label = (
            f"{detection.class_name}: "
            f"{detection.confidence:.2f}"
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 165, 0),
            2
        )

    # ========================================
    # Draw persons
    # ========================================

    for item in pipeline_output["persons"]:

        track = item["track"]
        compliance = item["compliance"]

        x1, y1, x2, y2 = track.bbox

        # Green = compliant
        # Red = violation

        if compliance.compliant:
            box_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        # ====================================
        # Label
        # ====================================

        if compliance.compliant:

            label = (
                f"ID {track.track_id} | COMPLIANT"
            )

        else:

            missing = ", ".join(
                compliance.missing_ppe
            )

            label = (
                f"ID {track.track_id} | "
                f"MISSING: {missing}"
            )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

    return output