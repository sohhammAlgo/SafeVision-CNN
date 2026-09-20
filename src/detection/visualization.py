import cv2


def draw_label(
    image,
    text,
    x1,
    y1,
    color,
    font_scale=0.5
):
    """
    Draw a readable label above a bounding box.
    If there is not enough space above the box,
    place the label inside the box.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    label_x = max(x1, 0)

    label_y = y1 - 5

    if label_y - text_height - baseline < 0:
        label_y = y1 + text_height + baseline + 5

    # Background rectangle
    cv2.rectangle(
        image,
        (
            label_x,
            label_y - text_height - baseline
        ),
        (
            label_x + text_width + 6,
            label_y + 3
        ),
        color,
        -1
    )

    # Text
    cv2.putText(
        image,
        text,
        (label_x + 3, label_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_detections(frame, pipeline_output):

    output = frame.copy()

    # ============================================
    # PPE detections
    # ============================================

    for detection in pipeline_output["ppe_detections"]:

        x1, y1, x2, y2 = detection.bbox

        color = (255, 165, 0)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f"{detection.class_name}: "
            f"{detection.confidence:.2f}"
        )

        draw_label(
            output,
            label,
            x1,
            y1,
            color,
            font_scale=0.5
        )

    # ============================================
    # Person detections
    # ============================================

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

        # ========================================
        # Person status label
        # ========================================

        if compliance.compliant:

            label = (
                f"ID {track.track_id} | "
                f"COMPLIANT"
            )

        else:

            missing = ", ".join(
                compliance.missing_ppe
            )

            label = (
                f"ID {track.track_id} | "
                f"MISSING: {missing}"
            )

        draw_label(
            output,
            label,
            x1,
            y1,
            box_color,
            font_scale=0.6
        )

    return output