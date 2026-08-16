def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def box_area(box):
    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a, box_b):
    intersection = intersection_area(
        box_a,
        box_b
    )

    union = (
        box_area(box_a)
        + box_area(box_b)
        - intersection
    )

    if union == 0:
        return 0.0

    return intersection / union


def center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )