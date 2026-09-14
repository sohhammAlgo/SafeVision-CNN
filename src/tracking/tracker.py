from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Track:
    """
    Represents a tracked person.
    """

    track_id: int
    bbox: Tuple[int, int, int, int]
    missed_frames: int = 0


class SimpleTracker:
    """
    Simple IoU-based tracker for persons.

    This is an initial lightweight tracker.
    Later it can be replaced with ByteTrack/BoT-SORT
    for production-level tracking.
    """

    def __init__(
        self,
        iou_threshold=0.3,
        max_missed_frames=10
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

        self.tracks: List[Track] = []
        self.next_id = 1

    # --------------------------------------------
    # IoU
    # --------------------------------------------

    @staticmethod
    def iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)

        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        intersection_width = max(
            0,
            x2 - x1
        )

        intersection_height = max(
            0,
            y2 - y1
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area_a = max(
            0,
            ax2 - ax1
        ) * max(
            0,
            ay2 - ay1
        )

        area_b = max(
            0,
            bx2 - bx1
        ) * max(
            0,
            by2 - by1
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union == 0:
            return 0.0

        return intersection / union

    # --------------------------------------------
    # Update
    # --------------------------------------------

    def update(self, detections):
        """
        Update tracker with person detections.

        Args:
            detections:
                List of person detection dictionaries.

        Returns:
            List of Track objects.
        """

        updated_tracks = []
        matched_track_ids = set()
        matched_detection_ids = set()

        # ----------------------------------------
        # Match existing tracks to detections
        # ----------------------------------------

        for track in self.tracks:

            best_iou = 0.0
            best_detection_index = None

            for index, detection in enumerate(
                detections
            ):

                if index in matched_detection_ids:
                    continue

                score = self.iou(
                    track.bbox,
                    detection["bbox"]
                )

                if score > best_iou:
                    best_iou = score
                    best_detection_index = index

            # ------------------------------------
            # Successful match
            # ------------------------------------

            if (
                best_detection_index is not None
                and best_iou >= self.iou_threshold
            ):

                detection = detections[
                    best_detection_index
                ]

                track.bbox = detection["bbox"]
                track.missed_frames = 0

                updated_tracks.append(track)

                matched_track_ids.add(
                    track.track_id
                )

                matched_detection_ids.add(
                    best_detection_index
                )

            # ------------------------------------
            # Track temporarily missing
            # ------------------------------------

            else:

                track.missed_frames += 1

                if (
                    track.missed_frames
                    <= self.max_missed_frames
                ):
                    updated_tracks.append(track)

        # ----------------------------------------
        # Create tracks for new persons
        # ----------------------------------------

        for index, detection in enumerate(
            detections
        ):

            if index in matched_detection_ids:
                continue

            new_track = Track(
                track_id=self.next_id,
                bbox=detection["bbox"],
                missed_frames=0
            )

            self.next_id += 1

            updated_tracks.append(
                new_track
            )

        self.tracks = updated_tracks

        return self.tracks