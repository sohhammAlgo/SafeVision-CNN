from .types import TrackedPerson


def convert_person_detections(persons):
    """
    Convert raw person detector outputs into TrackedPerson objects.

    Tracking is not implemented yet, so IDs are assigned
    sequentially for this frame.
    """

    tracked_persons = []

    for index, person in enumerate(persons):
        tracked_persons.append(
            TrackedPerson(
                track_id=index,
                bbox=person["bbox"]
            )
        )

    return tracked_persons