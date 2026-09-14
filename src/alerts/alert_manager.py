import time


class AlertManager:
    """
    Prevents repeated alerts for the same person
    and violation during a cooldown period.
    """

    def __init__(self, cooldown_seconds=10):
        self.cooldown_seconds = cooldown_seconds

        # (track_id, violation) -> last alert time
        self.last_alerts = {}

    def should_alert(
        self,
        track_id,
        missing_ppe
    ):
        """
        Returns True when a new alert should be generated.
        """

        if not missing_ppe:
            return False

        violation = ",".join(
            sorted(missing_ppe)
        )

        key = (
            track_id,
            violation
        )

        current_time = time.time()

        last_time = self.last_alerts.get(
            key
        )

        # First occurrence
        if last_time is None:
            self.last_alerts[key] = current_time
            return True

        # Cooldown check
        elapsed = (
            current_time - last_time
        )

        if elapsed >= self.cooldown_seconds:
            self.last_alerts[key] = current_time
            return True

        return False

    def reset_track(self, track_id):
        """
        Remove alert history for a track.
        """

        keys_to_remove = [
            key
            for key in self.last_alerts
            if key[0] == track_id
        ]

        for key in keys_to_remove:
            del self.last_alerts[key]