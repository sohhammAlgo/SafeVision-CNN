from .person_detector import PersonDetector
from .ppe_detector import PPEDetector
from .association import associate_ppe_to_person
from .compliance import evaluate_compliance

from src.tracking.tracker import SimpleTracker
from src.alerts.alert_manager import AlertManager
from src.db.database import initialize_database
from src.db.repository import save_compliance_result


class SafetyPipeline:
    """
    Complete SafeVision detection pipeline.

    Flow:
        Person Detection
            ↓
        Person Tracking
            ↓
        PPE Detection
            ↓
        PPE Association
            ↓
        Compliance Evaluation
            ↓
        Alert Manager
            ↓
        SQLite
    """

    def __init__(
        self,
        person_model="yolov8n.pt",
        ppe_model=(
            "runs/detect/models/ppe/"
            "ppe_yolov8n/weights/best.pt"
        )
    ):
        self.person_detector = PersonDetector(
            person_model
        )

        self.ppe_detector = PPEDetector(
            ppe_model
        )

        self.tracker = SimpleTracker(
            iou_threshold=0.3,
            max_missed_frames=10
        )

        self.alert_manager = AlertManager(
            cooldown_seconds=10
        )

        initialize_database()

    def process(self, frame):
        """
        Process one image/frame.

        Returns:
            List of ComplianceResult objects.
        """

        # ----------------------------------------
        # 1. Detect persons
        # ----------------------------------------

        persons = self.person_detector.detect(
            frame,
            confidence=0.5
        )

        # ----------------------------------------
        # 2. Track persons
        # ----------------------------------------

        tracks = self.tracker.update(
            persons
        )

        # ----------------------------------------
        # 3. Detect PPE
        # ----------------------------------------

        ppe_detections = self.ppe_detector.detect(
            frame,
            confidence=0.5
        )

        results = []

        # ----------------------------------------
        # 4. Process each person
        # ----------------------------------------

        for person in tracks:

            associated_ppe = (
                associate_ppe_to_person(
                    person,
                    ppe_detections
                )
            )

            # ------------------------------------
            # 5. Compliance
            # ------------------------------------

            compliance = evaluate_compliance(
                person.track_id,
                associated_ppe
            )

            results.append(
                compliance
            )

            # ------------------------------------
            # 6. Alert decision
            # ------------------------------------

            should_alert = (
                self.alert_manager.should_alert(
                    track_id=person.track_id,
                    missing_ppe=compliance.missing_ppe
                )
            )

            # ------------------------------------
            # 7. Save violation
            # ------------------------------------

            if should_alert:
                save_compliance_result(
                    compliance
                )

        return results

    def process_frame(self, frame,confidence=0.5):
        """
        Process one frame and return detailed
        detection information.

        Used by the live camera interface.
        """

        # ----------------------------------------
        # 1. Detect persons
        # ----------------------------------------

        persons = self.person_detector.detect(
            frame,
            confidence=confidence
        )

        # ----------------------------------------
        # 2. Track persons
        # ----------------------------------------

        tracks = self.tracker.update(
            persons
        )

        # ----------------------------------------
        # 3. Detect PPE
        # ----------------------------------------

        ppe_detections = self.ppe_detector.detect(
            frame,
            confidence=confidence
        )

        person_results = []

        # ----------------------------------------
        # 4. Process each person
        # ----------------------------------------

        for person in tracks:

            associated_ppe = (
                associate_ppe_to_person(
                    person,
                    ppe_detections
                )
            )

            compliance = evaluate_compliance(
                person.track_id,
                associated_ppe
            )

            # ------------------------------------
            # Alert + database
            # ------------------------------------

            should_alert = (
                self.alert_manager.should_alert(
                    track_id=person.track_id,
                    missing_ppe=compliance.missing_ppe
                )
            )

            if should_alert:
                save_compliance_result(
                    compliance
                )

            person_results.append({
                "track": person,
                "compliance": compliance,
                "associated_ppe": associated_ppe
            })

        return {
            "persons": person_results,
            "ppe_detections": ppe_detections
        }