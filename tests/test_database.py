from src.db.database import initialize_database
from src.db.repository import (
    save_compliance_result,
    get_recent_violations
)

from src.detection.types import ComplianceResult


def test_database_storage():

    initialize_database()

    result = ComplianceResult(
        track_id=1,

        helmet=True,
        mask=False,
        vest=True,

        compliant=False,

        missing_ppe=["mask"],

        severity="HIGH"
    )

    save_compliance_result(result)

    rows = get_recent_violations(
        limit=1
    )

    assert len(rows) == 1

    row = rows[0]

    assert row["track_id"] == 1
    assert row["helmet"] == 1
    assert row["mask"] == 0
    assert row["vest"] == 1
    assert row["compliant"] == 0
    assert row["severity"] == "HIGH"