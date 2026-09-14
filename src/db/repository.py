import json
from datetime import datetime

from .database import get_connection
from src.detection.types import ComplianceResult


def save_compliance_result(
    result: ComplianceResult
):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO violations (
            timestamp,
            track_id,
            helmet,
            mask,
            vest,
            compliant,
            missing_ppe,
            severity
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),

            result.track_id,

            int(result.helmet),
            int(result.mask),
            int(result.vest),

            int(result.compliant),

            json.dumps(result.missing_ppe),

            result.severity
        )
    )

    connection.commit()
    connection.close()


def get_recent_violations(limit=50):
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM violations
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return rows

def get_violation_statistics():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) AS high,
            SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) AS critical
        FROM violations
        """
    )

    summary = cursor.fetchone()

    cursor.execute(
        """
        SELECT missing_ppe, COUNT(*) AS count
        FROM violations
        GROUP BY missing_ppe
        ORDER BY count DESC
        """
    )

    ppe_rows = cursor.fetchall()

    connection.close()

    return summary, ppe_rows