from .types import ComplianceResult


REQUIRED_PPE = {
    "helmet",
    "mask",
    "vest"
}


def evaluate_compliance(
    track_id,
    associated_ppe
):
    """
    Evaluate PPE compliance for one tracked person.

    Returns:
        ComplianceResult containing:
        - PPE status
        - missing PPE
        - severity
        - overall compliance
    """

    helmet = (
        associated_ppe["helmet"] is not None
    )

    mask = (
        associated_ppe["mask"] is not None
    )

    vest = (
        associated_ppe["vest"] is not None
    )

    # ----------------------------------------
    # Find missing PPE
    # ----------------------------------------

    missing_ppe = []

    if not helmet:
        missing_ppe.append("helmet")

    if not mask:
        missing_ppe.append("mask")

    if not vest:
        missing_ppe.append("vest")

    # ----------------------------------------
    # Overall compliance
    # ----------------------------------------

    compliant = len(missing_ppe) == 0

    # ----------------------------------------
    # Severity
    # ----------------------------------------

    if compliant:
        severity = "NONE"

    elif len(missing_ppe) == 1:
        severity = "HIGH"

    else:
        severity = "CRITICAL"

    # ----------------------------------------
    # Result
    # ----------------------------------------

    return ComplianceResult(
        track_id=track_id,

        helmet=helmet,
        mask=mask,
        vest=vest,

        compliant=compliant,

        missing_ppe=missing_ppe,

        severity=severity
    )