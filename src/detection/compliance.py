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

    helmet = (
        associated_ppe["helmet"]
        is not None
    )

    mask = (
        associated_ppe["mask"]
        is not None
    )

    vest = (
        associated_ppe["vest"]
        is not None
    )

    compliant = (
        helmet
        and mask
        and vest
    )

    return ComplianceResult(
        track_id=track_id,
        helmet=helmet,
        mask=mask,
        vest=vest,
        compliant=compliant
    )