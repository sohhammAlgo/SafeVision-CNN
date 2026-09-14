from src.alerts.alert_manager import AlertManager


def test_alert_cooldown():

    manager = AlertManager(
        cooldown_seconds=10
    )

    # First violation should alert
    assert manager.should_alert(
        track_id=1,
        missing_ppe=["mask"]
    ) is True

    # Same violation immediately should not alert
    assert manager.should_alert(
        track_id=1,
        missing_ppe=["mask"]
    ) is False


def test_different_violation():

    manager = AlertManager(
        cooldown_seconds=10
    )

    # Missing mask
    assert manager.should_alert(
        track_id=1,
        missing_ppe=["mask"]
    ) is True

    # Different violation
    assert manager.should_alert(
        track_id=1,
        missing_ppe=["helmet"]
    ) is True