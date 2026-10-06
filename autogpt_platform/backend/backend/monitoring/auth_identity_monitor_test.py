from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from backend.data.user import OrphanedAuthIdentity, OrphanedAuthIdentityReport
from backend.monitoring import auth_identity_monitor as monitor_module
from backend.monitoring.auth_identity_monitor import (
    OrphanedAuthIdentityException,
    report_orphaned_auth_identities,
)


@pytest.fixture
def wiring(mocker):
    db = MagicMock()
    notifications = MagicMock()
    mocker.patch.object(monitor_module, "get_database_manager_client", return_value=db)
    mocker.patch.object(
        monitor_module, "get_notification_manager_client", return_value=notifications
    )
    sentry = mocker.patch.object(monitor_module, "sentry_capture_error")
    mocker.patch.dict(monitor_module._collisions_paged_at, clear=True)
    mocker.patch.object(
        monitor_module.config, "auth_identity_orphan_sweep_enabled", True
    )
    return db, notifications, sentry


def _collision(identity_id: str = "auth-new") -> OrphanedAuthIdentity:
    return OrphanedAuthIdentity(
        id=identity_id,
        email="taken@example.com",
        createdAt=datetime.now(timezone.utc),
        email_owner_id="user-old",
    )


def test_quiet_when_the_invariant_holds(wiring):
    db, notifications, sentry = wiring
    db.heal_orphaned_auth_identities.return_value = OrphanedAuthIdentityReport()

    result = report_orphaned_auth_identities()

    assert "No orphaned" in result
    notifications.discord_system_alert.assert_not_called()
    sentry.assert_not_called()
    # Grace window and batch size come from config, not hardcoded.
    kwargs = db.heal_orphaned_auth_identities.call_args.kwargs
    assert kwargs == {
        "grace_secs": monitor_module.config.auth_identity_orphan_grace_secs,
        "limit": monitor_module.config.auth_identity_orphan_check_limit,
    }


def test_the_sweep_is_off_by_default():
    assert monitor_module.Config().auth_identity_orphan_sweep_enabled is False


def test_heals_and_pages_nothing_while_switched_off(wiring, mocker):
    db, notifications, sentry = wiring
    mocker.patch.object(
        monitor_module.config, "auth_identity_orphan_sweep_enabled", False
    )

    result = report_orphaned_auth_identities()

    assert "disabled" in result
    db.heal_orphaned_auth_identities.assert_not_called()
    notifications.discord_system_alert.assert_not_called()
    sentry.assert_not_called()


def test_pages_when_it_had_to_heal_or_could_not(wiring):
    db, notifications, sentry = wiring
    collided = OrphanedAuthIdentity(
        id="auth-new",
        email="taken@example.com",
        createdAt=datetime.now(timezone.utc),
        email_owner_id="user-old",
    )
    db.heal_orphaned_auth_identities.return_value = OrphanedAuthIdentityReport(
        healed=["auth-1"], collided=[collided], failed=["auth-2"]
    )

    result = report_orphaned_auth_identities()

    # A heal is still an invariant breach: something upstream failed to
    # provision, and that has to be visible even though the user is fine now.
    sentry.assert_called_once()
    assert isinstance(sentry.call_args.args[0], OrphanedAuthIdentityException)
    notifications.discord_system_alert.assert_called_once_with(result)
    assert "Healed 1" in result and "auth-1" in result
    assert "auth-new (email owned by user-old)" in result
    assert "auth-2" in result
    # Ids only: no email addresses in an alert that goes to Discord/Sentry.
    assert "taken@example.com" not in result


def test_a_repeat_collision_pages_once_a_day(wiring, mocker):
    """A collision needs a human but comes back on every 15-minute sweep;
    paging each time would be ~96 alerts a day per account."""
    db, notifications, sentry = wiring
    clock = mocker.patch.object(monitor_module.time, "monotonic", return_value=1000.0)
    db.heal_orphaned_auth_identities.return_value = OrphanedAuthIdentityReport(
        collided=[_collision()]
    )
    warning = mocker.patch.object(monitor_module.logger, "warning")

    report_orphaned_auth_identities()
    report_orphaned_auth_identities()

    assert sentry.call_count == 1
    assert notifications.discord_system_alert.call_count == 1
    warning.assert_called_once()

    clock.return_value = 1000.0 + monitor_module._COLLISION_REPAGE_SECS
    report_orphaned_auth_identities()

    assert sentry.call_count == 2


def test_a_new_collision_still_pages_next_to_a_known_one(wiring):
    db, notifications, sentry = wiring
    db.heal_orphaned_auth_identities.return_value = OrphanedAuthIdentityReport(
        collided=[_collision("auth-a")]
    )
    report_orphaned_auth_identities()
    db.heal_orphaned_auth_identities.return_value = OrphanedAuthIdentityReport(
        collided=[_collision("auth-a"), _collision("auth-b")]
    )

    report_orphaned_auth_identities()

    assert sentry.call_count == 2


def test_a_heal_always_pages_even_beside_a_known_collision(wiring):
    db, notifications, sentry = wiring
    db.heal_orphaned_auth_identities.return_value = OrphanedAuthIdentityReport(
        collided=[_collision()]
    )
    report_orphaned_auth_identities()
    db.heal_orphaned_auth_identities.return_value = OrphanedAuthIdentityReport(
        healed=["auth-1"], collided=[_collision()]
    )

    report_orphaned_auth_identities()

    assert sentry.call_count == 2
