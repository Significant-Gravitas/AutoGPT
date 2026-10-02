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
    return db, notifications, sentry


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
