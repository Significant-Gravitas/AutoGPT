"""The stalled-erasure check: quiet while every claimed erasure finished,
pages with user IDs only for one claimed hours ago and not finished, and
pages for each one at most once a day."""

from unittest.mock import MagicMock

import pytest

from backend.monitoring import account_erasure_monitor as monitor
from backend.util.settings import Config

STALLED = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
ALSO_STALLED = "9d4f6a2b-1c3e-4b5a-8f7d-2e6c0a9b1d34"


@pytest.fixture
def wiring(mocker):
    db = MagicMock()
    db.stalled_account_erasures.return_value = []
    notifications = MagicMock()
    mocker.patch.object(monitor, "get_database_manager_client", return_value=db)
    mocker.patch.object(
        monitor, "get_notification_manager_client", return_value=notifications
    )
    sentry = mocker.patch.object(monitor, "sentry_capture_error")
    mocker.patch.dict(monitor._paged_at, clear=True)
    mocker.patch.object(monitor.config, "account_erasure_stall_alert_secs", 3 * 3600)
    return db, notifications, sentry


def test_it_pages_after_three_hours_by_default():
    assert Config.model_fields["account_erasure_stall_alert_secs"].default == 3 * 3600


def test_quiet_when_no_erasure_is_stalled(wiring):
    db, notifications, sentry = wiring

    result = monitor.report_stalled_account_erasures()

    assert "No stalled" in result
    db.stalled_account_erasures.assert_called_once_with(older_than_secs=3 * 3600)
    notifications.discord_system_alert.assert_not_called()
    sentry.assert_not_called()


def test_a_stalled_erasure_pages_with_its_user_id(wiring):
    db, notifications, sentry = wiring
    db.stalled_account_erasures.return_value = [STALLED, ALSO_STALLED]

    monitor.report_stalled_account_erasures()

    sentry.assert_called_once()
    error = sentry.call_args.args[0]
    assert isinstance(error, monitor.StalledAccountErasureException)
    # One Sentry issue for every stall: no IDs or counts in its text.
    assert STALLED not in str(error) and not str(error)[0].isdigit()
    message = notifications.discord_system_alert.call_args.args[0]
    assert message.startswith("2 account erasure(s) started over 3h ago")
    assert STALLED in message and ALSO_STALLED in message
    assert "@" not in message


def test_the_same_stall_pages_once_a_day_and_a_new_one_pages_again(wiring, mocker):
    db, notifications, sentry = wiring
    clock = mocker.patch.object(monitor, "monotonic", return_value=1000.0)
    db.stalled_account_erasures.return_value = [STALLED]

    monitor.report_stalled_account_erasures()
    monitor.report_stalled_account_erasures()
    assert notifications.discord_system_alert.call_count == 1

    # A new stall pages for itself alone, not the one paged already.
    db.stalled_account_erasures.return_value = [STALLED, ALSO_STALLED]
    monitor.report_stalled_account_erasures()
    assert notifications.discord_system_alert.call_count == 2
    page = notifications.discord_system_alert.call_args.args[0]
    assert page.startswith("1 account erasure(s)")
    assert ALSO_STALLED in page and STALLED not in page

    clock.return_value = 1000.0 + monitor._REPAGE_SECS
    monitor.report_stalled_account_erasures()
    assert notifications.discord_system_alert.call_count == 3
    assert sentry.call_count == 3


def test_a_page_that_failed_is_sent_again_on_the_next_check(wiring, caplog):
    """The IDs are logged even when the page fails."""
    db, notifications, _ = wiring
    db.stalled_account_erasures.return_value = [STALLED]
    notifications.discord_system_alert.side_effect = [ConnectionError("down"), None]

    with pytest.raises(ConnectionError):
        monitor.report_stalled_account_erasures()
    assert STALLED in caplog.text
    monitor.report_stalled_account_erasures()

    assert notifications.discord_system_alert.call_count == 2
