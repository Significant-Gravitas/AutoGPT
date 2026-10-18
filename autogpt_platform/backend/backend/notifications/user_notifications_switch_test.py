"""ENABLE_USER_NOTIFICATIONS: auth mail only, every user notification off.

Prod can turn on Postmark for verify-email, reset-password and change-email
before product has signed off on the notification families. With the switch
off every user notification is dropped at the consumer, and the auth path
still sends.
"""

import logging
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.notifications import email as email_module
from backend.notifications import notifications as delivery
from backend.util.settings import Config

USER_FAMILIES = [t for t in NotificationType if t is not NotificationType.OPS]


def _manager(sender: AsyncMock) -> delivery.NotificationManager:
    manager = delivery.NotificationManager.__new__(delivery.NotificationManager)
    manager.email_sender = MagicMock(
        send_notification=sender, send_email_or_raise=MagicMock()
    )
    return manager


def _event(notification_type: NotificationType) -> SimpleNamespace:
    return SimpleNamespace(type=notification_type, user_id="user-1", data=MagicMock())


# `raising=False` so that on a build without the switch these fail on what was
# sent, not in setup.
@pytest.fixture
def notifications_off(monkeypatch):
    monkeypatch.setattr(
        delivery.settings.config, "enable_user_notifications", False, raising=False
    )


@pytest.fixture
def notifications_on(monkeypatch):
    monkeypatch.setattr(
        delivery.settings.config, "enable_user_notifications", True, raising=False
    )


@contextmanager
def _consumer(manager, notification_type, db_client):
    """Everything around the user consumer stubbed to its "send it" answer, so
    the switch is the only thing that can stop the email."""
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                manager, "_parse_message", return_value=_event(notification_type)
            )
        )
        stack.enter_context(
            patch.object(
                delivery, "get_database_manager_async_client", return_value=db_client
            )
        )
        stack.enter_context(patch.object(delivery, "TrialUpdateData"))
        stack.enter_context(
            patch.object(delivery, "generate_unsubscribe_link", return_value="u")
        )
        stack.enter_context(
            patch.object(delivery, "generate_preference_link", return_value="p")
        )
        yield SimpleNamespace(
            disposition=stack.enter_context(
                patch.object(
                    delivery,
                    "trial_notice_disposition",
                    AsyncMock(return_value="current"),
                )
            ),
            cap=stack.enter_context(
                patch.object(delivery, "claim_daily_send", AsyncMock(return_value=True))
            ),
        )


def test_the_switch_defaults_on_so_existing_deploys_are_unchanged():
    assert Config.model_fields["enable_user_notifications"].default is True


def test_the_switch_reads_from_the_environment(monkeypatch):
    monkeypatch.setenv("ENABLE_USER_NOTIFICATIONS", "false")
    assert Config().enable_user_notifications is False


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_type", USER_FAMILIES)
async def test_every_user_family_is_dropped_when_switched_off(
    notification_type, db_client, notifications_off, caplog
):
    sender = AsyncMock()
    manager = _manager(sender)
    with _consumer(manager, notification_type, db_client) as mocks:
        with caplog.at_level(logging.INFO, logger=delivery.__name__):
            handled = await manager._process_user_notification("{}")

    # Acked, not retried or dead-lettered: dropped, not deferred.
    assert handled is True
    sender.assert_not_awaited()
    # Nothing downstream ran: no daily-cap slot claimed, no trial retry.
    mocks.cap.assert_not_awaited()
    mocks.disposition.assert_not_awaited()
    db_client.get_user_notification_preference.assert_not_awaited()
    assert f"Dropping {notification_type}" in caplog.text
    assert "ENABLE_USER_NOTIFICATIONS is off" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_type", USER_FAMILIES)
async def test_every_user_family_still_sends_when_switched_on(
    notification_type, db_client, notifications_on
):
    sender = AsyncMock()
    manager = _manager(sender)
    with _consumer(manager, notification_type, db_client):
        assert await manager._process_user_notification("{}") is True

    sender.assert_awaited_once()
    assert sender.await_args.kwargs["notification_type"] == notification_type


@pytest.mark.asyncio
async def test_ops_mail_to_the_refunds_team_is_not_a_user_notification(
    notifications_off,
):
    sender = AsyncMock()
    manager = _manager(sender)
    with patch.object(
        manager, "_parse_message", return_value=_event(NotificationType.OPS)
    ):
        assert await manager._process_ops_notification("{}") is True

    sender.assert_awaited_once()


@pytest.mark.asyncio
async def test_auth_mail_still_reaches_postmark_when_switched_off(
    notifications_off,
):
    """Verify email, reset password and change email all come through
    `send_email_or_raise`; the switch must never touch it."""
    postmark = MagicMock()
    sender = email_module.EmailSender.__new__(email_module.EmailSender)
    sender.postmark = postmark
    manager = delivery.NotificationManager.__new__(delivery.NotificationManager)
    manager.email_sender = sender

    await manager.send_email_or_raise("sam@example.com", "Reset", "<p>link</p>")

    postmark.emails.send.assert_called_once()
    assert postmark.emails.send.call_args.kwargs["To"] == "sam@example.com"
