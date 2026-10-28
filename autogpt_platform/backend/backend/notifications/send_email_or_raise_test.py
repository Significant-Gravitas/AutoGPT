"""How the notification service answers an auth email Postmark refuses.

A recipient Postmark has marked inactive must reach the REST API as
`EmailUndeliverableError`, across the RPC, so the sign-up flow can skip it;
every other refusal still fails the send.
"""

import traceback
from unittest.mock import MagicMock

import pytest
from postmarker.exceptions import ClientError

from backend.notifications.email import EmailSender
from backend.notifications.notifications import (
    NotificationManager,
    NotificationManagerClient,
)
from backend.util.exceptions import EmailUndeliverableError
from backend.util.service import AppService, get_service_client

RECIPIENT = "someone@example.com"
RPC_TEST_PORT = 8771


@pytest.mark.asyncio
async def test_an_inactive_recipient_is_undeliverable_and_unnamed():
    manager = _manager(_refusing_postmark(406))

    with pytest.raises(EmailUndeliverableError) as raised:
        await manager.send_email_or_raise(RECIPIENT, "Verify", "<p>Verify</p>")

    # Postmark's message names the address; the traceback a log prints must not.
    assert RECIPIENT not in "".join(traceback.format_exception(raised.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [300, 500])
async def test_any_other_postmark_refusal_still_fails_the_send(code):
    manager = _manager(_refusing_postmark(code))

    with pytest.raises(ClientError) as raised:
        await manager.send_email_or_raise(RECIPIENT, "Verify", "<p>Verify</p>")

    assert raised.value.error_code == code


def test_an_inactive_recipient_crosses_the_rpc_as_itself():
    with _RefusingNotificationManager():
        get_service_client(
            _RefusingNotificationManagerClient, call_timeout=10, request_retry=True
        ).health_check()
        client = get_service_client(_RefusingNotificationManagerClient)

        with pytest.raises(EmailUndeliverableError):
            client.send_email_or_raise(RECIPIENT, "Verify", "<p>Verify</p>")


def _manager(postmark: MagicMock) -> NotificationManager:
    manager = NotificationManager.__new__(NotificationManager)
    manager.email_sender = EmailSender.__new__(EmailSender)
    manager.email_sender.postmark = postmark
    return manager


def _refusing_postmark(code: int) -> MagicMock:
    postmark = MagicMock()
    postmark.emails.send.side_effect = ClientError(
        f"You tried to send to recipient(s) that have been marked as inactive. "
        f"Found inactive addresses: {RECIPIENT}.",
        error_code=code,
    )
    return postmark


class _RefusingNotificationManager(NotificationManager):
    """The real service and its RPC routes, without RabbitMQ, behind a
    Postmark that refuses the recipient."""

    # Routes are built from the class's own attributes, so name the real method here.
    send_email_or_raise = NotificationManager.send_email_or_raise

    @classmethod
    def get_port(cls) -> int:
        return RPC_TEST_PORT

    async def health_check(self) -> str:
        return "OK"

    def run_service(self):
        # Set in the child process: a mock does not survive the pickle there.
        self.email_sender.postmark = _refusing_postmark(406)
        AppService.run_service(self)


class _RefusingNotificationManagerClient(NotificationManagerClient):
    @classmethod
    def get_service_type(cls):
        return _RefusingNotificationManager
