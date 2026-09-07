from unittest.mock import MagicMock

import pytest

from backend.notifications.email import EmailSender


async def send(sender):
    await sender._send(
        user_email="recipient@example.com",
        sender="sender@example.com",
        subject="Test notification",
        html_body="<p>Test</p>",
        text_body="Test",
        headers=None,
    )


@pytest.mark.asyncio
async def test_missing_provider_raises_instead_of_reporting_success():
    sender = EmailSender.__new__(EmailSender)
    sender.postmark = None
    with pytest.raises(RuntimeError, match="Postmark is not configured"):
        await send(sender)


@pytest.mark.asyncio
async def test_configured_provider_errors_reach_the_queue_consumer():
    sender = EmailSender.__new__(EmailSender)
    sender.postmark = MagicMock()
    sender.postmark.emails.send.side_effect = RuntimeError("Provider unavailable")
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        await send(sender)


@pytest.mark.asyncio
async def test_successful_provider_send_is_unchanged():
    sender = EmailSender.__new__(EmailSender)
    sender.postmark = MagicMock()
    await send(sender)
    sender.postmark.emails.send.assert_called_once()
