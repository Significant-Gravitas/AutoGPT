"""The Postmark envelope preserves sender, stream and unsubscribe semantics."""

from unittest.mock import MagicMock

import pytest
from prisma.enums import NotificationType

from backend.notifications import email as delivery
from backend.notifications.renderer import RenderedEmail, build_urls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "notification_type,sender_key,reply_to_key",
    [
        (
            NotificationType.PAYMENT_FAILED,
            "billing_sender_email",
            "billing_reply_to_email",
        ),
        (NotificationType.ALERT, "product_sender_email", "product_reply_to_email"),
        (NotificationType.BRIEFING, "product_sender_email", "product_reply_to_email"),
        (NotificationType.VERDICT, "product_sender_email", "product_reply_to_email"),
    ],
)
async def test_reply_to_preserves_configured_sender_and_stream(
    monkeypatch, notification_type, sender_key, reply_to_key
):
    monkeypatch.setattr(
        delivery.settings.config, sender_key, "AutoGPT <sender@example.org>"
    )
    monkeypatch.setattr(delivery.settings.config, reply_to_key, "support@example.org")
    monkeypatch.setattr(
        delivery.settings.config, "postmark_transactional_stream", "service"
    )
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = MagicMock()
    urls = build_urls("https://example.org/unsubscribe?token=abc")
    message = RenderedEmail(
        subject="Account update", preheader="News", html="<p>News</p>", text="News"
    )

    await sender._deliver(notification_type, "user@example.org", message, urls)

    envelope = sender.postmark.emails.send.call_args.kwargs
    assert envelope["From"] == "AutoGPT <sender@example.org>"
    assert envelope["ReplyTo"] == "support@example.org"
    assert envelope["MessageStream"] == "service"
    assert envelope["HtmlBody"] == message.html
    assert envelope["TextBody"] == message.text
    if notification_type == NotificationType.PAYMENT_FAILED:
        assert not envelope["Headers"]
    else:
        assert envelope["Headers"] == {
            "List-Unsubscribe": f"<{urls.unsubscribe}>",
            "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
        }


def test_auth_delivery_includes_text_without_unsubscribe(monkeypatch):
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = MagicMock()
    monkeypatch.setattr(
        delivery.settings.config, "postmark_sender_email", "AutoGPT <auth@example.org>"
    )
    monkeypatch.setattr(
        delivery.settings.config, "postmark_transactional_stream", "service"
    )

    sender.send_email_or_raise(
        "user@example.org",
        "Verify email",
        "<p>Verify</p>",
        "Verify: https://example.org/verify?token=abc",
    )

    envelope = sender.postmark.emails.send.call_args.kwargs
    assert envelope["From"] == "AutoGPT <auth@example.org>"
    assert envelope["MessageStream"] == "service"
    assert envelope["TextBody"] == "Verify: https://example.org/verify?token=abc"
    assert not envelope.get("Headers")
    assert not envelope.get("ReplyTo")


@pytest.mark.asyncio
async def test_ops_keeps_its_sender_without_reply_to_or_unsubscribe(monkeypatch):
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = MagicMock()
    monkeypatch.setattr(
        delivery.settings.config, "ops_sender_email", "Ops <ops@example.org>"
    )

    await sender._deliver(
        NotificationType.OPS,
        "refunds@example.org",
        RenderedEmail(
            subject="Refund", preheader="Refund", html="<p>Refund</p>", text="Refund"
        ),
        build_urls("https://example.org/unsubscribe"),
    )

    envelope = sender.postmark.emails.send.call_args.kwargs
    assert envelope["From"] == "Ops <ops@example.org>"
    assert not envelope.get("ReplyTo")
    assert not envelope.get("Headers")


def test_auth_delivery_reports_missing_configuration():
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = None

    with pytest.raises(RuntimeError, match="Postmark is not configured"):
        sender.send_email_or_raise("user@example.org", "Verify email", "<p>Verify</p>")
