"""The Postmark envelope preserves sender, stream and unsubscribe semantics."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma.enums import NotificationType

from backend.notifications import email as delivery
from backend.notifications.notifications import (
    NotificationManager,
    NotificationManagerClient,
)
from backend.notifications.renderer import RenderedEmail, build_urls
from backend.util.service import get_service_client


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
    assert envelope["Tag"] == notification_type.value.lower()
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
    assert envelope["Tag"] == "auth"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_notification_tracking_follows_the_setting(monkeypatch, enabled):
    monkeypatch.setattr(
        delivery.settings.config, "postmark_track_opens_and_clicks", enabled
    )
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = MagicMock()

    await sender._deliver(
        NotificationType.BRIEFING,
        "user@example.org",
        RenderedEmail(subject="Brief", preheader="Brief", html="<p>B</p>", text="B"),
        build_urls("https://example.org/unsubscribe"),
    )

    envelope = sender.postmark.emails.send.call_args.kwargs
    assert envelope["TrackOpens"] is enabled
    assert envelope["TrackLinks"] == ("HtmlOnly" if enabled else "None")


def test_auth_mail_is_never_tracked(monkeypatch):
    """Its links carry sign-in tokens, which link tracking would hand to the
    click webhook."""
    monkeypatch.setattr(
        delivery.settings.config, "postmark_track_opens_and_clicks", True
    )
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = MagicMock()

    sender.send_email_or_raise("user@example.org", "Reset", "<p>Reset</p>")

    envelope = sender.postmark.emails.send.call_args.kwargs
    assert envelope["TrackOpens"] is False
    assert envelope["TrackLinks"] == "None"


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


def test_one_off_delivery_forwards_reply_to_when_given(monkeypatch):
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = MagicMock()
    monkeypatch.setattr(
        delivery.settings.config, "postmark_sender_email", "AutoGPT <auth@example.org>"
    )

    sender.send_email_or_raise(
        "user@example.org",
        "Your AutoGPT account is deleted",
        "<p>Deleted</p>",
        "Deleted",
        reply_to="contact@example.org",
    )

    envelope = sender.postmark.emails.send.call_args.kwargs
    assert envelope["ReplyTo"] == "contact@example.org"
    assert envelope["From"] == "AutoGPT <auth@example.org>"
    assert not envelope.get("Headers")


@pytest.mark.asyncio
async def test_manager_passes_reply_to_through_to_postmark():
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = MagicMock()
    manager = NotificationManager.__new__(NotificationManager)
    manager.email_sender = sender

    await manager.send_email_or_raise(
        "user@example.org",
        "Your AutoGPT account is deleted",
        "<p>Deleted</p>",
        "Deleted",
        reply_to="contact@example.org",
    )

    envelope = sender.postmark.emails.send.call_args.kwargs
    assert envelope["To"] == "user@example.org"
    assert envelope["TextBody"] == "Deleted"
    assert envelope["ReplyTo"] == "contact@example.org"


def test_reply_to_crosses_the_rpc_boundary_as_a_keyword():
    """The deletion service calls the notification service over RPC with
    reply_to as a keyword. The real client stub must send it by name, the
    exposed endpoint must accept it, and a caller that leaves it out still
    gets the default."""
    manager = NotificationManager.__new__(NotificationManager)
    manager.email_sender = MagicMock()
    app = FastAPI()
    app.post("/send_email_or_raise")(
        manager._create_fastapi_endpoint(NotificationManager.send_email_or_raise)
    )
    client = get_service_client(NotificationManagerClient)
    vars(client)["_sync_clients"][NotificationManager.__name__] = TestClient(app)

    client.send_email_or_raise(
        "user@example.org", "S", "<p>B</p>", reply_to="contact@example.org"
    )
    assert manager.email_sender.send_email_or_raise.call_args.args == (
        "user@example.org",
        "S",
        "<p>B</p>",
        None,
        "contact@example.org",
    )

    client.send_email_or_raise("user@example.org", "S", "<p>B</p>")
    assert manager.email_sender.send_email_or_raise.call_args.args[-1] is None
