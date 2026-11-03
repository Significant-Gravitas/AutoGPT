import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.notifications.email import EmailSender


def _sender(response: MagicMock) -> tuple[EmailSender, AsyncMock]:
    post = AsyncMock(return_value=response)
    sender = EmailSender.__new__(EmailSender)
    sender.server_token = "test-token"
    sender.postmark = MagicMock()
    sender._http_client = MagicMock(post=post)
    return sender, post


def _response(error_code: int = 0) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"ErrorCode": error_code, "MessageID": "message-1"}
    return response


@pytest.mark.asyncio
async def test_notification_send_uses_cancellable_postmark_api() -> None:
    sender, post = _sender(_response())

    await sender._send(
        "sam@example.com",
        "briefing",
        "AutoGPT <hello@example.com>",
        "Subject",
        "<p>Body</p>",
        "Body",
        {"List-Unsubscribe": "<https://example.com/unsubscribe>"},
    )

    request = post.await_args
    assert request.args == ("https://api.postmarkapp.com/email",)
    assert request.kwargs["headers"]["X-Postmark-Server-Token"] == "test-token"
    assert request.kwargs["json"]["Headers"] == [
        {
            "Name": "List-Unsubscribe",
            "Value": "<https://example.com/unsubscribe>",
        }
    ]


@pytest.mark.asyncio
async def test_http_200_with_postmark_error_is_a_delivery_failure() -> None:
    sender, _ = _sender(_response(error_code=406))

    with pytest.raises(RuntimeError, match="Postmark rejected"):
        await sender._send("to", "tag", "from", "subject", "html", "text", None)


@pytest.mark.asyncio
async def test_http_failure_is_a_delivery_failure() -> None:
    response = _response()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "unavailable",
        request=httpx.Request("POST", "https://api.postmarkapp.com/email"),
        response=httpx.Response(503),
    )
    sender, _ = _sender(response)

    with pytest.raises(httpx.HTTPStatusError):
        await sender._send("to", "tag", "from", "subject", "html", "text", None)


@pytest.mark.asyncio
async def test_unconfigured_postmark_never_reports_success() -> None:
    sender, post = _sender(_response())
    sender.server_token = None

    with pytest.raises(RuntimeError, match="not configured"):
        await sender._send("to", "tag", "from", "subject", "html", "text", None)

    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_in_flight_send_observes_cancellation() -> None:
    cancelled = asyncio.Event()

    async def stalled_post(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    sender = EmailSender.__new__(EmailSender)
    sender.server_token = "test-token"
    sender.postmark = MagicMock()
    sender._http_client = MagicMock(post=AsyncMock(side_effect=stalled_post))

    task = asyncio.create_task(
        sender._send("to", "tag", "from", "subject", "html", "text", None)
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()


"""The Postmark envelope preserves sender, stream and unsubscribe semantics."""

from unittest.mock import MagicMock

import pytest
from prisma.enums import NotificationType

from backend.notifications import email as delivery
from backend.notifications.renderer import RenderedEmail, build_urls


def _api_sender() -> tuple[delivery.EmailSender, AsyncMock]:
    """A sender whose notifications go to a stubbed Postmark API; auth mail
    still goes through the stubbed postmarker client."""
    response = MagicMock()
    response.json.return_value = {"ErrorCode": 0, "MessageID": "message-1"}
    post = AsyncMock(return_value=response)
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.server_token = "test-token"
    sender.postmark = MagicMock()
    sender._http_client = MagicMock(post=post)
    return sender, post


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
    sender, post = _api_sender()
    urls = build_urls("https://example.org/unsubscribe?token=abc")
    message = RenderedEmail(
        subject="Account update", preheader="News", html="<p>News</p>", text="News"
    )

    await sender._deliver(notification_type, "user@example.org", message, urls)

    envelope = post.await_args.kwargs["json"]
    assert envelope["From"] == "AutoGPT <sender@example.org>"
    assert envelope["ReplyTo"] == "support@example.org"
    assert envelope["MessageStream"] == "service"
    assert envelope["HtmlBody"] == message.html
    assert envelope["TextBody"] == message.text
    assert envelope["Tag"] == notification_type.value.lower()
    if notification_type == NotificationType.PAYMENT_FAILED:
        assert not envelope.get("Headers")
    else:
        assert envelope["Headers"] == [
            {"Name": "List-Unsubscribe-Post", "Value": "List-Unsubscribe=One-Click"},
            {"Name": "List-Unsubscribe", "Value": f"<{urls.unsubscribe}>"},
        ]


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
    sender, post = _api_sender()

    await sender._deliver(
        NotificationType.BRIEFING,
        "user@example.org",
        RenderedEmail(subject="Brief", preheader="Brief", html="<p>B</p>", text="B"),
        build_urls("https://example.org/unsubscribe"),
    )

    envelope = post.await_args.kwargs["json"]
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
    sender, post = _api_sender()
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

    envelope = post.await_args.kwargs["json"]
    assert envelope["From"] == "Ops <ops@example.org>"
    assert not envelope.get("ReplyTo")
    assert not envelope.get("Headers")


def test_auth_delivery_reports_missing_configuration():
    sender = delivery.EmailSender.__new__(delivery.EmailSender)
    sender.postmark = None

    with pytest.raises(RuntimeError, match="Postmark is not configured"):
        sender.send_email_or_raise("user@example.org", "Verify email", "<p>Verify</p>")
