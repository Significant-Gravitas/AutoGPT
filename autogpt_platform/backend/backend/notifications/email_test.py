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
        await sender._send("to", "from", "subject", "html", "text", None)


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
        await sender._send("to", "from", "subject", "html", "text", None)


@pytest.mark.asyncio
async def test_unconfigured_postmark_never_reports_success() -> None:
    sender, post = _sender(_response())
    sender.server_token = None

    with pytest.raises(RuntimeError, match="not configured"):
        await sender._send("to", "from", "subject", "html", "text", None)

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
        sender._send("to", "from", "subject", "html", "text", None)
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()
