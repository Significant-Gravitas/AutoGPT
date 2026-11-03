from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import VerdictData
from backend.notifications.email import EmailSender
from backend.notifications.renderer import RenderedEmail

SendPath = Literal["auth", "notification", "owner"]
SEND_PATHS: tuple[SendPath, ...] = ("auth", "notification", "owner")


@pytest.fixture
def postmark() -> MagicMock:
    return MagicMock()


@pytest.fixture
def api_post() -> AsyncMock:
    """The Postmark API call notifications are sent with (auth and owner mail
    go through the postmarker client)."""
    response = MagicMock()
    response.json.return_value = {"ErrorCode": 0, "MessageID": "message-1"}
    return AsyncMock(return_value=response)


@pytest.fixture
def sender(postmark: MagicMock, api_post: AsyncMock) -> EmailSender:
    sender = EmailSender.__new__(EmailSender)
    sender.postmark = postmark
    sender.server_token = "test-token"
    sender._http_client = MagicMock(post=api_post)
    return sender


def _delivered_to(postmark: MagicMock, api_post: AsyncMock) -> list[str]:
    return [call.kwargs["To"] for call in postmark.emails.send.call_args_list] + [
        call.kwargs["json"]["To"] for call in api_post.await_args_list
    ]


@pytest.mark.parametrize("path", SEND_PATHS)
@pytest.mark.parametrize(
    "recipient",
    [
        "test@previews.agpt.co",
        "test+run-123@previews.agpt.co",
        "Test@PREVIEWS.AGPT.CO",
        "  test@previews.agpt.co \t",
    ],
)
async def test_preview_recipients_never_reach_postmark(
    sender: EmailSender,
    postmark: MagicMock,
    api_post: AsyncMock,
    path: SendPath,
    recipient: str,
):
    await _send(sender, path, recipient)
    assert _delivered_to(postmark, api_post) == []


@pytest.mark.parametrize("path", SEND_PATHS)
@pytest.mark.parametrize(
    "recipient",
    [
        "sam@example.com",
        "sam@agpt.co",
        "sam@notpreviews.agpt.co",
        "sam@previews.agpt.com",
        "sam@previews.agpt.co.example.com",
        "sam@sub.previews.agpt.co",
    ],
)
async def test_other_recipients_still_reach_postmark(
    sender: EmailSender,
    postmark: MagicMock,
    api_post: AsyncMock,
    path: SendPath,
    recipient: str,
):
    await _send(sender, path, recipient)
    assert _delivered_to(postmark, api_post) == [recipient]


@pytest.mark.parametrize("path", SEND_PATHS)
async def test_real_recipient_provider_errors_still_propagate(
    sender: EmailSender, postmark: MagicMock, api_post: AsyncMock, path: SendPath
):
    postmark.emails.send.side_effect = RuntimeError("Provider unavailable")
    api_post.side_effect = RuntimeError("Provider unavailable")
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        await _send(sender, path, "sam@example.com")


def test_preview_auth_mail_needs_no_postmark_configuration(sender: EmailSender):
    sender.postmark = None
    sender.send_email_or_raise("test@previews.agpt.co", "Reset", "<p>Reset</p>")


async def _send(sender: EmailSender, path: SendPath, recipient: str) -> None:
    if path == "auth":
        sender.send_email_or_raise(recipient, "Reset", "<p>Reset</p>", "Reset")
    elif path == "owner":
        await sender.send_owner_email(
            recipient,
            RenderedEmail(
                subject="Result", preheader="", html="<p>Result</p>", text="Result"
            ),
            attachments=[],
        )
    else:
        await sender.send_notification(
            NotificationType.VERDICT,
            recipient,
            VerdictData(
                outcome="approved",
                agent_name="Example workflow",
                reviewer_name="Reviewer",
                reviewed_at_label="Today",
            ),
            "https://example.com/unsubscribe",
        )
