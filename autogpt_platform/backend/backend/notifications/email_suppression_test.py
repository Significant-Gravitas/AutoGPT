from typing import Literal
from unittest.mock import MagicMock

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
def sender(postmark: MagicMock) -> EmailSender:
    sender = EmailSender.__new__(EmailSender)
    sender.postmark = postmark
    return sender


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
    sender: EmailSender, postmark: MagicMock, path: SendPath, recipient: str
):
    await _send(sender, path, recipient)
    postmark.emails.send.assert_not_called()


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
    sender: EmailSender, postmark: MagicMock, path: SendPath, recipient: str
):
    await _send(sender, path, recipient)
    postmark.emails.send.assert_called_once()
    assert postmark.emails.send.call_args.kwargs["To"] == recipient


@pytest.mark.parametrize("path", SEND_PATHS)
async def test_real_recipient_provider_errors_still_propagate(
    sender: EmailSender, postmark: MagicMock, path: SendPath
):
    postmark.emails.send.side_effect = RuntimeError("Provider unavailable")
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
