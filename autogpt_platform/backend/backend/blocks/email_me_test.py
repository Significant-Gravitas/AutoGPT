"""SendEmailToMeBlock run the way Otto runs it, with Postmark as the only stub.

Everything from the copilot's block runner down to the notification service's
sender is real, except the RPC hop to that service, which is called in-process.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks import email_me
from backend.blocks.email_me import SendEmailToMeBlock
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import BlockOutputResponse, SetupRequirementsResponse
from backend.copilot.tools.run_block import RunBlockTool
from backend.notifications import notifications, owner_email
from backend.notifications.email import EmailSender

OWNER = "owner-user"
OWNER_ADDRESS = "owner@example.com"
OTHER = "other-user"
OTHER_ADDRESS = "other@example.com"
BLOCK_ID = SendEmailToMeBlock().id


@pytest.fixture
def postmark():
    return MagicMock()


@pytest.fixture
def accounts():
    return {
        OWNER: SimpleNamespace(address=OWNER_ADDRESS, verified=True),
        OTHER: SimpleNamespace(address=OTHER_ADDRESS, verified=True),
    }


@pytest.fixture
def counter():
    return _FakeRedis()


@pytest.fixture(autouse=True)
def platform(monkeypatch, postmark, accounts, counter):
    """The mail service behind the block: real code, stubbed at Postmark, the
    account table and Redis."""
    sender = EmailSender.__new__(EmailSender)
    sender.postmark = postmark
    manager = notifications.NotificationManager.__new__(
        notifications.NotificationManager
    )
    manager.email_sender = sender
    monkeypatch.setattr(
        email_me,
        "get_notification_manager_client",
        lambda: SimpleNamespace(send_owner_email=manager.send_owner_email),
    )

    db = MagicMock()
    db.get_user_email_by_id = AsyncMock(
        side_effect=lambda uid: accounts[uid].address if uid in accounts else None
    )
    db.get_user_email_verification = AsyncMock(
        side_effect=lambda uid: accounts[uid].verified
    )
    monkeypatch.setattr(
        owner_email, "get_database_manager_async_client", lambda **_: db
    )
    monkeypatch.setattr(owner_email, "get_redis_async", AsyncMock(return_value=counter))
    # This mail is sent while prod has every platform notification switched off.
    monkeypatch.setattr(
        notifications.settings.config, "enable_user_notifications", False
    )


@pytest.fixture(autouse=True)
def copilot_runtime():
    reviews = MagicMock()
    reviews.get_pending_reviews_for_chat_session = AsyncMock(return_value=[])
    workspace = MagicMock()
    workspace.get_or_create_workspace = AsyncMock(return_value=MagicMock(id="ws"))
    users = MagicMock()
    users.get_user_by_id = AsyncMock(side_effect=ValueError)
    with (
        patch("backend.copilot.tools.helpers.review_db", return_value=reviews),
        patch("backend.copilot.tools.helpers.workspace_db", return_value=workspace),
        patch("backend.copilot.tools.helpers.user_db", return_value=users),
        # An account with nothing connected, so a block that wanted a
        # credential would get its sign-in card here rather than an RPC.
        patch(
            "backend.copilot.tools.utils.get_user_credentials",
            AsyncMock(return_value=[]),
        ),
        patch(
            "backend.copilot.tools.utils.selected_credentials",
            AsyncMock(return_value={}),
        ),
    ):
        yield


async def run(input_data: dict, user_id: str = OWNER):
    return await RunBlockTool()._execute(
        user_id=user_id,
        session=make_session(user_id=user_id),
        block_id=BLOCK_ID,
        input_data=input_data,
        dry_run=False,
    )


def sent_to(postmark) -> list[str]:
    return [call.kwargs["To"] for call in postmark.emails.send.call_args_list]


async def test_an_ask_in_chat_emails_the_owner_with_no_sign_in_card(postmark):
    response = await run({"subject": "Your digest", "body": "Three things."})

    assert not isinstance(response, SetupRequirementsResponse)
    assert isinstance(response, BlockOutputResponse)
    assert response.outputs == {"status": ["Email sent to your account address"]}
    assert sent_to(postmark) == [OWNER_ADDRESS]
    assert postmark.emails.send.call_args.kwargs["Subject"] == "Your digest"


@pytest.mark.parametrize(
    "extra",
    [
        {"to": OTHER_ADDRESS},
        {"to_email": OTHER_ADDRESS},
        {"recipient": OTHER_ADDRESS},
        {"email": OTHER_ADDRESS},
        {"user_id": OTHER},
    ],
)
async def test_no_input_sends_it_anywhere_but_the_owners_address(postmark, extra):
    await run({"subject": "s", "body": "b", **extra})

    assert set(sent_to(postmark)) <= {OWNER_ADDRESS}


async def test_an_unverified_address_gets_nothing(postmark, accounts):
    accounts[OWNER].verified = False

    response = await run({"subject": "s", "body": "b"})

    assert "isn't verified" in _error(response)
    assert sent_to(postmark) == []


async def test_the_daily_cap_stops_a_looping_run(monkeypatch, postmark):
    monkeypatch.setattr(owner_email.settings.config, "owner_email_daily_limit", 2)

    for _ in range(2):
        await run({"subject": "s", "body": "b"})
    response = await run({"subject": "s", "body": "b"})

    assert "the most allowed in a day" in _error(response)
    assert sent_to(postmark) == [OWNER_ADDRESS, OWNER_ADDRESS]


async def test_a_failed_send_does_not_use_up_the_days_allowance(monkeypatch, postmark):
    monkeypatch.setattr(owner_email.settings.config, "owner_email_daily_limit", 1)
    postmark.emails.send.side_effect = [ConnectionError("postmark down"), None]

    await run({"subject": "s", "body": "b"})
    response = await run({"subject": "s", "body": "b"})

    assert response.outputs == {"status": ["Email sent to your account address"]}


async def test_an_unreadable_daily_count_sends_nothing(postmark, counter):
    counter.broken = True

    response = await run({"subject": "s", "body": "b"})

    assert "Couldn't check today's email count" in _error(response)
    assert sent_to(postmark) == []


async def test_without_postmark_the_run_reports_a_failure_not_a_send():
    sender = EmailSender.__new__(EmailSender)
    sender.postmark = None
    manager = notifications.NotificationManager.__new__(
        notifications.NotificationManager
    )
    manager.email_sender = sender
    with patch.object(
        email_me,
        "get_notification_manager_client",
        lambda: SimpleNamespace(send_owner_email=manager.send_owner_email),
    ):
        response = await run({"subject": "s", "body": "b"})

    assert "isn't set up" in _error(response)


async def test_the_body_cannot_inject_markup_or_a_script_link(postmark):
    await run({"subject": "s", "body": '<a href="https://evil.example">click</a>'})
    response = await run({"subject": "s", "body": "b", "link": "javascript:alert(1)"})

    html = postmark.emails.send.call_args_list[0].kwargs["HtmlBody"]
    assert '<a href="https://evil.example">' not in html
    assert "&lt;a href=" in html
    assert "must be a web address" in _error(response)
    assert len(sent_to(postmark)) == 1


def _error(response) -> str:
    if isinstance(response, BlockOutputResponse):
        return " ".join(response.outputs.get("error", []))
    return str(getattr(response, "error", None) or response.message)


class _FakeRedis:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.broken = False

    async def incr(self, key: str) -> int:
        if self.broken:
            raise ConnectionError("redis down")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def decr(self, key: str) -> int:
        self.counts[key] -= 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        pass
