"""SendEmailToMeBlock run the way Otto runs it, with Postmark as the only stub.

Everything from the copilot's block runner down to the notification service's
sender is real, except the RPC hop to that service, which is called in-process.
"""

import asyncio
import threading
from types import SimpleNamespace
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from postmarker.exceptions import ClientError

from backend.blocks import email_me
from backend.blocks.email_me import SendEmailToMeBlock
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import BlockOutputResponse, SetupRequirementsResponse
from backend.copilot.tools.run_block import RunBlockTool
from backend.data.execution import ExecutionContext
from backend.notifications import notifications, owner_email
from backend.notifications.email import EmailSender
from backend.util.exceptions import BlockExecutionError

REFUSED = ClientError("[406] Recipient is marked inactive", error_code=406)
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


async def test_a_send_postmark_refused_does_not_use_up_the_days_allowance(
    monkeypatch, postmark
):
    monkeypatch.setattr(owner_email.settings.config, "owner_email_daily_limit", 1)
    postmark.emails.send.side_effect = [REFUSED, None]

    refused = await run({"subject": "s", "body": "b"})
    response = await run({"subject": "s", "body": "b"})

    assert "The email service refused this email" in _error(refused)
    assert response.outputs == {"status": ["Email sent to your account address"]}


async def test_a_send_that_may_have_arrived_keeps_its_slot(monkeypatch, postmark):
    monkeypatch.setattr(owner_email.settings.config, "owner_email_daily_limit", 1)
    postmark.emails.send.side_effect = [requests.Timeout("read timed out"), None]

    unknown = await run({"subject": "s", "body": "b"})
    response = await run({"subject": "s", "body": "b"})

    assert "it may still arrive" in _error(unknown)
    assert "the most allowed in a day" in _error(response)


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


async def test_a_refused_send_does_not_lock_out_the_day_when_another_fails(
    monkeypatch, postmark
):
    """A refused at the cap while B is in flight, then B fails: the day's one
    slot must come back."""
    monkeypatch.setattr(owner_email.settings.config, "owner_email_daily_limit", 1)
    in_flight, release = threading.Event(), threading.Event()

    def postmark_send(**_):
        if not in_flight.is_set():
            in_flight.set()
            release.wait(5)
            raise REFUSED

    postmark.emails.send.side_effect = postmark_send
    first = asyncio.create_task(run({"subject": "s", "body": "b"}))
    await asyncio.to_thread(in_flight.wait, 5)
    refused = await run({"subject": "s", "body": "b"})
    release.set()
    await first
    retry = await run({"subject": "s", "body": "b"})

    assert "the most allowed in a day" in _error(refused)
    assert retry.outputs == {"status": ["Email sent to your account address"]}


@pytest.mark.parametrize(
    "body_format, body, shown",
    [
        ("plain", "Total: **3** <b>new</b>", "Total: **3** &lt;b&gt;new&lt;/b&gt;"),
        ("markdown", "Total: **3** new", "Total: <strong>3</strong> new"),
        (
            "html",
            "<p>Total: <strong>3</strong> new</p>",
            "Total: <strong>3</strong> new",
        ),
    ],
)
async def test_the_body_is_read_in_the_format_the_run_chose(
    postmark, body_format, body, shown
):
    await run({"subject": "s", "body": body, "body_format": body_format})

    assert shown in _sent(postmark)["HtmlBody"]


HOSTILE = (
    '<p onclick="steal()" style="background:url(https://t.example/a.gif)">Hi</p>'
    "<script>alert(1)</script>"
    '<img src="https://t.example/pixel.gif" width="1" height="1">'
    '<form action="https://evil.example"><input name="password"><button>Go</button></form>'
    '<a href="javascript:alert(2)">bad</a> <a href="https://ok.example">ok</a>'
)

GONE = ("<script", "alert(", "steal()", "t.example", "<form", "<input", "javascript:")
GONE += ("evil.example", ">Go<")


@pytest.mark.parametrize("body_format", ["markdown", "html"])
async def test_scripts_forms_and_tracking_pixels_never_reach_the_mail(
    postmark, body_format
):
    subject = '<img src="https://headline.invalid/s.gif"> Digest'
    await run({"subject": subject, "body": HOSTILE, "body_format": body_format})

    sent = _sent(postmark)
    body = sent["HtmlBody"].split("FROM OTTO", 1)[1]
    assert '<img src="https://headline.invalid' not in sent["HtmlBody"]
    for gone in GONE:
        assert gone not in body, gone
    assert 'href="https://ok.example"' in body
    assert "alert(" not in sent["TextBody"] and "Hi" in sent["TextBody"]


@pytest.mark.parametrize(
    "body_format, body",
    [
        ("plain", "Three things.\n\nFirst, the news."),
        ("markdown", "## Three things\n\n- First, the news."),
        ("html", "<h2>Three things</h2><ul><li>First, the news.</li></ul>"),
    ],
)
async def test_every_format_sends_a_plain_text_part(postmark, body_format, body):
    await run({"subject": "s", "body": body, "body_format": body_format})

    text = _sent(postmark)["TextBody"]
    assert "Three things" in text and "First, the news." in text
    assert "<" not in text


async def test_a_link_that_is_not_a_web_address_is_refused(postmark):
    response = await run({"subject": "s", "body": "b", "link": "javascript:alert(1)"})

    assert "must be a web address" in _error(response)
    assert sent_to(postmark) == []


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("Email sending isn't set up"), TimeoutError("it may still arrive")],
)
async def test_an_expected_failure_is_a_block_error_not_an_unknown_one(
    monkeypatch, failure
):
    """Uncaught, these would be filed as BlockUnknownError, the class kept for bugs."""
    block = SendEmailToMeBlock()
    monkeypatch.setattr(block, "send_to_owner", AsyncMock(side_effect=failure))
    context = ExecutionContext(
        user_id=OWNER, graph_id="g", graph_exec_id="ge", graph_version=1
    )

    with pytest.raises(BlockExecutionError, match=str(failure)):
        async for _ in block.execute(
            {"subject": "s", "body": "b"},
            execution_context=context,
            user_id=OWNER,
            graph_id="g",
            graph_exec_id="ge",
            graph_version=1,
            node_id="n",
            node_exec_id="ne",
        ):
            pass


def _sent(postmark) -> dict:
    assert len(postmark.emails.send.call_args_list) == 1
    return postmark.emails.send.call_args.kwargs


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

    def pipeline(self, transaction: bool) -> "_FakePipeline":
        assert transaction
        return _FakePipeline(self)


class _FakePipeline:
    """Only the counter's two commands; the live-Redis test covers their meaning."""

    def __init__(self, redis: _FakeRedis):
        self.redis = redis
        self.keys: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_) -> None:
        pass

    def incr(self, key: str) -> None:
        self.keys.append(key)

    def expire(self, key: str, seconds: int) -> None:
        pass

    async def execute(self) -> list:
        return [await self.redis.incr(key) for key in self.keys] + [True]
