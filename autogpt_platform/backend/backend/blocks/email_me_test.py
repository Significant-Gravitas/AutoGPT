"""SendEmailToMeBlock run the way Otto runs it, with Postmark as the only stub.

Everything from the copilot's block runner down to the notification service's
sender is real, except the RPC hop to that service, which is called in-process.
"""

import asyncio
import base64
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import requests
from postmarker.exceptions import ClientError

from backend.blocks import email_me
from backend.blocks.email_me import SendEmailToMeBlock
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import BlockOutputResponse, SetupRequirementsResponse
from backend.copilot.tools.run_block import RunBlockTool
from backend.data.execution import ExecutionContext
from backend.data.workspace import WorkspaceFile
from backend.notifications import email as email_module
from backend.notifications import notifications, owner_email, owner_email_images
from backend.notifications.email import EmailSender
from backend.util import file as file_module
from backend.util import workspace as workspace_module
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
    '<img src="https://203.0.113.9/t.example/pixel.gif" width="1" height="1">'
    '<form action="https://evil.example"><input name="password"><button>Go</button></form>'
    '<a href="javascript:alert(2)">bad</a> <a href="https://ok.example">ok</a>'
    '<a href="data:text/html;base64,PHNjcmlwdD4=">click</a>'
)

GONE = ("<script", "alert(", "steal()", "t.example", "<form", "<input", "javascript:")
GONE += ("evil.example", ">Go<", "data:text/html")


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


async def test_a_link_keeps_its_attributes_and_gets_the_template_style(postmark):
    body = '<p><a href="https://ok.example" title="A -> B">the report</a></p>'
    await run({"subject": "s", "body": body, "body_format": "html"})

    assert (
        '<a href="https://ok.example" title="A -&gt; B" '
        'style="color:#4D36AD; text-decoration:underline;">the report</a>'
    ) in _sent(postmark)["HtmlBody"]


@pytest.mark.parametrize(
    "subject, body, refusal",
    [(" \n ", "b", "subject"), ("s", " \n\n ", "The body is empty")],
)
async def test_a_blank_subject_or_body_is_refused(postmark, subject, body, refusal):
    response = await run({"subject": subject, "body": body})

    assert refusal in _error(response)
    assert sent_to(postmark) == []


@pytest.mark.parametrize(
    "opening, closing",
    # Each of the last three hides the nesting from a naive count: a closing tag
    # in an attribute value, a self-closing div, an end tag for nothing open.
    [
        ("<div>", "</div>"),
        ("<div title='</div>'>", "</div>"),
        ("<div/>", ""),
        ("<div></b>", ""),
    ],
)
async def test_html_nested_too_deeply_is_refused(postmark, opening, closing):
    body = opening * 150 + "deep" + closing * 150

    response = await run({"subject": "s", "body": body, "body_format": "html"})

    assert "nested more than 100 levels deep" in _error(response)
    assert sent_to(postmark) == []


async def test_an_unclosed_script_takes_the_rest_of_the_body_with_it(postmark):
    body = "<p>Before</p><script>steal(document.cookie)"

    await run({"subject": "s", "body": body, "body_format": "html"})

    html = _sent(postmark)["HtmlBody"]
    assert "Before" in html
    assert "steal(" not in html


async def test_a_dropped_element_after_wide_unicode_is_cut_where_it_ends(postmark):
    # "İ" lowercases to two characters, which shifts any offset taken from .lower().
    body = "İstanbul " * 5 + "<script>steal()</script><p>after the script</p>"

    await run({"subject": "s", "body": body, "body_format": "html"})

    html = _sent(postmark)["HtmlBody"]
    assert "after the script" in html
    assert "steal(" not in html


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


PNG = b"\x89PNG\r\n\x1a\n" + bytes(64)


@pytest.fixture(autouse=True)
def network(monkeypatch) -> list[str]:
    """Every URL that reached the HTTP transport; nothing does in these tests."""
    reached: list[str] = []

    async def no_network(self, method, url, *args, **kwargs):
        reached.append(str(url))
        raise aiohttp.ClientConnectionError("no network in tests")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", no_network)
    return reached


@pytest.fixture
def workspace(monkeypatch, tmp_path) -> dict[str, tuple[str, bytes]]:
    """Files by id, as (owning workspace, content). The run's workspace is "ws";
    the lookups answer the way the database queries do."""
    files: dict[str, tuple[str, bytes]] = {}

    def record(file_id: str) -> WorkspaceFile:
        now = datetime.now(tz=UTC)
        return WorkspaceFile(
            id=file_id,
            workspace_id=files[file_id][0],
            created_at=now,
            updated_at=now,
            name=f"{file_id}.png",
            path=f"/{file_id}.png",
            storage_path=file_id,
            mime_type="image/png",
            size_bytes=len(files[file_id][1]),
        )

    async def scoped(file_id: str, workspace_id: str):
        owner = files.get(file_id, (None, b""))[0]
        return record(file_id) if owner == workspace_id else None

    async def unscoped(file_id: str):
        return record(file_id) if file_id in files else None

    async def retrieve(storage_path: str) -> bytes:
        return files[storage_path][1]

    db = SimpleNamespace(get_workspace_file=scoped, get_workspace_file_by_id=unscoped)
    storage = SimpleNamespace(retrieve=retrieve)
    monkeypatch.setattr(workspace_module, "workspace_db", lambda: db)
    monkeypatch.setattr(
        workspace_module, "get_workspace_storage", AsyncMock(return_value=storage)
    )
    monkeypatch.setattr(file_module, "scan_content_safe", AsyncMock())
    monkeypatch.setattr(file_module, "TEMP_DIR", tmp_path)
    return files


async def test_a_workspace_image_is_attached_and_shown_inline(postmark, workspace):
    workspace["chart"] = ("ws", PNG)

    await run({"subject": "s", "body": "Results:\n\n![Chart](workspace://chart)"})

    sent = _sent(postmark)
    assert sent["Attachments"] == [
        {
            "Name": "image-1.png",
            "Content": base64.b64encode(PNG).decode(),
            "ContentType": "image/png",
            "ContentID": "cid:image-1",
        }
    ]
    assert '<img src="cid:image-1" alt="Chart"' in sent["HtmlBody"]
    assert "workspace://" not in sent["HtmlBody"]


async def test_the_plain_text_part_shows_an_images_alt_text(postmark, workspace):
    workspace["chart"] = ("ws", PNG)

    await run(
        {"subject": "s", "body": "Results:\n\n![Weekly chart](workspace://chart)"}
    )

    text = _sent(postmark)["TextBody"]
    assert "Weekly chart" in text
    assert "cid:" not in text and "workspace://" not in text


async def test_an_image_in_another_users_workspace_is_never_read(postmark, workspace):
    workspace["theirs"] = ("ws-someone-else", PNG)

    await run({"subject": "s", "body": "![Their chart](workspace://theirs)"})

    sent = _sent(postmark)
    assert not sent["Attachments"]
    assert "Their chart" in sent["HtmlBody"]
    assert 'src="cid:' not in sent["HtmlBody"]


async def test_an_image_on_an_internal_address_is_never_fetched(postmark, network):
    await run({"subject": "s", "body": "![Internal](https://127.0.0.1/secret.png)"})

    sent = _sent(postmark)
    assert not [url for url in network if "127.0.0.1" in url]
    assert not sent["Attachments"]
    assert "Internal" in sent["HtmlBody"]


@pytest.mark.parametrize(
    "content",
    [b"<svg onload='steal()'></svg>", b"%PDF-1.7 not an image"],
)
async def test_a_file_that_is_not_an_image_is_left_out(postmark, workspace, content):
    workspace["fake"] = ("ws", content)

    await run({"subject": "s", "body": "![Fake](workspace://fake)"})

    sent = _sent(postmark)
    assert not sent["Attachments"]
    assert "Fake" in sent["HtmlBody"]


async def test_an_oversized_image_is_left_out_and_its_alt_text_shown(
    monkeypatch, postmark, workspace
):
    monkeypatch.setattr(owner_email_images, "MAX_IMAGE_BYTES", len(PNG) - 1)
    workspace["big"] = ("ws", PNG)

    await run({"subject": "s", "body": "![Big chart](workspace://big)"})

    sent = _sent(postmark)
    assert not sent["Attachments"]
    assert "Big chart" in sent["HtmlBody"]


@pytest.mark.parametrize(
    "limit",
    [("MAX_IMAGES", 1), ("MAX_TOTAL_BYTES", len(PNG) + 1)],
)
async def test_images_past_the_count_or_total_size_are_left_out(
    monkeypatch, postmark, workspace, limit
):
    monkeypatch.setattr(owner_email_images, *limit)
    workspace["one"] = ("ws", PNG)
    workspace["two"] = ("ws", PNG)

    await run(
        {"subject": "s", "body": "![One](workspace://one) ![Two](workspace://two)"}
    )

    sent = _sent(postmark)
    assert [a["ContentID"] for a in sent["Attachments"]] == ["cid:image-1"]
    assert "Two" in sent["HtmlBody"]


async def test_an_owner_email_is_tagged_and_never_tracked_with_tracking_on(
    monkeypatch, postmark, workspace
):
    """Postmark tracking is on for notification mail, but this mail's links are
    the user's own and can carry their tokens."""
    monkeypatch.setattr(
        email_module.settings.config, "postmark_track_opens_and_clicks", True
    )
    workspace["chart"] = ("ws", PNG)
    body = "[The report](https://docs.example/d/abc?token=s3cret)\n\n![Chart](workspace://chart)"

    await run({"subject": "s", "body": body})

    sent = _sent(postmark)
    assert sent["Tag"] == "owner_email"
    assert sent["TrackOpens"] is False and sent["TrackLinks"] == "None"
    assert [a["ContentID"] for a in sent["Attachments"]] == ["cid:image-1"]
    assert 'href="https://docs.example/d/abc?token=s3cret"' in sent["HtmlBody"]


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
