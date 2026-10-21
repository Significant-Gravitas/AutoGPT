"""Postmark's delivery, open, click and bounce webhooks become PostHog events
for the user the email went to, without the address and without link secrets."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest

from backend.util import posthog_client

from . import postmark
from .analytics import parse_postmark_time, safe_link

TOKEN = "test-webhook-token"
USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
RECIPIENT = "person@example.com"
MESSAGE_ID = "883953f4-6105-42a2-a16a-77a8eac79483"

app = fastapi.FastAPI()
app.include_router(postmark.router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def webhook_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(postmark.postmark_api_key_auth, "expected_token", TOKEN)


@pytest.fixture
def posthog(monkeypatch: pytest.MonkeyPatch) -> Mock:
    posthog = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: posthog)
    monkeypatch.setattr(posthog_client, "_environment", lambda: "local")
    return posthog


@pytest.fixture
def lookup(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    lookup = AsyncMock(return_value=SimpleNamespace(id=USER_ID))
    monkeypatch.setattr(postmark, "get_user_by_email", lookup)
    return lookup


@pytest.fixture
def unverify(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    unverify = AsyncMock()
    monkeypatch.setattr(postmark, "set_user_email_verification", unverify)
    return unverify


def test_a_delivery_is_sent_for_the_recipients_user(posthog: Mock, lookup: AsyncMock):
    _post(_delivery())

    lookup.assert_awaited_once_with(RECIPIENT)
    call = posthog.capture.call_args.kwargs
    assert call["distinct_id"] == USER_ID
    assert call["event"] == "email_delivered"
    assert call["timestamp"] == datetime(2019, 11, 5, 16, 33, 54, 907025, tzinfo=UTC)
    properties = call["properties"]
    assert properties["email_type"] == "briefing"
    assert properties["message_stream"] == "outbound"
    assert properties["message_id"] == MESSAGE_ID
    assert RECIPIENT not in str(call)


def test_an_open_carries_whether_it_was_the_first(posthog: Mock, lookup: AsyncMock):
    _post(_open())

    call = posthog.capture.call_args.kwargs
    assert call["event"] == "email_opened"
    assert call["properties"]["first_open"] is True
    assert "Geo" not in str(call) and "188.2.95.4" not in str(call)
    assert RECIPIENT not in str(call)


def test_a_click_keeps_only_the_links_scheme_host_and_path(
    posthog: Mock, lookup: AsyncMock
):
    _post(
        _click(
            "https://platform.agpt.co/library/agents/"
            "8f14e45f-ceea-467a-9575-0d7e1b2c3a4f?executionId=1#top"
        )
    )

    call = posthog.capture.call_args.kwargs
    assert call["event"] == "email_clicked"
    assert call["properties"]["link"] == "https://platform.agpt.co/library/agents/:id"


def test_a_hard_bounce_is_sent_and_unverifies_the_address(
    posthog: Mock, lookup: AsyncMock, unverify: AsyncMock
):
    _post(_bounce(type_name="HardBounce", type_code=1))

    call = posthog.capture.call_args.kwargs
    assert call["event"] == "email_bounced"
    assert call["properties"]["bounce_type"] == "HardBounce"
    assert RECIPIENT not in str(call)
    unverify.assert_awaited_once_with(USER_ID, False)


def test_a_soft_bounce_is_sent_but_keeps_the_address_verified(
    posthog: Mock, lookup: AsyncMock, unverify: AsyncMock
):
    _post(_bounce(type_name="SoftBounce", type_code=4096))

    assert posthog.capture.call_args.kwargs["properties"]["bounce_type"] == (
        "SoftBounce"
    )
    unverify.assert_not_awaited()


@pytest.mark.parametrize("record_type", ["Delivery", "Open", "Click", "Bounce"])
def test_an_address_with_no_account_sends_nothing(
    record_type: str, posthog: Mock, lookup: AsyncMock, unverify: AsyncMock
):
    lookup.return_value = None
    payload = {
        "Delivery": _delivery,
        "Open": _open,
        "Click": lambda: _click("https://platform.agpt.co/copilot"),
        "Bounce": lambda: _bounce(type_name="HardBounce", type_code=1),
    }[record_type]()

    _post(payload)

    posthog.capture.assert_not_called()
    unverify.assert_not_awaited()


def test_a_redelivered_webhook_is_the_same_event(posthog: Mock, lookup: AsyncMock):
    _post(_open())
    _post(_open())
    _post(_open(received_at="2019-11-05T16:40:00.0000000Z"))

    first, retry, later = (c.kwargs for c in posthog.capture.call_args_list)
    assert first["uuid"] == retry["uuid"]
    assert first["timestamp"] == retry["timestamp"]
    assert later["uuid"] != first["uuid"]


def test_two_links_clicked_at_once_are_two_events(posthog: Mock, lookup: AsyncMock):
    _post(_click("https://platform.agpt.co/copilot"))
    _post(_click("https://platform.agpt.co/library"))

    first, second = (c.kwargs for c in posthog.capture.call_args_list)
    assert first["uuid"] != second["uuid"]
    assert "platform.agpt.co" not in first["properties"]["$insert_id"]


def test_an_untagged_email_sends_no_email_type(posthog: Mock, lookup: AsyncMock):
    _post({**_delivery(), "Tag": ""})

    assert "email_type" not in posthog.capture.call_args.kwargs["properties"]


@pytest.mark.parametrize("headers", [{}, {"X-Postmark-Webhook-Token": "wrong"}])
def test_a_request_without_the_webhook_token_is_refused(
    headers: dict[str, str], posthog: Mock, lookup: AsyncMock
):
    response = client.post("/", json=_open(), headers=headers)

    assert response.status_code == 401
    lookup.assert_not_awaited()
    posthog.capture.assert_not_called()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://platform.agpt.co/copilot", "https://platform.agpt.co/copilot"),
        (
            "https://platform.agpt.co/settings/account?f=weekly",
            "https://platform.agpt.co/settings/account",
        ),
        (
            "https://platform.agpt.co/api/email/unsubscribe?token=abc",
            "https://platform.agpt.co/api/email/unsubscribe",
        ),
        (
            "https://platform.agpt.co/reset-password/Kq8sPzLmWvTnRbYxHdJcFgAe",
            "https://platform.agpt.co/reset-password/:id",
        ),
        (
            "https://platform.agpt.co/reset-password/kqsplmwvtnrbyxhdjcfgaeqwertyuiop",
            "https://platform.agpt.co/reset-password/:id",
        ),
        (
            "https://platform.agpt.co/marketplace/agent/autogpt/blog-writer",
            "https://platform.agpt.co/marketplace/agent/autogpt/blog-writer",
        ),
        (
            "https://example.com/members/alice@example.com/profile",
            "https://example.com/members/:id/profile",
        ),
        (
            "https://example.com/members/alice%40example.com",
            "https://example.com/members/:id",
        ),
        ("https://example.com/Tok", "https://example.com/:id"),
        (
            "HTTPS://user:secret@Platform.AGPT.co:8443/x",
            "https://platform.agpt.co:8443/x",
        ),
        ("mailto:support@agpt.co", "mailto:"),
        ("not a link", None),
        ("https://[::1", None),
    ],
)
def test_safe_link(url: str, expected: str | None):
    assert safe_link(url) == expected


def test_an_unreadable_postmark_time_falls_back_to_now():
    assert parse_postmark_time("yesterday") is None


def _post(payload: dict) -> None:
    response = client.post(
        "/", json=payload, headers={"X-Postmark-Webhook-Token": TOKEN}
    )
    assert response.status_code == 200, response.text


def _delivery() -> dict:
    return {
        "RecordType": "Delivery",
        "ServerID": 23,
        "MessageStream": "outbound",
        "MessageID": MESSAGE_ID,
        "Recipient": RECIPIENT,
        "Tag": "briefing",
        "DeliveredAt": "2019-11-05T16:33:54.9070259Z",
        "Details": "Test delivery webhook details",
        "Metadata": {},
    }


def _engagement(received_at: str) -> dict:
    return {
        "MessageStream": "outbound",
        "Metadata": {},
        "Recipient": RECIPIENT,
        "MessageID": MESSAGE_ID,
        "ReceivedAt": received_at,
        "Platform": "WebMail",
        "Tag": "alert",
        "UserAgent": "Mozilla/5.0",
        "OS": {"Name": "OS X 10.7 Lion", "Family": "OS X 10", "Company": "Apple"},
        "Client": {"Name": "Chrome 35.0", "Family": "Chrome", "Company": "Google"},
        "Geo": {"Country": "Serbia", "City": "Novi Sad", "IP": "188.2.95.4"},
    }


def _open(received_at: str = "2019-11-05T16:33:54.9070259Z") -> dict:
    return {
        **_engagement(received_at),
        "RecordType": "Open",
        "FirstOpen": True,
        "ReadSeconds": 5,
    }


def _click(link: str) -> dict:
    return {
        **_engagement("2019-11-05T16:35:00.0000000Z"),
        "RecordType": "Click",
        "ClickLocation": "HTML",
        "OriginalLink": link,
    }


def _bounce(*, type_name: str, type_code: int) -> dict:
    return {
        "RecordType": "Bounce",
        "ID": 42,
        "Type": type_name,
        "TypeCode": type_code,
        "Tag": "trial_update",
        "MessageID": MESSAGE_ID,
        "Details": "Test bounce details",
        "Email": RECIPIENT,
        "From": "notify@agpt.co",
        "BouncedAt": "2019-11-05T16:33:54.9070259Z",
        "Inactive": True,
        "DumpAvailable": False,
        "CanActivate": True,
        "Subject": "Your trial",
        "ServerID": 23,
        "MessageStream": "outbound",
        "Content": "",
        "Name": "Hard bounce",
        "Description": "The server was unable to deliver your message.",
        "Metadata": {},
    }
