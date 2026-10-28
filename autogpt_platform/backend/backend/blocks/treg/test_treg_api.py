"""Unit tests for the Treg HTTP client: what it sends and how it reads answers."""

import asyncio
import json
from typing import Any, ClassVar

import aiohttp
import pytest

from backend.blocks.treg import _api
from backend.blocks.treg._api import TregError, TregResponse, error_message
from backend.blocks.treg._config import TEST_CREDENTIALS


class FakeResponse:
    def __init__(self, status: int, content: bytes, headers: dict[str, str]):
        self.status = status
        self.content = content
        self.headers = headers

    def text(self) -> str:
        return self.content.decode()

    def json(self) -> Any:
        return json.loads(self.content)


class FakeRequests:
    response: ClassVar[FakeResponse]
    sent: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs):
        FakeRequests.sent = {"init": kwargs}

    async def request(self, method, url, **kwargs):
        FakeRequests.sent.update(method=method, url=url, kwargs=kwargs)
        return FakeRequests.response


def answer(
    status: int = 200, body: Any = None, headers: dict[str, str] | None = None
) -> FakeResponse:
    content = body if isinstance(body, bytes) else json.dumps(body or {}).encode()
    return FakeResponse(status, content, headers or {})


@pytest.fixture
def fake_requests(monkeypatch):
    monkeypatch.setattr(_api, "Requests", FakeRequests)
    FakeRequests.response = answer()
    return FakeRequests


@pytest.mark.asyncio
async def test_send_authenticates_and_never_retries_or_redirects(fake_requests):
    await _api.send(
        TEST_CREDENTIALS,
        "POST",
        "call/treg.people.search",
        params={"limit": 5, "tags": ["a", "b"], "fresh": True, "skip": None},
        body={"q": "cto"},
        headers={"X-Treg-Route-Max-Cost": "0.5"},
    )
    sent = fake_requests.sent
    assert sent["method"] == "POST"
    assert sent["url"] == "https://treg.to/call/treg.people.search"
    assert sent["init"]["extra_headers"] == {"X-Treg-Token": "mock-treg-api-key"}
    # A retried paid call can bill twice, and a redirect would carry the token.
    assert sent["init"]["retry_max_attempts"] == 1
    assert sent["kwargs"]["allow_redirects"] is False
    assert sent["kwargs"]["timeout"] == _api.REQUEST_TIMEOUT
    assert sent["kwargs"]["json"] == {"q": "cto"}
    assert sent["kwargs"]["params"] == [
        ("limit", "5"),
        ("tags", "a"),
        ("tags", "b"),
        ("fresh", "true"),
    ]
    assert sent["kwargs"]["headers"] == {"X-Treg-Route-Max-Cost": "0.5"}


@pytest.mark.asyncio
async def test_send_drops_headers_that_would_replace_the_credential(fake_requests):
    await _api.send(
        TEST_CREDENTIALS,
        "GET",
        "call/x.y",
        headers={"x-treg-token": "other", "Authorization": "Bearer x", "model": "m"},
    )
    assert fake_requests.sent["kwargs"]["headers"] == {"model": "m"}


@pytest.mark.asyncio
async def test_send_reads_treg_headers(fake_requests):
    fake_requests.response = answer(
        202,
        {"task_id": "t1"},
        {
            "X-Treg-Cost-Micro": "24500",
            "X-Treg-Call-Id": "call_1",
            "X-Treg-Served-By": "hunter.email.finder",
            "X-Treg-Async": '{"poll": {"endpoint": "minimax.query"}}',
            "X-Treg-Cache": "hit",
            "X-Treg-Upstream-Failed": "1",
        },
    )
    response = await _api.send(TEST_CREDENTIALS, "POST", "call/minimax.generate")
    assert response == TregResponse(
        status=202,
        body={"task_id": "t1"},
        cost_usd=0.0245,
        call_id="call_1",
        served_by="hunter.email.finder",
        async_task={"poll": {"endpoint": "minimax.query"}},
        cached=True,
        upstream_failed=True,
    )


@pytest.mark.asyncio
async def test_send_keeps_a_non_json_body_as_text(fake_requests):
    fake_requests.response = answer(200, b"<html>ok</html>")
    response = await _api.send(TEST_CREDENTIALS, "GET", "call/x.y")
    assert response.body == "<html>ok</html>"
    assert response.cost_usd == 0.0


@pytest.mark.asyncio
async def test_network_failures_are_handled_errors(monkeypatch):
    class FailingRequests(FakeRequests):
        async def request(self, method, url, **kwargs):
            raise aiohttp.ClientConnectionError("refused")

    monkeypatch.setattr(_api, "Requests", FailingRequests)
    with pytest.raises(TregError, match="Could not reach Treg"):
        await _api.send(TEST_CREDENTIALS, "GET", "catalog/search")


@pytest.mark.asyncio
async def test_a_hung_call_says_it_may_have_charged(monkeypatch):
    class SlowRequests(FakeRequests):
        async def request(self, method, url, **kwargs):
            await asyncio.Event().wait()  # never set: the call hangs
            return FakeRequests.response

    monkeypatch.setattr(_api, "Requests", SlowRequests)
    monkeypatch.setattr(_api, "REQUEST_DEADLINE_SECONDS", 0.01)
    with pytest.raises(TregError, match="may still have answered and charged"):
        await _api.send(TEST_CREDENTIALS, "GET", "call/x.y")


def failed(status: int, body: Any, call_id: str | None = "call_9") -> TregResponse:
    return TregResponse(
        status=status,
        body=body,
        cost_usd=0.0,
        call_id=call_id,
        served_by=None,
        async_task=None,
        cached=False,
        upstream_failed=False,
    )


@pytest.mark.parametrize(
    "response, message",
    [
        (
            failed(
                402,
                {
                    "detail": {
                        "error": "insufficient_balance",
                        "message": "Your team balance is $0.00.",
                        "topup_url": "https://treg.to/billing",
                    }
                },
            ),
            (
                "HTTP 402 (call id call_9): insufficient_balance: Your team balance "
                "is $0.00. Top up at https://treg.to/billing."
            ),
        ),
        (
            failed(502, {"error": "route_failed", "message": "every candidate failed"}),
            "HTTP 502 (call id call_9): route_failed: every candidate failed",
        ),
        (failed(404, {"detail": "unknown tool"}, None), "HTTP 404: unknown tool"),
        (
            failed(401, {"detail": "invalid token"}),
            (
                "HTTP 401 (call id call_9): invalid token. Check your Treg API key, "
                "or make a new one at https://treg.to."
            ),
        ),
        (failed(400, "bad domain"), "HTTP 400 (call id call_9): bad domain"),
        (
            failed(400, {"errors": ["x"]}),
            'HTTP 400 (call id call_9): {"errors": ["x"]}',
        ),
    ],
)
def test_error_message_names_the_cause(response, message):
    assert error_message(response) == f"Treg call failed with {message}"


@pytest.mark.parametrize(
    "tool",
    [
        "treg.people.email.find",
        "tikhub.tiktok.user.profile",
        "intercom/conversations",
        "intercom/v2/contacts/123",
        "https://api.stripe.com/v1/charges",
    ],
)
def test_validate_tool_accepts_every_documented_form(tool):
    assert _api.validate_tool(f" {tool} ") == tool


@pytest.mark.parametrize(
    "tool",
    [
        "../orgs/1/api-keys",
        "intercom/../../orgs",
        "intercom/%2e%2e/orgs",
        "intercom%2F..",
        "x.y?admin=1",
        "x.y#frag",
        "http://api.stripe.com/v1",
        "/catalog/search",
        "has space",
        "back\\slash",
    ],
)
def test_validate_tool_rejects_anything_that_leaves_the_call_route(tool):
    with pytest.raises(TregError, match="is not a Treg tool"):
        _api.validate_tool(tool)


@pytest.mark.parametrize(
    "value, usd", [("1500000", 1.5), ("0", 0.0), ("-5", 0.0), (None, 0.0), ("x", 0.0)]
)
def test_micro_to_usd(value, usd):
    assert _api.micro_to_usd(value) == usd
