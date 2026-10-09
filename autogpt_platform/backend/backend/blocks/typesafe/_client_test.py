import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterator

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score
from typesafe_sdk._core import transport as sdk_transport

from backend.blocks.typesafe import _client

RESPONSE = """ { "model": "jev-latest", "answers": {
  "choice": {"type":"choice","choice":"a","probabilities":{"a":0.9,"b":0.1},"confidence":0.8},
  "score": {"type":"score","score":0.75,"legend":{"0":"Low","1":"High"},"probabilities":{"0":0.25,"1":0.75},"confidence":0.6},
  "noul": {"type":"noul","noul":0.82}
}, "usage": {"input_tokens":123,"output_tokens":45} }\n"""


@pytest.fixture(params=[False, True], ids=["logging-enabled", "logging-disabled"])
def logging_mode(request: pytest.FixtureRequest) -> Iterator[None]:
    previous = logging.root.manager.disable
    if request.param:
        logging.disable(logging.INFO)
    try:
        yield
        assert logging.root.manager.disable == (
            logging.INFO if request.param else previous
        )
    finally:
        logging.disable(previous)


def mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx2.Request], Awaitable[httpx2.Response]],
) -> list[AsyncTypeSafeClient]:
    clients = []

    def factory(
        *, api_key: str, base_url: str, retry: RetryPolicy
    ) -> AsyncTypeSafeClient:
        client = AsyncTypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            transport=httpx2.MockTransport(handler),
            retry=retry,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(_client, "AsyncTypeSafeClient", factory)
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    return clients


async def test_actual_async_sdk_captures_verbatim_bodies_and_typed_answers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    logging_mode: None,
):
    sent: list[httpx2.Request] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        await asyncio.sleep(0.01)
        return httpx2.Response(
            200,
            content=RESPONSE.encode(),
            headers={"x-typesafe-request-id": "req-123"},
        )

    clients = mock_transport(monkeypatch, handler)
    questions = {
        "choice": Choice(instructions="Pick", criteria={"a": "A", "b": "B"}),
        "score": Score(instructions="Rate", criteria=["Low", "High"]),
        "noul": Noul(instructions="Yes?"),
    }
    with caplog.at_level(logging.DEBUG, logger="typesafe_sdk"):
        result = await _client.call_jev(
            "test-secret", {"secret": "private-state"}, questions
        )
    assert result.request.encode() == sent[0].content
    assert result.response == RESPONSE
    assert result.answers == json.loads(RESPONSE)["answers"]
    assert result.request_id == "req-123"
    assert result.input_tokens == 123
    assert result.output_tokens == 45
    assert result.latency_ms >= 10
    assert not result.truncated
    assert result.truncation_note == ""
    assert result.error == ""
    assert sent[0].method == "POST"
    assert str(sent[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert sent[0].headers["authorization"] == "Bearer test-secret"
    assert json.loads(result.request)["model"] == "jev-latest"
    assert json.loads(result.request)["state"] == '{"secret":"private-state"}'
    assert clients[0]._http_client.is_closed
    assert "private-state" not in caplog.text
    assert "test-secret" not in caplog.text
    assert '"noul":0.82' not in caplog.text


async def test_concurrent_calls_do_not_mix_wire_bodies(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    logging_mode: None,
):
    arrived = 0
    both_arrived = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal arrived
        state = json.loads(request.content)["state"]
        arrived += 1
        if arrived == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=2)
        return httpx2.Response(
            200,
            content=RESPONSE.replace("0.82", "0.81" if state == "first" else "0.83"),
            headers={"x-typesafe-request-id": state},
        )

    mock_transport(monkeypatch, handler)
    original_logger = sdk_transport.logger
    questions = {"noul": Noul(instructions="Yes?")}
    first, second = await asyncio.gather(
        _client.call_jev("test", "first", questions),
        _client.call_jev("test", "second", questions),
    )
    assert json.loads(first.request)["state"] == first.request_id == "first"
    assert json.loads(second.request)["state"] == second.request_id == "second"
    assert first.answers["noul"]["noul"] == 0.81
    assert second.answers["noul"]["noul"] == 0.83
    assert "body=" not in caplog.text
    assert sdk_transport.logger is original_logger


async def test_http_failure_restores_logger_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401,
            content=' {"error":"private-response"}\n',
            headers={"x-typesafe-request-id": "error-id"},
        )

    clients = mock_transport(monkeypatch, handler)
    logger = logging.getLogger("typesafe_sdk")
    before = (logger.level, logger.disabled, list(logger.filters))
    original_logger = sdk_transport.logger
    result = await _client.call_jev(
        "test", "private-state", {"q": Noul(instructions="Yes?")}
    )
    assert result.error == "Jev API request failed (HTTP 401)."
    assert result.response == ' {"error":"private-response"}\n'
    assert json.loads(result.request)["state"] == "private-state"
    assert result.input_tokens is None and result.output_tokens is None
    assert result.answers == {}
    assert result.request_id == "error-id"
    assert sdk_transport.logger is original_logger
    assert (logger.level, logger.disabled, logger.filters) == before
    assert clients[0]._http_client.is_closed
    assert "private-state" not in caplog.text
    assert "private-response" not in caplog.text


async def test_cancellation_restores_logger_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, logging_mode: None
):
    waiting = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        waiting.set()
        await asyncio.Future()
        raise AssertionError("Unreachable")

    clients = mock_transport(monkeypatch, handler)
    logger = logging.getLogger("typesafe_sdk")
    before = (logger.level, logger.disabled, list(logger.filters))
    original_logger = sdk_transport.logger
    task = asyncio.create_task(
        _client.call_jev("test", "state", {"q": Noul(instructions="Yes?")})
    )
    await asyncio.wait_for(waiting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (logger.level, logger.disabled, logger.filters) == before
    assert clients[0]._http_client.is_closed
    assert sdk_transport.logger is original_logger


async def test_truncation_is_visible_in_result(monkeypatch: pytest.MonkeyPatch):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, content=RESPONSE, headers={"x-typesafe-request-id": "truncated"}
        )

    mock_transport(monkeypatch, handler)
    result = await _client.call_jev(
        "test", "x" * 40_000, {"q": Noul(instructions="Yes?")}
    )
    assert result.truncated
    assert result.truncation_note
    assert len(json.loads(result.request)["state"]) < 40_000


async def test_missing_token_usage_is_not_invented(monkeypatch: pytest.MonkeyPatch):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            content=RESPONSE.replace('"input_tokens":123,"output_tokens":45', ""),
            headers={"x-typesafe-request-id": "missing-usage"},
        )

    mock_transport(monkeypatch, handler)
    result = await _client.call_jev("test", "state", {"q": Noul(instructions="Yes?")})
    assert result.error == "Jev response did not report token usage."
    assert result.input_tokens is None and result.output_tokens is None
    assert result.response == RESPONSE.replace(
        '"input_tokens":123,"output_tokens":45', ""
    )


async def test_overload_is_not_retried_and_environment_cannot_redirect_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    sent: list[httpx2.Request] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(529, json={"error": "overloaded"})

    mock_transport(monkeypatch, handler)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://untrusted.invalid")
    result = await _client.call_jev("test", "state", {"q": Noul(instructions="Yes?")})
    assert result.error == "Jev API request failed (HTTP 529)."
    assert len(sent) == 1
    assert str(sent[0].url) == "https://api.typesafe.ai/v1/systemone"


@pytest.mark.parametrize("response", ["{invalid-json", '{"model":"jev-latest"}'])
async def test_invalid_response_remains_visible(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, content=response, headers={"x-typesafe-request-id": "bad"}
        )

    mock_transport(monkeypatch, handler)
    result = await _client.call_jev("test", "state", {"q": Noul(instructions="Yes?")})
    assert result.response == response
    assert result.error == "Jev returned an invalid response (HTTP 200)."
    assert result.input_tokens is None and result.output_tokens is None
    assert result.request_id == "bad"


async def test_connection_failure_has_no_fabricated_response(
    monkeypatch: pytest.MonkeyPatch,
):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("test connection failure", request=request)

    clients = mock_transport(monkeypatch, handler)
    result = await _client.call_jev("test", "state", {"q": Noul(instructions="Yes?")})
    assert result.response is None
    assert result.request and result.request_id == ""
    assert result.input_tokens is None and result.output_tokens is None
    assert (
        result.error
        == "Jev connection failed or timed out; no HTTP response was received."
    )
    assert clients[0]._http_client.is_closed


async def test_missing_request_id_preserves_response(monkeypatch: pytest.MonkeyPatch):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=RESPONSE)

    mock_transport(monkeypatch, handler)
    result = await _client.call_jev("test", "state", {"q": Noul(instructions="Yes?")})
    assert result.response == RESPONSE
    assert result.request_id == ""
    assert result.error == "Jev response did not include a request ID."
