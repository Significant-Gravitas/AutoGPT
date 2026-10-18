import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import pytest

from backend.copilot.provider_failure import ProviderFailure, ProviderFailureKind
from backend.copilot.sdk.model_gateway import relay
from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.stream import (
    AnthropicStreamBuilder,
    Event,
    StreamItem,
)
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

_ROUTE = GatewayRoute(
    adapter="anthropic_passthrough", model="local-model", base_url="http://127.0.0.1:1"
)
_SECRET_PROMPT = "the user's private prompt text"


def _answer(text: str = "hi", usage: GatewayUsage | None = None) -> list[StreamItem]:
    builder = AnthropicStreamBuilder(model="local-model", input_estimate=10)
    events = builder.text(text) + builder.finish("end_turn", usage)
    return [
        StreamItem(event=e, usage=usage if e["type"] == "message_delta" else None)
        for e in events
    ]


class _FakeUpstream:
    """An upstream that plays back a script, failing where it is told to."""

    def __init__(
        self,
        items: list[StreamItem] | None = None,
        *,
        fail: BaseException | None = None,
        fail_after: int = 0,
        pause: float = 0.0,
    ) -> None:
        self.items = items if items is not None else _answer()
        self.fail = fail
        self.fail_after = fail_after
        self.pause = pause
        self.bodies: list[Event] = []
        self.finished_streams = 0
        self.closed = False

    async def stream(self, body: Event) -> AsyncGenerator[StreamItem, None]:
        self.bodies.append(body)
        try:
            for index, item in enumerate(self.items):
                if self.fail is not None and index == self.fail_after:
                    raise self.fail
                await asyncio.sleep(self.pause)
                yield item
        finally:
            self.finished_streams += 1

    async def complete(self, body: Event) -> Event:
        self.bodies.append(body)
        if self.fail is not None:
            raise self.fail
        return {
            "type": "message",
            "content": [],
            "usage": {"input_tokens": 7, "output_tokens": 2},
        }

    async def aclose(self) -> None:
        self.closed = True


async def _post(
    gateway: ModelGateway, body: dict[str, Any], **headers: str
) -> tuple[int, dict[str, str], bytes]:
    auth = {"Authorization": f"Bearer {gateway.auth_token}"} | headers
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{gateway.base_url}/v1/messages?beta=true", json=body, headers=auth
        ) as response:
            headers = {k.lower(): v for k, v in response.headers.items()}
            return response.status, headers, await response.read()


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/models?limit=1000"),
        ("HEAD", "/api/hello"),
        ("GET", "/healthz"),
        ("GET", "/anything/else"),
    ],
)
async def test_every_route_needs_the_token(method, path):
    async with ModelGateway(_ROUTE, upstream=_FakeUpstream()) as gateway:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                f"{gateway.base_url}{path}",
                headers={"Authorization": "Bearer wrong"},
            ) as response:
                assert response.status == 401
                if method != "HEAD":
                    assert await response.json() == {
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": "Unauthorized",
                        },
                    }


async def test_the_token_is_accepted_as_a_bearer_or_an_api_key():
    async with ModelGateway(_ROUTE, upstream=_FakeUpstream()) as gateway:
        async with aiohttp.ClientSession() as session:
            for headers in (
                {"Authorization": f"Bearer {gateway.auth_token}"},
                {"x-api-key": gateway.auth_token},
            ):
                async with session.get(
                    f"{gateway.base_url}/healthz", headers=headers
                ) as response:
                    assert response.status == 200
        assert gateway.base_url.startswith("http://127.0.0.1:")


async def test_models_count_tokens_and_hello():
    async with ModelGateway(_ROUTE, upstream=_FakeUpstream()) as gateway:
        auth = {"x-api-key": gateway.auth_token}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{gateway.base_url}/v1/models?limit=1000", headers=auth
            ) as response:
                listing = await response.json()
            async with session.post(
                f"{gateway.base_url}/v1/messages/count_tokens",
                headers=auth,
                json={"messages": [{"role": "user", "content": "x" * 400}]},
            ) as response:
                counted = await response.json()
            async with session.head(
                f"{gateway.base_url}/api/hello", headers=auth
            ) as response:
                hello = response.status
    assert [m["id"] for m in listing["data"]] == ["local-model"]
    assert listing["has_more"] is False
    assert counted == {"input_tokens": 101}
    assert hello == 200


async def test_a_stream_is_relayed_and_its_usage_kept():
    usage = GatewayUsage(
        input_tokens=30, cache_read_tokens=70, output_tokens=5, reasoning_tokens=2
    )
    upstream = _FakeUpstream(_answer("hello", usage))
    async with ModelGateway(_ROUTE, upstream=upstream) as gateway:
        status, headers, body = await _post(gateway, {"messages": [], "stream": True})
        await _post(gateway, {"messages": [], "stream": True})

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert body.count(b"event: ") == 6
    assert b'"text":"hello"' in body
    assert gateway.requests == 2
    assert gateway.usage == usage.plus(usage)
    assert gateway.peak_prompt_tokens == 100
    assert upstream.closed


async def test_raw_frames_are_written_exactly_as_received():
    raw = b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n\n'
    item = StreamItem(
        event={"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        raw=raw,
    )
    async with ModelGateway(_ROUTE, upstream=_FakeUpstream([item])) as gateway:
        _, _, body = await _post(gateway, {"messages": [], "stream": True})
    assert body == raw
    assert gateway.usage == GatewayUsage(input_tokens=3)


async def test_a_failure_before_anything_streams_is_a_real_http_error():
    failure = ProviderFailure(
        kind=ProviderFailureKind.USAGE_LIMIT,
        message="Limit reached.",
        auth_provider="codex",
    )
    error = GatewayError(
        429,
        "rate_limit_error",
        "Limit reached.",
        headers={"retry-after": "60"},
        failure=failure,
    )
    upstream = _FakeUpstream(fail=error)
    async with ModelGateway(_ROUTE, upstream=upstream) as gateway:
        status, headers, body = await _post(gateway, {"messages": [], "stream": True})
    assert status == 429
    assert headers["retry-after"] == "60"
    assert (
        body
        == b'{"type": "error", "error": {"type": "rate_limit_error", "message": "Limit reached."}}'
    )
    assert gateway.last_failure == failure
    assert upstream.finished_streams == 1


async def test_a_failure_mid_stream_ends_it_with_an_error_event():
    upstream = _FakeUpstream(
        fail=GatewayError(529, "overloaded_error", "Busy"), fail_after=2
    )
    async with ModelGateway(_ROUTE, upstream=upstream) as gateway:
        status, _, body = await _post(gateway, {"messages": [], "stream": True})
    assert status == 200
    assert body.endswith(
        b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Busy"}}\n\n'
    )


async def test_our_own_bug_is_a_500_that_logs_only_its_type(caplog):
    upstream = _FakeUpstream(fail=ValueError(_SECRET_PROMPT))
    caplog.set_level(logging.DEBUG)
    async with ModelGateway(_ROUTE, upstream=upstream) as gateway:
        status, _, body = await _post(
            gateway,
            {"messages": [{"role": "user", "content": _SECRET_PROMPT}], "stream": True},
        )
    assert status == 500
    assert b"The model gateway failed" in body
    assert "ValueError" in caplog.text
    assert _SECRET_PROMPT not in caplog.text


async def test_the_request_log_line_carries_no_content(caplog):
    caplog.set_level(logging.DEBUG)
    async with ModelGateway(_ROUTE, upstream=_FakeUpstream()) as gateway:
        await _post(
            gateway,
            {"messages": [{"role": "user", "content": _SECRET_PROMPT}], "stream": True},
        )
    assert "model gateway: adapter=anthropic_passthrough" in caplog.text
    assert _SECRET_PROMPT not in caplog.text
    assert gateway.auth_token not in caplog.text


async def test_a_non_streaming_request_returns_the_whole_message():
    upstream = _FakeUpstream()
    async with ModelGateway(_ROUTE, upstream=upstream) as gateway:
        status, _, body = await _post(gateway, {"messages": [], "stream": False})
    assert status == 200
    assert (
        body
        == b'{"type": "message", "content": [], "usage": {"input_tokens": 7, "output_tokens": 2}}'
    )
    assert gateway.usage == GatewayUsage(input_tokens=7, output_tokens=2)


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]"])
async def test_a_body_that_is_not_a_json_object_is_a_400(raw):
    async with ModelGateway(_ROUTE, upstream=_FakeUpstream()) as gateway:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{gateway.base_url}/v1/messages",
                data=raw,
                headers={
                    "Authorization": f"Bearer {gateway.auth_token}",
                    "Content-Type": "application/json",
                },
            ) as response:
                assert response.status == 400
                assert (await response.json())["error"][
                    "type"
                ] == "invalid_request_error"


async def test_a_quiet_upstream_gets_pings(monkeypatch):
    monkeypatch.setattr(relay, "PING_INTERVAL_SECONDS", 0.05)
    upstream = _FakeUpstream(pause=0.3)
    async with ModelGateway(_ROUTE, upstream=upstream) as gateway:
        _, _, body = await _post(gateway, {"messages": [], "stream": True})
    assert b'event: ping\ndata: {"type":"ping"}' in body
    assert b"message_stop" in body


async def test_a_cli_that_hangs_up_closes_the_upstream_stream():
    upstream = _FakeUpstream(_answer() * 50, pause=0.05)
    async with ModelGateway(_ROUTE, upstream=upstream) as gateway:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{gateway.base_url}/v1/messages",
                json={"messages": [], "stream": True},
                headers={"Authorization": f"Bearer {gateway.auth_token}"},
            )
            await response.content.readline()
            response.close()
        for _ in range(100):
            if upstream.finished_streams:
                break
            await asyncio.sleep(0.05)
    assert upstream.finished_streams == 1


async def test_a_closed_gateway_stays_closed():
    upstream = _FakeUpstream()
    gateway = ModelGateway(_ROUTE, upstream=upstream)
    with pytest.raises(RuntimeError, match="not running"):
        _ = gateway.base_url
    await gateway.start()
    await gateway.close()
    await gateway.close()
    assert upstream.closed
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        await gateway.start()
