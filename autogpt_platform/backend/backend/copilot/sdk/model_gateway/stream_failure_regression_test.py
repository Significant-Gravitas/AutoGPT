import asyncio
import json

import httpx
import pytest

from backend.copilot.sdk.model_gateway.adapters.anthropic_passthrough import (
    AnthropicPassthrough,
)
from backend.copilot.sdk.model_gateway.adapters.codex_responses import CodexResponses
from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.route import GatewayRoute
from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.codex_lease import (
    codex_route,
    fake_backend_client,
)
from backend.copilot.sdk.model_gateway.testing.failure_streams import (
    ScriptedByteStream,
    native_events,
    streamed_gateway,
)
from backend.copilot.sdk.model_gateway.testing.regressions import (
    BODY,
    USAGE,
    chat_reply,
    chunk,
    gateway_for,
    post,
    terminal,
    text_events,
)
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

CHAT_USAGE = {
    "prompt_tokens": 100,
    "prompt_tokens_details": {"cached_tokens": 40},
    "completion_tokens": 50,
    "completion_tokens_details": {"reasoning_tokens": 30},
}
EXPECTED = GatewayUsage(
    input_tokens=60, cache_read_tokens=40, output_tokens=50, reasoning_tokens=30
)


@pytest.mark.parametrize("through_usage_delta", [False, True])
async def test_native_eof_is_failed_and_retry_is_successful(through_usage_delta):
    partial = CannedReply.of_sse(
        native_events()[:-1] if through_usage_delta else native_events()[:2]
    )
    success = CannedReply.of_sse(native_events("retry succeeded"))
    async with CannedUpstream(partial, success) as upstream, gateway_for(
        "anthropic_passthrough", upstream
    ) as gateway:
        status, raw = await post(gateway, streaming=True)
        assert (
            status == 200 and "event: error" in raw and "event: message_stop" not in raw
        )
        assert gateway.usage == GatewayUsage()
        assert gateway.failed_usage == GatewayUsage(
            input_tokens=100, output_tokens=9 if through_usage_delta else 0
        )
        await post(gateway, streaming=True)
    assert gateway.usage == GatewayUsage(input_tokens=100, output_tokens=9)
    assert gateway.requests == 2


async def test_native_reset_attaches_observed_usage_to_error():
    wire = ScriptedByteStream(CannedReply.of_sse(native_events()[:2]).body, "reset")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=wire)),
        trust_env=False,
    ) as client:
        route = GatewayRoute(
            adapter="anthropic_passthrough", model="m", base_url="http://127.0.0.1:9"
        )
        with pytest.raises(GatewayError) as caught:
            await anext(
                _drain_failure(AnthropicPassthrough(route, client=client), BODY)
            )
    assert caught.value.usage == GatewayUsage(input_tokens=100)
    assert wire.closed


async def _drain_failure(adapter: AnthropicPassthrough, body: dict[str, object]):
    async for _ in adapter.stream(body):
        pass
    yield None


@pytest.mark.parametrize("streaming", [False, True])
async def test_malformed_chat_tool_usage_is_failed_before_successful_retry(streaming):
    bad = chat_reply(
        [
            chunk({"content": "Let me check."}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_bad",
                            "function": {"name": "read_doc", "arguments": '{"page":'},
                        }
                    ]
                },
                "tool_calls",
            ),
            {"choices": [], "usage": CHAT_USAGE},
        ]
    )
    good = chat_reply(
        [
            chunk({"content": "retry succeeded"}, "stop"),
            {"choices": [], "usage": CHAT_USAGE},
        ]
    )
    async with CannedUpstream(bad, good) as upstream, gateway_for(
        "openai_chat", upstream
    ) as gateway:
        status, raw = await post(gateway, streaming=streaming)
        assert status == (200 if streaming else 502) and "invalid tool arguments" in raw
        assert gateway.usage == GatewayUsage() and gateway.failed_usage == EXPECTED
        await post(gateway, streaming=streaming)
    assert gateway.usage == EXPECTED and gateway.failed_usage == EXPECTED


@pytest.mark.parametrize("adapter", ["openai_chat", "codex_responses"])
@pytest.mark.parametrize("ending", ["eof", "reset", "error"])
async def test_adapter_error_exits_keep_received_usage(adapter, ending):
    if adapter == "openai_chat":
        events = [chunk({"content": "partial"}), {"choices": [], "usage": CHAT_USAGE}]
        if ending == "error":
            events.append(
                {"error": {"message": "fake failure", "code": "server_error"}}
            )
        reply = CannedReply.of_sse(events, named=False)
    else:
        events = [
            *text_events(),
            {"type": "response.in_progress", "response": {"usage": USAGE}},
        ]
        if ending == "error":
            events.append(
                {"type": "error", "message": "fake failure", "code": "server_error"}
            )
        reply = CannedReply.of_sse(events)
    wire = ScriptedByteStream(reply.body, "reset" if ending == "reset" else "eof")
    async with streamed_gateway(adapter, wire) as gateway:
        _, raw = await post(gateway, streaming=True)
    assert "event: error" in raw
    assert gateway.usage == GatewayUsage() and gateway.failed_usage == EXPECTED


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", ["response.completed", "response.incomplete"])
@pytest.mark.parametrize("ending", ["reset", "open"])
async def test_codex_terminal_completes_without_reading_past_it(
    streaming, kind, ending
):
    fields = (
        {"incomplete_details": {"reason": "max_output_tokens"}}
        if kind == "response.incomplete"
        else {}
    )
    wire = ScriptedByteStream(
        CannedReply.of_sse([*text_events(), terminal(kind, **fields)]).body, ending
    )
    async with streamed_gateway("codex_responses", wire) as gateway:
        async with asyncio.timeout(1):
            status, raw = await post(gateway, streaming=streaming)
    assert status == 200 and "event: error" not in raw
    assert (
        "message_stop" in raw
        if streaming
        else json.loads(raw)["stop_reason"] == ("max_tokens" if fields else "end_turn")
    )
    assert gateway.usage == EXPECTED and gateway.failed_usage == GatewayUsage()
    assert wire.closed and not wire.read_after_body


async def test_codex_reset_header_reaches_last_failure():
    reply = CannedReply.of_json(
        {"error": {"type": "usage_limit_reached", "message": "fake limit"}},
        status=429,
        **{"x-ratelimit-reset-requests": "1999999999"}
    )
    async with CannedUpstream(reply) as upstream, gateway_for(
        "codex_responses", upstream
    ) as gateway:
        status, _ = await post(gateway, streaming=False)
    assert status == 429 and gateway.last_failure is not None
    assert gateway.last_failure.resets_at == 1999999999


async def test_codex_session_header_is_stable_for_one_adapter():
    reply = CannedReply.of_sse([*text_events(), terminal()])
    async with CannedUpstream(reply, reply, reply) as upstream, fake_backend_client(
        upstream.url + "/codex"
    ) as client:
        route = codex_route("m")
        adapter = CodexResponses(route, client=client)
        await adapter.complete(BODY)
        await adapter.complete(BODY)
        await CodexResponses(route, client=client).complete(BODY)
    sessions = [call.headers["session_id"] for call in upstream.calls]
    assert sessions[0] == sessions[1] and sessions[0] != sessions[2]
