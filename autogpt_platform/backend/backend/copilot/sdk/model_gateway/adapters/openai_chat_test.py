from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import SecretStr

from backend.copilot.sdk.model_gateway.adapters.openai_chat import OpenAIChat
from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.route import ApiKeyCredential, GatewayRoute
from backend.copilot.sdk.model_gateway.stream import StreamItem
from backend.copilot.sdk.model_gateway.testing import cli_requests
from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.wire import (
    LLAMA_CPP_OVERFLOW,
    VLLM_OVERFLOW,
)
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

_KEY = "sk-vllm-upstream-0002"


def _route(url: str) -> GatewayRoute:
    return GatewayRoute(
        adapter="openai_chat",
        model="qwen-local",
        base_url=f"{url}/v1",
        credential=ApiKeyCredential(api_key=SecretStr(_KEY)),
        context_window=32_768,
    )


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


async def _drain(items: AsyncIterator[StreamItem]) -> list[StreamItem]:
    return [item async for item in items]


async def test_a_streamed_answer_reaches_the_cli_as_anthropic_events():
    reply = CannedReply.of_sse(
        [
            _chunk({"role": "assistant", "content": "Hi"}),
            _chunk({}, "stop"),
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 2}},
        ],
        named=False,
        done=True,
    )
    async with CannedUpstream(reply) as upstream:
        adapter = OpenAIChat(_route(upstream.url))
        items = await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()

    (call,) = upstream.calls
    assert call.path == "/v1/chat/completions"
    assert call.headers["authorization"] == f"Bearer {_KEY}"
    assert call.body["model"] == "qwen-local"
    assert [item.event["type"] for item in items] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert items[-2].usage == GatewayUsage(input_tokens=50, output_tokens=2)


async def test_a_non_streaming_request_is_folded_from_the_stream():
    reply = CannedReply.of_sse(
        [_chunk({"content": "folded"}, "stop")], named=False, done=True
    )
    async with CannedUpstream(reply) as upstream:
        adapter = OpenAIChat(_route(upstream.url))
        message = await adapter.complete(
            {**cli_requests.first_request(), "stream": False}
        )
        await adapter.aclose()
    assert message["content"] == [{"type": "text", "text": "folded"}]
    assert message["stop_reason"] == "end_turn"


@pytest.mark.parametrize(
    "body", [VLLM_OVERFLOW, LLAMA_CPP_OVERFLOW], ids=["vllm", "llama.cpp"]
)
async def test_an_overflow_is_rewritten_before_anything_streams(body):
    async with CannedUpstream(CannedReply.of_json(body, status=400)) as upstream:
        adapter = OpenAIChat(_route(upstream.url))
        with pytest.raises(GatewayError) as caught:
            await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    assert caught.value.status == 400
    assert caught.value.message == "prompt is too long: 40000 tokens > 32768 maximum"


async def test_an_error_sent_as_a_stream_chunk_is_raised():
    reply = CannedReply.of_sse(
        [
            {
                "error": {
                    "message": "request (40000 tokens) exceeds the available context size (32768 tokens)",
                    "type": "exceed_context_size_error",
                }
            }
        ],
        named=False,
    )
    async with CannedUpstream(reply) as upstream:
        adapter = OpenAIChat(_route(upstream.url))
        with pytest.raises(GatewayError) as caught:
            await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    assert caught.value.message.startswith("prompt is too long: 40000 tokens")


async def test_a_stream_that_stops_without_finishing_is_an_error_not_an_answer():
    reply = CannedReply(
        headers={"Content-Type": "text/event-stream"},
        chunks=[b'data: {"choices":[{"index":0,"delta":{"content":"half"}}]}\n\n'],
    )
    async with CannedUpstream(reply) as upstream:
        adapter = OpenAIChat(_route(upstream.url))
        relayed: list[StreamItem] = []
        with pytest.raises(GatewayError) as caught:
            async for item in adapter.stream(cli_requests.first_request()):
                relayed.append(item)
        await adapter.aclose()
    assert relayed, "the partial text was relayed before the failure"
    assert caught.value.status == 502


async def test_an_auth_failure_keeps_its_status_and_hides_the_key():
    body = {
        "error": {
            "message": f"Incorrect API key provided: {_KEY}",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }
    async with CannedUpstream(CannedReply.of_json(body, status=401)) as upstream:
        adapter = OpenAIChat(_route(upstream.url))
        with pytest.raises(GatewayError) as caught:
            await _drain(adapter.stream(cli_requests.first_request()))
        await adapter.aclose()
    assert caught.value.status == 401
    assert _KEY not in caught.value.message
