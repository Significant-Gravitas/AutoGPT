import json

import pytest

from backend.copilot.sdk.model_gateway.testing.canned_upstream import (
    CannedReply,
    CannedUpstream,
)
from backend.copilot.sdk.model_gateway.testing.regressions import (
    chat_reply,
    chunk,
    gateway_for,
    post,
    terminal,
    text_events,
)
from backend.copilot.sdk.model_gateway.usage import (
    GatewayUsage,
    merge_stream_usage,
    usage_from_anthropic,
)


async def test_output_only_usage_keeps_an_input_estimate():
    reply = chat_reply(
        [
            chunk({"content": "answer"}, "stop"),
            {"choices": [], "usage": {"completion_tokens": 2}},
        ]
    )
    async with CannedUpstream(reply) as upstream, gateway_for(
        "openai_chat", upstream
    ) as gateway:
        status, raw = await post(gateway, streaming=False)
    assert status == 200
    assert json.loads(raw)["usage"]["input_tokens"] == 101
    assert gateway.usage.input_tokens == 101


def test_explicit_zero_replaces_start_while_absent_fields_survive():
    start = GatewayUsage(input_tokens=100, cache_read_tokens=5)
    delta = usage_from_anthropic({"input_tokens": 0, "output_tokens": 2})
    assert delta is not None
    assert merge_stream_usage(start, delta) == GatewayUsage(
        input_tokens=0, cache_read_tokens=5, output_tokens=2
    )


async def test_fully_cached_nonstream_usage_is_not_the_estimate():
    reply = chat_reply(
        [
            chunk({"content": "answer"}, "stop"),
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 100},
                    "completion_tokens": 2,
                },
            },
        ]
    )
    async with CannedUpstream(reply) as upstream, gateway_for(
        "openai_chat", upstream
    ) as gateway:
        status, raw = await post(gateway, streaming=False)
    assert status == 200
    assert json.loads(raw)["usage"]["input_tokens"] == 0
    assert gateway.usage == GatewayUsage(cache_read_tokens=100, output_tokens=2)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("after_output", [False, True])
async def test_failed_codex_usage_is_separate_from_the_successful_retry(
    streaming, after_output
):
    failed = CannedReply.of_sse(
        [
            *(text_events() if after_output else []),
            terminal(
                "response.failed",
                error={"code": "server_error", "message": "temporary failure"},
            ),
        ]
    )
    success = CannedReply.of_sse([*text_events(), terminal()])
    async with CannedUpstream(failed, success) as upstream, gateway_for(
        "codex_responses", upstream
    ) as gateway:
        status, raw = await post(gateway, streaming=streaming)
        assert status == (200 if streaming and after_output else 500)
        assert "temporary failure" in raw
        await post(gateway, streaming=streaming)
    expected = GatewayUsage(
        input_tokens=60, cache_read_tokens=40, output_tokens=50, reasoning_tokens=30
    )
    assert gateway.usage == expected
    assert gateway.failed_usage == expected
    assert gateway.requests == 2


@pytest.mark.parametrize("adapter", ["openai_chat", "codex_responses"])
async def test_streaming_and_nonstreaming_reasoning_usage_match(adapter):
    reply = (
        CannedReply.of_sse([*text_events(), terminal()])
        if adapter == "codex_responses"
        else chat_reply(
            [
                chunk({"content": "answer"}, "stop"),
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 40},
                        "completion_tokens": 50,
                        "completion_tokens_details": {"reasoning_tokens": 30},
                    },
                },
            ]
        )
    )
    async with CannedUpstream(reply, reply) as upstream, gateway_for(
        adapter, upstream
    ) as gateway:
        await post(gateway, streaming=True)
        first = gateway.usage
        await post(gateway, streaming=False)
    assert first.reasoning_tokens == 30
    assert gateway.usage == first.plus(first)
