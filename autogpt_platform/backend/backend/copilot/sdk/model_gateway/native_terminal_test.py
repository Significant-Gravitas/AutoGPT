import json

import pytest

from backend.copilot.sdk.model_gateway.stream import Event
from backend.copilot.sdk.model_gateway.testing.canned_upstream import CannedReply
from backend.copilot.sdk.model_gateway.testing.failure_streams import (
    ScriptedByteStream,
    native_events,
    streamed_gateway,
)
from backend.copilot.sdk.model_gateway.testing.regressions import post
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

EXPECTED = GatewayUsage(input_tokens=100, cache_read_tokens=7, output_tokens=17)


@pytest.mark.parametrize("ending", ["eof", "error", "reset"])
async def test_native_late_failure_withholds_terminal_and_keeps_cumulative_usage(
    ending,
):
    events = multiple_deltas()[:-1]
    if ending == "error":
        events.append(
            {"type": "error", "error": {"type": "api_error", "message": "fake failure"}}
        )
    wire = ScriptedByteStream(
        CannedReply.of_sse(events).body, "reset" if ending == "reset" else "eof"
    )
    async with streamed_gateway("anthropic_passthrough", wire) as gateway:
        status, raw = await post(gateway, streaming=True)
    assert status == 200 and "event: error" in raw
    assert "already visible partial" in raw
    received = [
        json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")
    ]
    assert not any(event.get("delta", {}).get("stop_reason") for event in received)
    assert "event: message_stop" not in raw
    assert gateway.usage == GatewayUsage()
    assert gateway.failed_usage == EXPECTED
    assert wire.closed


async def test_native_complete_stream_preserves_delta_order_and_cumulative_usage():
    reply = CannedReply.of_sse(multiple_deltas())
    wire = ScriptedByteStream(reply.body, "open")
    async with streamed_gateway("anthropic_passthrough", wire) as gateway:
        status, raw = await post(gateway, streaming=True)
    assert status == 200 and raw == reply.body.decode()
    assert gateway.usage == EXPECTED
    assert gateway.failed_usage == GatewayUsage()
    assert wire.closed and not wire.read_after_body


def multiple_deltas() -> list[Event]:
    events = native_events("already visible partial")
    return [
        *events[:4],
        {
            "type": "message_delta",
            "delta": {},
            "usage": {"output_tokens": 4, "cache_read_input_tokens": 7},
        },
        events[4],
        {"type": "ping"},
        {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 17}},
        events[5],
    ]
