from collections.abc import AsyncIterator

from backend.copilot.sdk.model_gateway.stream import (
    AnthropicStreamBuilder,
    Event,
    StreamItem,
    closing_items,
    collect_message,
)
from backend.copilot.sdk.model_gateway.usage import GatewayUsage


def _types(events: list[Event]) -> list[str]:
    return [str(e["type"]) for e in events]


def _blocks(events: list[Event]) -> list[object]:
    return [e["content_block"] for e in events if e["type"] == "content_block_start"]


def test_message_start_waits_for_the_first_content_with_zero_provisional_input():
    builder = AnthropicStreamBuilder(model="m", input_estimate=1234)
    assert builder.text("") == []

    events = builder.text("hi")

    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
    ]
    message = events[0]["message"]
    assert isinstance(message, dict)
    assert str(message.pop("id")).startswith("msg_gateway_")
    assert message == {
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def test_blocks_open_and_close_in_order_with_rising_indexes():
    builder = AnthropicStreamBuilder(model="m", input_estimate=1)
    events = (
        builder.thinking("why")
        + builder.signature("sig")
        + builder.text("answer")
        + builder.tool_use("call_1", "lookup", '{"q": 1}')
        + builder.finish("tool_use", GatewayUsage(input_tokens=5, output_tokens=7))
    )

    starts = [e["index"] for e in events if e["type"] == "content_block_start"]
    stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
    assert starts == stops == [0, 1, 2]
    assert _blocks(events) == [
        {"type": "thinking", "thinking": "", "signature": ""},
        {"type": "text", "text": ""},
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
    ]
    delta = next(e for e in events if e["type"] == "message_delta")
    assert delta["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    assert delta["usage"] == {
        "input_tokens": 5,
        "output_tokens": 7,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert events[-1] == {"type": "message_stop"}
    assert builder.tool_used


def test_a_signature_after_text_seals_a_thinking_block_of_its_own():
    builder = AnthropicStreamBuilder(model="m", input_estimate=1)
    events = builder.text("a") + builder.signature("enc")

    assert _types(events) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    assert _blocks(events)[-1] == {"type": "thinking", "thinking": "", "signature": ""}
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert deltas[-1] == {"type": "signature_delta", "signature": "enc"}
    assert events[-1] == {"type": "content_block_stop", "index": 1}


def test_an_empty_response_still_frames_a_message():
    events = AnthropicStreamBuilder(model="m", input_estimate=3).finish(
        "end_turn", None
    )
    assert _types(events) == ["message_start", "message_delta", "message_stop"]
    assert events[1]["usage"] == GatewayUsage(input_tokens=3).anthropic()


def test_the_exact_usage_rides_only_on_the_message_delta():
    usage = GatewayUsage(input_tokens=2, reasoning_tokens=9)
    items = closing_items(
        [{"type": "content_block_stop", "index": 0}, {"type": "message_delta"}],
        usage,
    )
    assert [item.usage for item in items] == [None, usage]


async def _replay(events: list[Event]) -> AsyncIterator[StreamItem]:
    for event in events:
        yield StreamItem(event=event)


async def test_a_stream_folds_into_the_non_streaming_message():
    builder = AnthropicStreamBuilder(model="m", input_estimate=40)
    events = [
        *builder.thinking("plan"),
        *builder.signature("sig-1"),
        *builder.text("Hel"),
        *builder.text("lo"),
    ]
    tool = builder.tool_use("call_9", "lookup", '{"q": 1}')
    # A tool's JSON may arrive split; the fold joins it before parsing.
    tool[-2]["delta"]["partial_json"] = '{"q": '
    tool.insert(
        -1,
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": "1}"},
        },
    )
    events += tool
    events += builder.finish("tool_use", GatewayUsage(input_tokens=30, output_tokens=4))

    message = await collect_message(_replay(events))

    assert message["content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
        {"type": "text", "text": "Hello"},
        {"type": "tool_use", "id": "call_9", "name": "lookup", "input": {"q": 1}},
    ]
    assert message["stop_reason"] == "tool_use"
    assert message["usage"] == {
        "input_tokens": 30,
        "output_tokens": 4,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
