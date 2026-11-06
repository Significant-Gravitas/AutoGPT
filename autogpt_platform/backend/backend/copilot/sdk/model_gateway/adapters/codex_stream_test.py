from typing import Any

from backend.copilot.sdk.model_gateway.adapters.codex_signature import (
    reasoning_tag,
    seal,
)
from backend.copilot.sdk.model_gateway.adapters.codex_stream import (
    CodexStreamMapper,
    responses_usage,
)
from backend.copilot.sdk.model_gateway.stream import AnthropicStreamBuilder, Event
from backend.copilot.sdk.model_gateway.testing.script import Decision
from backend.copilot.sdk.model_gateway.testing.wire import responses_events
from backend.copilot.sdk.model_gateway.usage import GatewayUsage

_TAG = reasoning_tag("acct-test", "m")
_ENCRYPTED = "gAAAAAB-" + "e" * 5000


def _mapper(names: dict[str, str] | None = None) -> CodexStreamMapper:
    return CodexStreamMapper(
        AnthropicStreamBuilder(model="gpt", input_estimate=99), names or {}, _TAG
    )


def _run(mapper: CodexStreamMapper, events: list[dict[str, Any]]) -> list[Event]:
    out: list[Event] = []
    for event in events:
        out += mapper.feed(event)
    return out + (mapper.finish() if mapper.completed else [])


def _blocks(events: list[Event]) -> list[Any]:
    return [e["content_block"] for e in events if e["type"] == "content_block_start"]


def _deltas(events: list[Event]) -> list[Any]:
    return [e["delta"] for e in events if e["type"] == "content_block_delta"]


def _stop_reason(events: list[Event]) -> Any:
    delta: Any = next(e for e in events if e["type"] == "message_delta")["delta"]
    return delta["stop_reason"]


def test_a_tool_turn_streams_reasoning_with_a_sealed_signature_then_the_call():
    events = responses_events(
        Decision(kind="tool", page=2),
        model="gpt",
        tool="mcp__h__read_doc",
        n=3,
        input_tokens=800,
        encrypted=_ENCRYPTED,
    )
    mapper = _mapper({"mcp__h__read_doc": "mcp__h__read_doc"})
    out = _run(mapper, events)

    assert _blocks(out) == [
        {"type": "thinking", "thinking": "", "signature": ""},
        {"type": "tool_use", "id": "call_3", "name": "mcp__h__read_doc", "input": {}},
    ]
    assert _deltas(out) == [
        {"type": "thinking_delta", "thinking": "Planning the next step."},
        {"type": "signature_delta", "signature": seal(_ENCRYPTED, _TAG)},
        {"type": "input_json_delta", "partial_json": '{"page": 2}'},
    ]
    assert _stop_reason(out) == "tool_use"
    assert mapper.usage == GatewayUsage(
        input_tokens=600, cache_read_tokens=200, output_tokens=12, reasoning_tokens=3
    )


def test_a_text_turn_ends_the_turn():
    events = responses_events(
        Decision(kind="text", text="All done."),
        model="gpt",
        tool="t",
        n=1,
        input_tokens=40,
        encrypted=_ENCRYPTED,
    )
    out = _run(_mapper(), events)
    assert [b["type"] for b in _blocks(out)] == ["thinking", "text"]
    assert {"type": "text_delta", "text": "All done."} in _deltas(out)
    assert next(e for e in out if e["type"] == "message_delta")["delta"] == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }


def test_nothing_reaches_the_cli_before_the_first_output_item():
    mapper = _mapper()
    assert mapper.feed({"type": "response.created", "response": {"id": "r"}}) == []
    assert mapper.feed({"type": "response.in_progress", "response": {"id": "r"}}) == []
    started = mapper.feed(
        {"type": "response.output_item.added", "item": {"type": "message", "id": "m"}}
    )
    assert [e["type"] for e in started] == ["message_start"]


def test_a_message_item_without_deltas_still_delivers_its_text():
    mapper = _mapper()
    item = {
        "type": "message",
        "id": "m1",
        "content": [{"type": "output_text", "text": "whole"}],
    }
    out = mapper.feed({"type": "response.output_item.done", "item": item})
    assert {"type": "text_delta", "text": "whole"} in _deltas(out)


def test_summary_parts_are_separated_and_a_missing_signature_closes_the_block():
    mapper = _mapper()
    out = mapper.feed(
        {
            "type": "response.output_item.added",
            "item": {"type": "reasoning", "id": "rs"},
        }
    )
    out += mapper.feed(
        {"type": "response.reasoning_summary_text.delta", "delta": "one"}
    )
    out += mapper.feed(
        {"type": "response.reasoning_summary_part.added", "summary_index": 1}
    )
    out += mapper.feed(
        {"type": "response.reasoning_summary_text.delta", "delta": "two"}
    )
    out += mapper.feed(
        {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs"}}
    )
    assert [d.get("thinking") for d in _deltas(out)] == ["one", "\n\n", "two"]
    assert out[-1]["type"] == "content_block_stop"


def test_an_incomplete_response_maps_its_reason():
    mapper = _mapper()
    mapper.feed(
        {"type": "response.output_item.added", "item": {"type": "message", "id": "m"}}
    )
    mapper.feed(
        {
            "type": "response.incomplete",
            "response": {"incomplete_details": {"reason": "max_output_tokens"}},
        }
    )
    assert _stop_reason(mapper.finish()) == "max_tokens"


def test_failures_are_recorded_for_the_adapter_to_raise():
    for event in (
        {
            "type": "response.failed",
            "response": {"error": {"code": "server_error", "message": "boom"}},
        },
        {"type": "error", "code": "rate_limit_exceeded", "message": "slow down"},
    ):
        mapper = _mapper()
        assert mapper.feed(event) == []
        assert mapper.failure is not None
        assert mapper.failure.code in ("server_error", "rate_limit_exceeded")


def test_responses_usage_counts_the_cached_share_once():
    assert responses_usage(
        {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 900},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 40},
        }
    ) == GatewayUsage(
        input_tokens=100, cache_read_tokens=900, output_tokens=50, reasoning_tokens=40
    )
