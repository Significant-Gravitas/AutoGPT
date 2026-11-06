from typing import Any

from backend.copilot.sdk.model_gateway.adapters.openai_chat_stream import (
    ChatStreamMapper,
    chat_usage,
)
from backend.copilot.sdk.model_gateway.stream import AnthropicStreamBuilder, Event
from backend.copilot.sdk.model_gateway.usage import GatewayUsage


def _mapper(names: dict[str, str] | None = None) -> ChatStreamMapper:
    return ChatStreamMapper(
        AnthropicStreamBuilder(model="m", input_estimate=10), names or {}
    )


def _chunk(
    delta: dict[str, Any] | None = None, finish: str | None = None
) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def _run(mapper: ChatStreamMapper, *chunks: dict[str, Any]) -> list[Event]:
    events: list[Event] = []
    for chunk in chunks:
        events += mapper.feed(chunk)
    return events + mapper.finish()


def _blocks(events: list[Event]) -> list[Any]:
    return [e["content_block"] for e in events if e["type"] == "content_block_start"]


def _deltas(events: list[Event]) -> list[Any]:
    return [e["delta"] for e in events if e["type"] == "content_block_delta"]


def _stop(events: list[Event]) -> Any:
    return next(e for e in events if e["type"] == "message_delta")["delta"]


def test_reasoning_then_text_become_a_thinking_block_then_a_text_block():
    events = _run(
        _mapper(),
        _chunk({"role": "assistant", "reasoning_content": "Hmm."}),
        _chunk({"reasoning": " Right."}),
        _chunk({"content": "Hel"}),
        _chunk({"content": "lo"}, "stop"),
    )
    assert [b["type"] for b in _blocks(events)] == ["thinking", "text"]
    assert _deltas(events) == [
        {"type": "thinking_delta", "thinking": "Hmm."},
        {"type": "thinking_delta", "thinking": " Right."},
        {"type": "text_delta", "text": "Hel"},
        {"type": "text_delta", "text": "lo"},
    ]
    assert _stop(events)["stop_reason"] == "end_turn"


def test_a_tool_call_split_across_chunks_is_emitted_whole_with_its_id_and_name():
    names = {"mcp__copilot__read_doc": "mcp__copilot__read_doc"}
    events = _run(
        _mapper(names),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_7",
                        "type": "function",
                        "function": {"name": "mcp__copilot__read_doc", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ge": 2}'}}]}),
        _chunk({}, "tool_calls"),
    )
    assert _blocks(events) == [
        {
            "type": "tool_use",
            "id": "call_7",
            "name": "mcp__copilot__read_doc",
            "input": {},
        }
    ]
    assert _deltas(events) == [
        {"type": "input_json_delta", "partial_json": '{"page": 2}'}
    ]
    assert _stop(events)["stop_reason"] == "tool_use"


def test_parallel_calls_keep_their_order_and_stand_in_names_map_back():
    names = {"tool_ab12": "mcp__weird.name"}
    events = _run(
        _mapper(names),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 1,
                        "id": "c2",
                        "function": {"name": "other", "arguments": "{}"},
                    }
                ]
            }
        ),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "function": {"name": "tool_ab12", "arguments": {"a": 1}},
                    }
                ]
            }
        ),
        _chunk({}, "stop"),
    )
    assert [(b["id"], b["name"]) for b in _blocks(events)] == [
        ("c1", "mcp__weird.name"),
        ("c2", "other"),
    ]
    assert _deltas(events)[0] == {
        "type": "input_json_delta",
        "partial_json": '{"a": 1}',
    }
    # "stop" with tool calls is still a tool_use turn.
    assert _stop(events)["stop_reason"] == "tool_use"


def test_a_call_without_an_id_or_arguments_still_reaches_the_cli():
    events = _run(
        _mapper(),
        _chunk({"tool_calls": [{"function": {"name": "t"}}]}),
        _chunk({}, "tool_calls"),
    )
    (block,) = _blocks(events)
    assert block["id"].startswith("toolu_")
    assert _deltas(events) == [{"type": "input_json_delta", "partial_json": "{}"}]


def test_length_and_content_filter_stops():
    assert (
        _stop(_run(_mapper(), _chunk({"content": "x"}, "length")))["stop_reason"]
        == "max_tokens"
    )
    assert (
        _stop(_run(_mapper(), _chunk({}, "content_filter")))["stop_reason"]
        == "end_turn"
    )


def test_usage_moves_the_cached_share_out_of_input():
    mapper = _mapper()
    events = _run(
        mapper,
        _chunk({"content": "x"}, "stop"),
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
        },
    )
    assert mapper.usage == GatewayUsage(
        input_tokens=400, cache_read_tokens=600, output_tokens=20
    )
    assert next(e for e in events if e["type"] == "message_delta")["usage"] == {
        "input_tokens": 400,
        "output_tokens": 20,
        "cache_read_input_tokens": 600,
        "cache_creation_input_tokens": 0,
    }


def test_usage_reads_cache_writes_and_reasoning_where_servers_report_them():
    assert chat_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 300},
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    ) == GatewayUsage(
        input_tokens=600,
        cache_read_tokens=100,
        cache_write_tokens=300,
        output_tokens=50,
        reasoning_tokens=30,
    )
    assert (
        chat_usage(
            {"prompt_tokens": 10, "cache_creation_input_tokens": 4}
        ).cache_write_tokens
        == 4
    )


def test_a_response_with_nothing_but_a_finish_is_complete():
    mapper = _mapper()
    assert not mapper.complete
    mapper.feed(_chunk({}, "stop"))
    assert mapper.complete
