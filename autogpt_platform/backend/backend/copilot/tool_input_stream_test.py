"""A tool call's part opens at its name, streams its arguments, closes on its input."""

import json

import pytest

from backend.copilot.response_model import (
    StreamToolInputAvailable,
    StreamToolInputDelta,
    StreamToolInputError,
    StreamToolInputStart,
)
from backend.copilot.tool_input_stream import (
    _FLUSH_CHARS,
    _FLUSH_INTERVAL_S,
    _MAX_STREAMED_CHARS,
    ToolInputStreams,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock() -> _Clock:
    return _Clock()


def _deltas(events) -> list[str]:
    return [e.inputTextDelta for e in events if isinstance(e, StreamToolInputDelta)]


def test_start_goes_out_before_any_argument(clock):
    streams = ToolInputStreams(clock=clock)

    events = streams.start("c1", "web_search")

    assert events == [StreamToolInputStart(toolCallId="c1", toolName="web_search")]
    assert streams.has_open


def test_a_second_start_for_an_open_call_sends_nothing(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "web_search")

    assert streams.start("c1", "web_search") == []


def test_small_pieces_coalesce_until_the_char_threshold(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "write_file")
    piece = "x" * 10
    sent = []
    for _ in range(_FLUSH_CHARS // len(piece) - 1):
        sent += streams.append("c1", piece)
    assert sent == []

    sent += streams.append("c1", piece)

    assert _deltas(sent) == [piece * (_FLUSH_CHARS // len(piece))]


def test_a_piece_after_the_interval_flushes_what_is_buffered(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "web_search")
    assert streams.append("c1", '{"query":') == []

    clock.now += _FLUSH_INTERVAL_S * 2
    events = streams.append("c1", '"x"}')

    assert _deltas(events) == ['{"query":"x"}']


def test_completing_sends_the_tail_then_the_input_in_order(clock):
    streams = ToolInputStreams(clock=clock)
    events = streams.start("c1", "web_search")
    args = '{"query": "weather in Paris", "limit": 3}'
    for char in args:
        events += streams.append("c1", char)
        clock.now += 0.01
    events += streams.complete("c1", "web_search", json.loads(args))

    assert isinstance(events[0], StreamToolInputStart)
    assert isinstance(events[-1], StreamToolInputAvailable)
    assert events[-1].input == {"query": "weather in Paris", "limit": 3}
    assert "".join(_deltas(events)) == args
    assert len(_deltas(events)) < len(args)
    assert not streams.has_open


def test_a_call_never_started_completes_with_a_start_first(clock):
    streams = ToolInputStreams(clock=clock)

    events = streams.complete("c1", "find_block", {"query": "x"})

    assert [type(e) for e in events] == [
        StreamToolInputStart,
        StreamToolInputAvailable,
    ]


def test_the_dispatcher_is_held_until_its_input_names_the_tool(clock):
    streams = ToolInputStreams(clock=clock)

    assert streams.start("c1", "run_capability") == []
    assert streams.append("c1", '{"id": "skill:x"}') == []
    events = streams.complete("c1", "read_skill", {"name": "x"})

    assert events[0] == StreamToolInputStart(toolCallId="c1", toolName="read_skill")
    assert events[1].toolName == "read_skill"


def test_streamed_text_stops_at_the_cap_but_the_input_is_whole(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "write_file")
    content = "y" * (_MAX_STREAMED_CHARS * 2)

    events = streams.append("c1", content)
    events += streams.append("c1", "more")
    events += streams.complete("c1", "write_file", {"content": content})

    assert len("".join(_deltas(events))) == _MAX_STREAMED_CHARS
    assert events[-1].input == {"content": content}


def test_flush_sends_buffered_text_and_keeps_the_call_open(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "web_search")
    streams.append("c1", '{"q"')

    assert _deltas(streams.flush()) == ['{"q"']
    assert streams.has_open


def test_settle_readies_a_refused_call_for_its_output(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "web_search")
    streams.append("c1", '{"q"')

    assert _deltas(streams.settle("c1", "web_search")) == ['{"q"']
    assert streams.settle("c2", "web_search") == [
        StreamToolInputStart(toolCallId="c2", toolName="web_search")
    ]
    assert not streams.has_open


def test_abandon_errors_every_open_part_once(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "web_search")
    streams.start("c2", "write_file")
    streams.complete("c2", "write_file", {})

    events = streams.abandon()

    assert [(type(e), e.toolCallId) for e in events] == [(StreamToolInputError, "c1")]
    assert events[0].toolName == "web_search"
    assert streams.abandon() == []


def test_new_chunks_use_the_ai_sdk_wire_shape(clock):
    streams = ToolInputStreams(clock=clock)
    streams.start("c1", "web_search")
    clock.now += _FLUSH_INTERVAL_S * 2
    (delta,) = streams.append("c1", "{")
    (error,) = streams.abandon()

    assert json.loads(delta.to_sse().removeprefix("data: ")) == {
        "type": "tool-input-delta",
        "toolCallId": "c1",
        "inputTextDelta": "{",
    }
    assert json.loads(error.to_sse().removeprefix("data: ")) == {
        "type": "tool-input-error",
        "toolCallId": "c1",
        "toolName": "web_search",
        "input": {},
        "errorText": error.errorText,
    }
