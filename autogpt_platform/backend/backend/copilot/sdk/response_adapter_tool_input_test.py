"""The SDK adapter opens a tool part when the model starts the call.

With partial messages on, the CLI relays ``content_block_start`` (tool_use)
and ``input_json_delta`` events ahead of the summary ``AssistantMessage``.
"""

import json
from unittest.mock import MagicMock, patch

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from backend.copilot.capabilities.dispatch import DispatchedToolCall
from backend.copilot.response_model import (
    StreamFinishStep,
    StreamStartStep,
    StreamToolInputAvailable,
    StreamToolInputDelta,
    StreamToolInputError,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.tool_input_stream import ToolInputStreams

from .response_adapter import SDKResponseAdapter
from .tool_adapter import MCP_TOOL_PREFIX

_ARGS = {"query": "weather in Paris", "limit": 3}


class _Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


def _adapter(*, stream_tool_inputs: bool = True) -> SDKResponseAdapter:
    adapter = SDKResponseAdapter(
        message_id="msg-1",
        session_id="session-1",
        stream_tool_inputs=stream_tool_inputs,
    )
    adapter.tool_inputs = ToolInputStreams(clock=_Clock())
    adapter.convert_message(SystemMessage(subtype="init", data={}))
    return adapter


def _event(payload: dict) -> StreamEvent:
    return StreamEvent(
        uuid="evt", session_id="session-1", parent_tool_use_id=None, event=payload
    )


def _tool_use_start(call_id: str, name: str, index: int = 0) -> StreamEvent:
    return _event(
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": call_id, "name": name},
        }
    )


def _args_delta(text: str, index: int = 0) -> StreamEvent:
    return _event(
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": text},
        }
    )


def _block_stop(index: int = 0) -> StreamEvent:
    return _event({"type": "content_block_stop", "index": index})


def _summary(call_id: str, name: str, args: dict) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id=call_id, name=name, input=args)], model="test"
    )


def _stream_call(adapter, call_id, name, args, *, pieces=4):
    """Feed one tool call's partial events; return what each step emitted."""
    text = json.dumps(args)
    size = max(1, len(text) // pieces)
    at_start = adapter.convert_message(_event({"type": "message_start"}))
    at_start += adapter.convert_message(_tool_use_start(call_id, name))
    during = []
    for i in range(0, len(text), size):
        during += adapter.convert_message(_args_delta(text[i : i + size]))
    during += adapter.convert_message(_block_stop())
    return at_start, during


def _deltas(events) -> str:
    return "".join(
        e.inputTextDelta for e in events if isinstance(e, StreamToolInputDelta)
    )


def test_the_part_opens_before_the_arguments_are_written():
    adapter = _adapter()

    at_start, during = _stream_call(
        adapter, "t1", f"{MCP_TOOL_PREFIX}web_search", _ARGS
    )
    at_summary = adapter.convert_message(
        _summary("t1", f"{MCP_TOOL_PREFIX}web_search", _ARGS)
    )

    assert at_start == [StreamToolInputStart(toolCallId="t1", toolName="web_search")]
    assert not any(isinstance(e, StreamToolInputAvailable) for e in during)
    assert not any(isinstance(e, StreamToolInputStart) for e in at_summary)
    available = [e for e in at_summary if isinstance(e, StreamToolInputAvailable)]
    assert available == [
        StreamToolInputAvailable(toolCallId="t1", toolName="web_search", input=_ARGS)
    ]
    assert _deltas(during + at_summary) == json.dumps(_ARGS)


def test_argument_deltas_are_coalesced_and_in_order():
    adapter = _adapter()
    args = {"path": "notes.md", "content": "line\n" * 200}

    _, during = _stream_call(adapter, "t1", "Write", args, pieces=400)

    deltas = [e for e in during if isinstance(e, StreamToolInputDelta)]
    assert _deltas(during) == json.dumps(args)
    assert 1 < len(deltas) < 50


def test_a_dispatcher_call_starts_under_the_tool_it_runs():
    adapter = _adapter()
    dispatched = DispatchedToolCall(MagicMock(), "read_skill", {"name": "brief"})
    args = {"id": "skill:brief"}

    with patch(
        "backend.copilot.sdk.response_adapter.resolve_tool_dispatch",
        return_value=dispatched,
    ):
        at_start, during = _stream_call(
            adapter, "t1", f"{MCP_TOOL_PREFIX}run_capability", args
        )
        at_summary = adapter.convert_message(
            _summary("t1", f"{MCP_TOOL_PREFIX}run_capability", args)
        )

    assert not any(isinstance(e, StreamToolInputStart) for e in at_start + during)
    assert not any(isinstance(e, StreamToolInputDelta) for e in during)
    tool_events = [
        e
        for e in at_summary
        if isinstance(e, (StreamToolInputStart, StreamToolInputAvailable))
    ]
    assert tool_events == [
        StreamToolInputStart(toolCallId="t1", toolName="read_skill"),
        StreamToolInputAvailable(
            toolCallId="t1", toolName="read_skill", input={"name": "brief"}
        ),
    ]


def test_a_refused_call_still_ends_with_its_output():
    adapter = _adapter()
    _stream_call(adapter, "t1", f"{MCP_TOOL_PREFIX}web_search", _ARGS)
    adapter.convert_message(_summary("t1", f"{MCP_TOOL_PREFIX}web_search", _ARGS))

    refused = adapter.convert_message(
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="t1", content="Refused", is_error=True)
            ]
        )
    )

    outputs = [e for e in refused if isinstance(e, StreamToolOutputAvailable)]
    assert [(o.toolCallId, o.success) for o in outputs] == [("t1", False)]
    assert not adapter.has_open_tool_input


def test_a_call_cut_off_mid_arguments_is_closed_before_the_step_ends():
    adapter = _adapter()
    adapter.convert_message(_event({"type": "message_start"}))
    adapter.convert_message(_tool_use_start("t1", f"{MCP_TOOL_PREFIX}web_search"))
    adapter.convert_message(_args_delta('{"query": "wea'))
    assert adapter.has_open_tool_input

    ended = adapter.convert_message(
        ResultMessage(
            subtype="error",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="session-1",
            result="stream died",
        )
    )

    errors = [e for e in ended if isinstance(e, StreamToolInputError)]
    assert [(e.toolCallId, e.toolName) for e in errors] == [("t1", "web_search")]
    assert ended.index(errors[0]) < next(
        i for i, e in enumerate(ended) if isinstance(e, StreamFinishStep)
    )
    assert not adapter.has_open_tool_input


def test_a_call_after_a_tool_result_opens_its_step_first():
    adapter = _adapter()
    _stream_call(adapter, "t1", "Read", {"file_path": "a"})
    adapter.convert_message(_summary("t1", "Read", {"file_path": "a"}))
    adapter.convert_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")])
    )

    at_start, _ = _stream_call(adapter, "t2", "Read", {"file_path": "b"})
    at_summary = adapter.convert_message(_summary("t2", "Read", {"file_path": "b"}))

    assert [type(e) for e in at_start] == [StreamStartStep, StreamToolInputStart]
    assert not any(isinstance(e, StreamStartStep) for e in at_summary)


def test_a_result_while_a_sibling_streams_keeps_the_step_open():
    adapter = _adapter()
    _stream_call(adapter, "t1", "Read", {"file_path": "a"})
    adapter.convert_message(_summary("t1", "Read", {"file_path": "a"}))
    adapter.convert_message(_tool_use_start("t2", "Read", index=1))

    result = adapter.convert_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")])
    )
    at_summary = adapter.convert_message(_summary("t2", "Read", {"file_path": "b"}))

    assert not any(
        isinstance(e, (StreamFinishStep, StreamToolInputError)) for e in result
    )
    assert not any(isinstance(e, StreamStartStep) for e in at_summary)
    assert [type(e) for e in at_summary][-1] is StreamToolInputAvailable


def test_codex_calls_keep_their_start_and_input_together():
    adapter = _adapter(stream_tool_inputs=False)

    at_start, during = _stream_call(
        adapter, "t1", f"{MCP_TOOL_PREFIX}web_search", _ARGS
    )
    at_summary = adapter.convert_message(
        _summary("t1", f"{MCP_TOOL_PREFIX}web_search", _ARGS)
    )

    assert at_start == []
    assert during == []
    assert [type(e) for e in at_summary] == [
        StreamToolInputStart,
        StreamToolInputAvailable,
    ]
