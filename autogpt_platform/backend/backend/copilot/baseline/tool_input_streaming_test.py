"""The baseline engine opens a tool part while the model writes the call."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from backend.copilot import turn_end
from backend.copilot.baseline.service import (
    _baseline_llm_caller,
    _baseline_tool_executor,
    _BaselineStreamState,
)
from backend.copilot.capabilities.dispatch import DispatchedToolCall
from backend.copilot.model import ChatSession
from backend.copilot.response_model import (
    StreamFinishStep,
    StreamToolInputAvailable,
    StreamToolInputDelta,
    StreamToolInputError,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.tool_input_stream import ToolInputStreams

_ARGS = {"query": "weather in Paris", "limit": 3}
_LONG_ARGS = {"path": "notes.md", "content": "line\n" * 200}


def _state() -> _BaselineStreamState:
    state = _BaselineStreamState(model="anthropic/claude-sonnet-4-6")
    state.tool_inputs = ToolInputStreams(clock=lambda: 10.0)
    return state


def _chunk(*, call_id: str | None, name: str | None, arguments: str):
    tc = MagicMock()
    tc.index = 0
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    delta = ChoiceDelta.model_validate({"role": "assistant"})
    object.__setattr__(delta, "tool_calls", [tc])
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.usage = None
    chunk.choices = [choice]
    return chunk


def _call_chunks(call_id: str, name: str, args: dict, pieces: int = 40):
    text = json.dumps(args)
    size = max(1, len(text) // pieces)
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    return [_chunk(call_id=call_id, name=name, arguments=parts[0])] + [
        _chunk(call_id=None, name=None, arguments=part) for part in parts[1:]
    ]


def _stream(chunks, error: Exception | None = None):
    stream = MagicMock()
    stream.close = AsyncMock()

    async def aiter():
        for chunk in chunks:
            yield chunk
        if error is not None:
            raise error

    stream.__aiter__ = lambda self: aiter()
    return stream


async def _llm_round(state, chunks, error: Exception | None = None):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_stream(chunks, error))
    with patch(
        "backend.copilot.baseline.service._get_main_client", return_value=client
    ):
        return await _baseline_llm_caller(
            messages=[{"role": "user", "content": "hi"}], tools=[], state=state
        )


async def _execute(state, call, session):
    execute = AsyncMock(
        return_value=StreamToolOutputAvailable(toolCallId=call.id, output="ok")
    )
    with patch("backend.copilot.baseline.service.execute_tool", new=execute):
        return await _baseline_tool_executor(
            call,
            [],
            state=state,
            user_id="user-1",
            session=session,
            disabled_groups=[],
            disabled_tools=frozenset(),
        )


def _deltas(events) -> str:
    return "".join(
        e.inputTextDelta for e in events if isinstance(e, StreamToolInputDelta)
    )


@pytest.fixture()
def session():
    session = ChatSession.new("user-1", dry_run=False)
    yield session
    turn_end.clear(session.session_id)


@pytest.mark.asyncio
async def test_the_part_opens_while_the_arguments_stream(session):
    state = _state()

    response = await _llm_round(
        state, _call_chunks("c1", "write_file", _LONG_ARGS, pieces=200)
    )
    streamed = list(state.emitted_events)
    await _execute(state, response.tool_calls[0], session)
    executed = state.emitted_events[len(streamed) :]

    starts = [e for e in streamed if isinstance(e, StreamToolInputStart)]
    assert starts == [StreamToolInputStart(toolCallId="c1", toolName="write_file")]
    finish = next(i for i, e in enumerate(streamed) if isinstance(e, StreamFinishStep))
    assert streamed.index(starts[0]) < finish
    assert _deltas(streamed) == json.dumps(_LONG_ARGS)
    assert 1 < len([e for e in streamed if isinstance(e, StreamToolInputDelta)]) < 10
    assert not any(isinstance(e, StreamToolInputStart) for e in executed)
    available = [e for e in executed if isinstance(e, StreamToolInputAvailable)]
    assert available == [
        StreamToolInputAvailable(
            toolCallId="c1", toolName="write_file", input=_LONG_ARGS
        )
    ]


@pytest.mark.asyncio
async def test_arguments_sent_before_the_name_open_the_part_with_them():
    state = _state()
    chunks = [
        _chunk(call_id="c1", name=None, arguments='{"query": '),
        _chunk(call_id=None, name="web_search", arguments='"x"}'),
    ]

    await _llm_round(state, chunks)

    assert _deltas(state.emitted_events) == '{"query": "x"}'


@pytest.mark.asyncio
async def test_a_dispatcher_call_starts_under_the_tool_it_runs(session):
    state = _state()
    args = {"id": "skill:brief"}
    response = await _llm_round(state, _call_chunks("c1", "run_capability", args))
    streamed = list(state.emitted_events)

    with patch(
        "backend.copilot.baseline.service.resolve_tool_dispatch",
        return_value=DispatchedToolCall(MagicMock(), "read_skill", {"name": "brief"}),
    ):
        await _execute(state, response.tool_calls[0], session)

    assert not any(
        isinstance(e, (StreamToolInputStart, StreamToolInputDelta)) for e in streamed
    )
    tool_events = [
        e
        for e in state.emitted_events
        if isinstance(e, (StreamToolInputStart, StreamToolInputAvailable))
    ]
    assert [(type(e), e.toolName) for e in tool_events] == [
        (StreamToolInputStart, "read_skill"),
        (StreamToolInputAvailable, "read_skill"),
    ]


@pytest.mark.asyncio
async def test_a_call_refused_after_the_question_closes_its_part(session):
    state = _state()
    response = await _llm_round(state, _call_chunks("c1", "web_search", _ARGS))
    turn_end.request_end(session.session_id)

    result = await _execute(state, response.tool_calls[0], session)

    assert result.is_error
    starts = [e for e in state.emitted_events if isinstance(e, StreamToolInputStart)]
    outputs = [
        e for e in state.emitted_events if isinstance(e, StreamToolOutputAvailable)
    ]
    assert len(starts) == 1
    assert [(o.toolCallId, o.success) for o in outputs] == [("c1", False)]
    assert not state.tool_inputs.has_open


@pytest.mark.asyncio
async def test_a_round_cut_off_mid_arguments_ends_the_part_before_the_step():
    state = _state()
    chunks = _call_chunks("c1", "web_search", _ARGS)[:3]

    with pytest.raises(RuntimeError):
        await _llm_round(state, chunks, error=RuntimeError("provider dropped"))

    events = state.emitted_events
    errors = [e for e in events if isinstance(e, StreamToolInputError)]
    assert [(e.toolCallId, e.toolName) for e in errors] == [("c1", "web_search")]
    finish = next(i for i, e in enumerate(events) if isinstance(e, StreamFinishStep))
    assert events.index(errors[0]) < finish
    assert not state.tool_inputs.has_open
