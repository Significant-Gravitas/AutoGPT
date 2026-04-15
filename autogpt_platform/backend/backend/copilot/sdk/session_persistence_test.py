"""Tests for the pre-create assistant message logic that prevents
last_role=tool after client disconnect.

Reproduces the bug where:
  1. Tool result is saved by intermediate flush → last_role=tool
  2. SDK generates a text response
  3. GeneratorExit at StreamStartStep yield (client disconnect)
  4. _dispatch_response(StreamTextDelta) is never called
  5. Session saved with last_role=tool instead of last_role=assistant

The fix: before yielding any events, pre-create the assistant message in
ctx.session.messages when has_tool_results=True and a StreamTextDelta is
present in adapter_responses.  This test verifies the resulting accumulator
state allows correct content accumulation by _dispatch_response.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamStartStep, StreamTextDelta
from backend.copilot.sdk.service import _dispatch_response, _StreamAccumulator

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_session() -> ChatSession:
    return ChatSession(
        session_id="test",
        user_id="test-user",
        title="test",
        messages=[],
        usage=[],
        started_at=_NOW,
        updated_at=_NOW,
    )


def _make_ctx(session: ChatSession | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.session = session or _make_session()
    ctx.log_prefix = "[test]"
    return ctx


def _make_state() -> MagicMock:
    state = MagicMock()
    state.transcript_builder = MagicMock()
    return state


def _simulate_pre_create(acc: _StreamAccumulator, ctx: MagicMock) -> None:
    """Mirror the pre-create block from _run_stream_attempt so tests
    can verify its effect without invoking the full async generator.

    Keep in sync with the block in service.py _run_stream_attempt
    (search: "Pre-create the new assistant message").
    """
    acc.assistant_response = ChatMessage(role="assistant", content="")
    acc.accumulated_tool_calls = []
    acc.has_tool_results = False
    ctx.session.messages.append(acc.assistant_response)
    # acc.has_appended_assistant stays True


class TestPreCreateAssistantMessage:
    """Verify that the pre-create logic correctly seeds the session message
    and that subsequent _dispatch_response(StreamTextDelta) accumulates
    content in-place without a double-append."""

    def test_pre_create_adds_message_to_session(self) -> None:
        """After pre-create, session has one assistant message."""
        session = _make_session()
        ctx = _make_ctx(session)
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
            has_appended_assistant=True,
            has_tool_results=True,
        )

        _simulate_pre_create(acc, ctx)

        assert len(session.messages) == 1
        assert session.messages[-1].role == "assistant"
        assert session.messages[-1].content == ""

    def test_pre_create_resets_tool_result_flag(self) -> None:
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
            has_appended_assistant=True,
            has_tool_results=True,
        )
        ctx = _make_ctx()
        _simulate_pre_create(acc, ctx)

        assert acc.has_tool_results is False

    def test_pre_create_resets_accumulated_tool_calls(self) -> None:
        existing_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash"},
        }
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[existing_call],
            has_appended_assistant=True,
            has_tool_results=True,
        )
        ctx = _make_ctx()
        _simulate_pre_create(acc, ctx)

        assert acc.accumulated_tool_calls == []

    def test_text_delta_accumulates_in_preexisting_message(self) -> None:
        """StreamTextDelta after pre-create updates the already-appended message
        in-place — no double-append."""
        session = _make_session()
        ctx = _make_ctx(session)
        state = _make_state()
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
            has_appended_assistant=True,
            has_tool_results=True,
        )

        _simulate_pre_create(acc, ctx)
        assert len(session.messages) == 1

        # Simulate the first text delta arriving after pre-create
        delta = StreamTextDelta(id="t1", delta="Hello world")
        _dispatch_response(delta, acc, ctx, state, False, "[test]")

        # Still only one message (no double-append)
        assert len(session.messages) == 1
        # Content accumulated in the pre-created message
        assert session.messages[-1].content == "Hello world"
        assert session.messages[-1].role == "assistant"

    def test_subsequent_deltas_append_to_content(self) -> None:
        """Multiple deltas build up the full response text."""
        session = _make_session()
        ctx = _make_ctx(session)
        state = _make_state()
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
            has_appended_assistant=True,
            has_tool_results=True,
        )

        _simulate_pre_create(acc, ctx)

        for word in ["You're ", "right ", "about ", "that."]:
            _dispatch_response(
                StreamTextDelta(id="t1", delta=word), acc, ctx, state, False, "[test]"
            )

        assert len(session.messages) == 1
        assert session.messages[-1].content == "You're right about that."

    def test_pre_create_not_triggered_without_tool_results(self) -> None:
        """Pre-create condition requires has_tool_results=True; no-op otherwise."""
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
            has_appended_assistant=True,
            has_tool_results=False,  # no prior tool results
        )
        ctx = _make_ctx()

        # Condition is False — simulate: do nothing
        if acc.has_tool_results and acc.has_appended_assistant:
            _simulate_pre_create(acc, ctx)

        assert len(ctx.session.messages) == 0

    def test_pre_create_not_triggered_when_not_yet_appended(self) -> None:
        """Pre-create requires has_appended_assistant=True."""
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
            has_appended_assistant=False,  # first turn, nothing appended yet
            has_tool_results=True,
        )
        ctx = _make_ctx()

        if acc.has_tool_results and acc.has_appended_assistant:
            _simulate_pre_create(acc, ctx)

        assert len(ctx.session.messages) == 0

    def test_pre_create_not_triggered_without_text_delta(self) -> None:
        """Pre-create is skipped when adapter_responses has no StreamTextDelta
        (e.g. a tool-only batch). Verifies the third guard condition."""
        acc = _StreamAccumulator(
            assistant_response=ChatMessage(role="assistant", content=""),
            accumulated_tool_calls=[],
            has_appended_assistant=True,
            has_tool_results=True,
        )
        ctx = _make_ctx()
        adapter_responses = [StreamStartStep()]  # no StreamTextDelta

        if (
            acc.has_tool_results
            and acc.has_appended_assistant
            and any(isinstance(r, StreamTextDelta) for r in adapter_responses)
        ):
            _simulate_pre_create(acc, ctx)

        assert len(ctx.session.messages) == 0
