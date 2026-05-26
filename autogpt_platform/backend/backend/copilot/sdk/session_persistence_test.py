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

from backend.copilot.constants import (
    STOPPED_BY_USER_MARKER,
    STREAM_ERROR_MARKER,
    STREAM_INCOMPLETE_MARKER,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamStartStep, StreamTextDelta
from backend.copilot.sdk.service import _dispatch_response, _StreamAccumulator
from backend.copilot.session_cleanup import prune_orphan_tool_calls

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


class TestPruneOrphanToolCalls:
    """A Stop mid-tool-call leaves the session ending on an assistant row whose
    ``tool_calls`` have no matching ``role="tool"`` row.  Unless pruned before
    the next turn, the ``--resume`` transcript would hand Claude CLI a
    ``tool_use`` without a paired ``tool_result`` and the SDK would fail.
    """

    @staticmethod
    def _tool_call(call_id: str, name: str = "bash_exec") -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }

    def test_stop_mid_tool_leaves_orphan_assistant(self) -> None:
        """Stop between StreamToolInputAvailable and StreamToolOutputAvailable:
        the assistant row has ``tool_calls`` but no matching tool row."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="do something"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[self._tool_call("tc_abc")],
            ),
        ]

        removed = prune_orphan_tool_calls(messages)

        assert removed == 1
        assert len(messages) == 1
        assert messages[-1].role == "user"

    def test_stop_strips_stopped_by_user_marker_and_orphan(self) -> None:
        """The service also appends a ``STOPPED_BY_USER_MARKER`` after a
        user stop when the stream loop exits cleanly; both tail rows must go."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="do something"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[self._tool_call("tc_abc")],
            ),
            ChatMessage(role="assistant", content=STOPPED_BY_USER_MARKER),
        ]

        removed = prune_orphan_tool_calls(messages)

        assert removed == 2
        assert len(messages) == 1
        assert messages[-1].role == "user"

    def test_stop_strips_stream_incomplete_marker_and_orphan(self) -> None:
        """When the SDK CLI ends without a ResultMessage (per-query budget
        exhausted, max_turns hit, OOM, crash) the service appends a
        ``STREAM_INCOMPLETE_MARKER`` notice; the next turn's prune must drop
        both the orphan assistant tool_use and the trailing notice so the
        ``--resume`` transcript stays clean."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="do something"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[self._tool_call("tc_abc")],
            ),
            ChatMessage(role="assistant", content=STREAM_INCOMPLETE_MARKER),
        ]

        removed = prune_orphan_tool_calls(messages)

        assert removed == 2
        assert len(messages) == 1
        assert messages[-1].role == "user"

    def test_stop_strips_stream_error_marker_and_orphan(self) -> None:
        """SECRT-2333: ``STREAM_ERROR_MARKER`` is the post-stream marker
        appended when ``ended_with_stream_error=True``.  Like the other
        synthetic notices, it must be stripped on the next turn so it
        doesn't leak into the ``--resume`` transcript and confuse the
        model."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="do something"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[self._tool_call("tc_abc")],
            ),
            ChatMessage(role="assistant", content=STREAM_ERROR_MARKER),
        ]

        removed = prune_orphan_tool_calls(messages)

        assert removed == 2
        assert len(messages) == 1
        assert messages[-1].role == "user"

    def test_stream_error_marker_alone_is_stripped(self) -> None:
        """A trailing STREAM_ERROR_MARKER without an orphan tool_use is
        still stripped — the marker is a one-shot user-facing notice, not
        history the next turn should resume from."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="do something"),
            ChatMessage(role="assistant", content=STREAM_ERROR_MARKER),
        ]

        removed = prune_orphan_tool_calls(messages)

        assert removed == 1
        assert len(messages) == 1
        assert messages[-1].role == "user"

    def test_completed_tool_call_is_preserved(self) -> None:
        """An assistant row whose tool_calls are all resolved is a healthy
        trailing state and must not be popped."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="do something"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[self._tool_call("tc_abc")],
            ),
            ChatMessage(
                role="tool",
                content="ok",
                tool_call_id="tc_abc",
            ),
        ]

        removed = prune_orphan_tool_calls(messages)

        assert removed == 0
        assert len(messages) == 3

    def test_partial_resolution_still_pops(self) -> None:
        """If an assistant emits multiple tool_calls and only some are
        resolved, the assistant row is still unsafe for ``--resume``."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="do something"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    self._tool_call("tc_1"),
                    self._tool_call("tc_2"),
                ],
            ),
            ChatMessage(
                role="tool",
                content="ok",
                tool_call_id="tc_1",
            ),
        ]

        removed = prune_orphan_tool_calls(messages)

        # Both the orphan assistant and its partial tool row are dropped.
        assert removed == 2
        assert len(messages) == 1
        assert messages[-1].role == "user"

    def test_plain_assistant_text_preserved(self) -> None:
        """A regular text-only assistant tail is healthy and must be kept."""
        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]

        removed = prune_orphan_tool_calls(messages)

        assert removed == 0
        assert len(messages) == 2

    def test_empty_session_is_noop(self) -> None:
        messages: list[ChatMessage] = []
        assert prune_orphan_tool_calls(messages) == 0


class TestPruneOrphanToolCallsLogging:
    """``prune_orphan_tool_calls`` emits an INFO log when the caller passes
    ``log_prefix`` and something was actually popped.  Shared by the SDK
    and baseline turn-start cleanup so both paths log in the same shape."""

    def _tool_call(self, call_id: str) -> dict:
        return {"id": call_id, "type": "function", "function": {"name": "bash"}}

    def test_logs_when_something_was_pruned(self, caplog) -> None:
        import backend.copilot.session_cleanup as sc

        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant", content="", tool_calls=[self._tool_call("tc_1")]
            ),
        ]

        sc.logger.propagate = True
        caplog.set_level("INFO", logger=sc.logger.name)
        removed = prune_orphan_tool_calls(messages, log_prefix="[TEST] [abc123]")

        assert removed == 1
        assert any(
            "[TEST] [abc123]" in r.message and "Dropped 1" in r.message
            for r in caplog.records
        ), caplog.text

    def test_no_log_when_nothing_to_prune(self, caplog) -> None:
        import backend.copilot.session_cleanup as sc

        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]

        sc.logger.propagate = True
        caplog.set_level("INFO", logger=sc.logger.name)
        removed = prune_orphan_tool_calls(messages, log_prefix="[TEST] [xyz]")

        assert removed == 0
        assert not any("[TEST] [xyz]" in r.message for r in caplog.records), caplog.text

    def test_no_log_when_log_prefix_is_none(self, caplog) -> None:
        """Without ``log_prefix``, ``prune_orphan_tool_calls`` is silent."""
        import backend.copilot.session_cleanup as sc

        messages: list[ChatMessage] = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant", content="", tool_calls=[self._tool_call("tc_1")]
            ),
        ]

        sc.logger.propagate = True
        caplog.set_level("INFO", logger=sc.logger.name)
        removed = prune_orphan_tool_calls(messages)

        assert removed == 1
        assert caplog.text == ""
