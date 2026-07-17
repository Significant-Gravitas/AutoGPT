"""Tests for partial-work preservation when an SDK turn is interrupted.

Covers the regression SECRT-2275 surfaced: when the SDK retry loop rolls
back ``session.messages`` for a failed attempt (correct so a successful
retry doesn't duplicate content) it MUST re-attach the rolled-back work on
final-failure exit. Without that, the user's UI streamed tokens live then
a refresh shows an empty turn — described by users as "the turn is gone".

Tests target the ``_InterruptedAttempt`` dataclass + the orphan-tool flush
directly. Full retry-loop coverage lives in ``retry_scenarios_test.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable

from .service import (
    _RETRYABLE_STREAM_ERROR_CODES,
    _classify_final_failure,
    _FinalFailure,
    _flush_orphan_tool_uses_to_session,
    _HandledErrorInfo,
    _InterruptedAttempt,
)


def _make_session(messages: list[ChatMessage] | None = None) -> ChatSession:
    session = ChatSession.new(user_id="user-1", dry_run=False)
    session.messages = list(messages or [])
    return session


def _tool_output(tool_call_id: str, output) -> StreamToolOutputAvailable:
    return StreamToolOutputAvailable(
        toolCallId=tool_call_id, toolName="t", output=output
    )


def _adapter_with_unresolved(responses: list[StreamToolOutputAvailable]):
    """Stub _RetryState whose adapter flushes the given responses."""
    adapter = MagicMock()
    adapter.has_unresolved_tool_calls = bool(responses)

    def _flush(out: list) -> None:
        out.extend(responses)
        adapter.has_unresolved_tool_calls = False

    adapter.flush_unresolved_tool_calls.side_effect = _flush
    state = MagicMock()
    state.adapter = adapter
    return state


def _builder_stub() -> MagicMock:
    builder = MagicMock()
    builder.restore = MagicMock()
    return builder


class TestInterruptedAttemptCapture:
    def test_keeps_partial_when_no_marker_present(self):
        session = _make_session(
            [
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="part-1"),
            ]
        )
        attempt = _InterruptedAttempt()
        attempt.capture(session, _builder_stub(), object(), pre_attempt_msg_count=1)
        assert [m.content for m in attempt.partial] == ["part-1"]
        assert [m.content for m in session.messages] == ["hi"]

    def test_strips_trailing_error_marker(self):
        # _run_stream_attempt may append a marker (idle timeout, circuit
        # breaker) before raising _HandledStreamError. Carrying it forward
        # would let finalize() replay it and then add its own.
        marker = (
            f"{COPILOT_RETRYABLE_ERROR_PREFIX} The session has been idle "
            "for too long. Please try again."
        )
        session = _make_session(
            [
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="part-1"),
                ChatMessage(role="assistant", content=marker),
            ]
        )
        attempt = _InterruptedAttempt()
        attempt.capture(session, _builder_stub(), object(), pre_attempt_msg_count=1)
        assert [m.content for m in attempt.partial] == ["part-1"]

    def test_strips_consecutive_error_markers(self):
        session = _make_session(
            [
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="part-1"),
                ChatMessage(role="assistant", content=f"{COPILOT_ERROR_PREFIX} a"),
                ChatMessage(
                    role="assistant", content=f"{COPILOT_RETRYABLE_ERROR_PREFIX} b"
                ),
            ]
        )
        attempt = _InterruptedAttempt()
        attempt.capture(session, _builder_stub(), object(), pre_attempt_msg_count=1)
        assert [m.content for m in attempt.partial] == ["part-1"]

    def test_preserves_non_marker_assistant(self):
        session = _make_session(
            [
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="Important note"),
            ]
        )
        attempt = _InterruptedAttempt()
        attempt.capture(session, _builder_stub(), object(), pre_attempt_msg_count=1)
        assert [m.content for m in attempt.partial] == ["Important note"]


class TestInterruptedAttemptFinalize:
    def test_appends_partial_then_marker(self):
        session = _make_session([ChatMessage(role="user", content="hi")])
        attempt = _InterruptedAttempt(
            partial=[
                ChatMessage(role="assistant", content="working"),
                ChatMessage(role="tool", content="result", tool_call_id="t1"),
            ]
        )
        attempt.finalize(session, state=None, display_msg="Boom", retryable=False)
        roles = [m.role for m in session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert session.messages[-1].content.startswith(COPILOT_ERROR_PREFIX)
        # partial consumed so a follow-up finalize() is a no-op for partial.
        assert attempt.partial == []

    def test_only_marker_when_partial_empty(self):
        session = _make_session([ChatMessage(role="user", content="hi")])
        attempt = _InterruptedAttempt()
        attempt.finalize(session, state=None, display_msg="Boom", retryable=True)
        assert len(session.messages) == 2
        assert session.messages[-1].content.startswith(COPILOT_RETRYABLE_ERROR_PREFIX)

    def test_noop_when_session_is_none(self):
        attempt = _InterruptedAttempt(
            partial=[ChatMessage(role="assistant", content="x")]
        )
        events = attempt.finalize(None, state=None, display_msg="Boom", retryable=False)
        assert events == []

    def test_flushes_unresolved_tools_between_partial_and_marker(self):
        session = _make_session([ChatMessage(role="user", content="hi")])
        attempt = _InterruptedAttempt(
            partial=[
                ChatMessage(
                    role="assistant",
                    content="calling",
                    tool_calls=[
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                ),
            ]
        )
        flushed = [_tool_output("t1", "interrupted")]
        state = _adapter_with_unresolved(flushed)
        events = attempt.finalize(
            session, state=state, display_msg="Boom", retryable=False
        )
        roles = [m.role for m in session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert session.messages[2].tool_call_id == "t1"
        assert session.messages[2].content == "interrupted"
        # The same events that were persisted to history are returned to the
        # caller so the caller can yield them to the client — without this
        # the frontend's spinner widgets stay open until refresh because the
        # adapter's has_unresolved_tool_calls flag is already flipped to False.
        assert events == flushed

    def test_clear_drops_both_partial_and_handled_error(self):
        attempt = _InterruptedAttempt(
            partial=[ChatMessage(role="assistant", content="x")],
            handled_error=_HandledErrorInfo(
                error_msg="m", code="c", retryable=True, already_yielded=False
            ),
        )
        attempt.clear()
        assert attempt.partial == []
        assert attempt.handled_error is None


class TestFlushOrphanToolUses:
    def test_appends_synthetic_tool_results_for_unresolved(self):
        session = _make_session()
        flushed = [_tool_output("t1", "r1"), _tool_output("t2", {"ok": False})]
        state = _adapter_with_unresolved(flushed)
        events = _flush_orphan_tool_uses_to_session(session, state)
        assert [m.tool_call_id for m in session.messages] == ["t1", "t2"]
        # Dict outputs are JSON-encoded so structure survives the str-only
        # ChatMessage content field for the next-turn LLM read.
        assert session.messages[1].content == '{"ok": false}'
        assert events == flushed

    def test_noop_when_state_is_none(self):
        session = _make_session()
        events = _flush_orphan_tool_uses_to_session(session, None)
        assert session.messages == []
        assert events == []

    def test_noop_when_no_unresolved(self):
        adapter = MagicMock()
        adapter.has_unresolved_tool_calls = False
        state = MagicMock()
        state.adapter = adapter
        events = _flush_orphan_tool_uses_to_session(_make_session(), state)
        adapter.flush_unresolved_tool_calls.assert_not_called()
        assert events == []


class TestClassifyFinalFailure:
    """Ensures the history marker (via finalize) and the SSE StreamError yield
    share one source of truth for display message + stream code — any drift
    would let the chat bubble and the SSE event show different copy for the
    same failure."""

    def test_handled_error_wins(self):
        interrupted = _InterruptedAttempt(
            handled_error=_HandledErrorInfo(
                error_msg="circuit tripped",
                code="circuit_breaker",
                retryable=False,
                already_yielded=True,
            )
        )
        result = _classify_final_failure(
            interrupted,
            attempts_exhausted=False,
            transient_exhausted=False,
            stream_err=RuntimeError("ignored"),
        )
        assert result == _FinalFailure(
            display_msg="circuit tripped",
            code="circuit_breaker",
            retryable=False,
        )

    def test_attempts_exhausted(self):
        result = _classify_final_failure(
            _InterruptedAttempt(),
            attempts_exhausted=True,
            transient_exhausted=False,
            stream_err=RuntimeError("x"),
        )
        assert result is not None
        assert result.code == "all_attempts_exhausted"
        assert result.retryable is False

    def test_transient_exhausted(self):
        result = _classify_final_failure(
            _InterruptedAttempt(),
            attempts_exhausted=False,
            transient_exhausted=True,
            stream_err=RuntimeError("x"),
        )
        assert result is not None
        assert result.code == "transient_api_error"
        assert result.retryable is True

    def test_stream_err_fallback(self):
        result = _classify_final_failure(
            _InterruptedAttempt(),
            attempts_exhausted=False,
            transient_exhausted=False,
            stream_err=RuntimeError("some sdk error"),
        )
        assert result is not None
        assert result.code == "sdk_stream_error"
        assert result.retryable is False

    def test_returns_none_when_no_failure_recorded(self):
        assert (
            _classify_final_failure(
                _InterruptedAttempt(),
                attempts_exhausted=False,
                transient_exhausted=False,
                stream_err=None,
            )
            is None
        )


class TestRetryRollbackContract:
    """End-to-end contract: capture on a rolled-back attempt + finalize yields
    the exact content the user saw streaming live, plus the error marker."""

    def test_capture_then_finalize_matches_streamed_sequence(self):
        session = _make_session([ChatMessage(role="user", content="hi")])
        pre = len(session.messages)
        # Simulate incremental SDK appends during the attempt.
        session.messages.extend(
            [
                ChatMessage(role="assistant", content="part-1"),
                ChatMessage(role="assistant", content="part-2"),
            ]
        )
        attempt = _InterruptedAttempt()
        attempt.capture(session, _builder_stub(), object(), pre)
        # Final-failure path — no retry, no success clear().
        attempt.finalize(session, state=None, display_msg="Boom", retryable=False)
        assert [m.content for m in session.messages] == [
            "hi",
            "part-1",
            "part-2",
            f"{COPILOT_ERROR_PREFIX} Boom",
        ]


class TestRetryableStreamErrorCodes:
    """SECRT-2252: ``_dispatch_response`` consults this set to decide whether
    the StreamError flowing through it should append a retryable marker (UI
    shows a retry button) or a terminal one (UI shows ErrorCard only)."""

    def test_transient_api_error_is_retryable(self):
        assert "transient_api_error" in _RETRYABLE_STREAM_ERROR_CODES

    def test_empty_completion_is_retryable(self):
        # The adapter emits this for ghost-finished SDK turns. The user
        # message ("The model returned an empty response.") only makes sense
        # if the UI offers a retry — otherwise the user sees a dead error.
        assert "empty_completion" in _RETRYABLE_STREAM_ERROR_CODES

    def test_unknown_codes_are_not_retryable(self):
        assert "sdk_error" not in _RETRYABLE_STREAM_ERROR_CODES
        assert "all_attempts_exhausted" not in _RETRYABLE_STREAM_ERROR_CODES
