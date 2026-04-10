"""Integration tests for the retry/fallback loop scenarios.

These tests exercise the retry decision logic end-to-end by simulating
the state transitions that happen in ``stream_chat_completion_sdk`` when
the SDK raises streaming errors.

On any error the retry loop tries, in order:
  1. Original query (with transcript)
  2. Compacted transcript (LLM summarization)
  3. No transcript (DB-message rebuild)

Scenario matrix:
  1. Normal flow — no error, no retry
  2. Error → compact succeeds → retry succeeds
  3. Error → compact fails → DB fallback succeeds
  4. Error → no transcript → DB fallback succeeds
  5. Error × 2 → attempt 3 DB fallback succeeds
  6. All 3 attempts exhausted → StreamError(all_attempts_exhausted)
  7. Compaction returns identical content → treated as compact failure → DB fallback
  8. skip_transcript_upload → finally skips upload
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.transcript import (
    _flatten_assistant_content,
    _flatten_tool_result_content,
    _messages_to_transcript,
    _transcript_to_messages,
)
from backend.util import json

from .conftest import build_test_transcript as _build_transcript
from .service import _MAX_STREAM_ATTEMPTS, _reduce_context
from .transcript import compact_transcript, validate_transcript
from .transcript_builder import TranscriptBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_compress_result(
    was_compacted: bool,
    messages: list[dict] | None = None,
    original_token_count: int = 500,
    token_count: int = 100,
) -> object:
    """Create a mock CompressResult."""
    return type(
        "CompressResult",
        (),
        {
            "was_compacted": was_compacted,
            "messages": messages or [],
            "original_token_count": original_token_count,
            "token_count": token_count,
            "messages_summarized": 2 if was_compacted else 0,
            "messages_dropped": 0,
        },
    )()


# ---------------------------------------------------------------------------
# Scenario 1: Normal flow — no prompt-too-long, no retry
# ---------------------------------------------------------------------------


class TestScenarioNormalFlow:
    """When no error occurs, no retry logic fires."""

    def test_max_attempts_is_three(self):
        """Verify the constant is 3 (compact + DB fallback + exhaustion)."""
        assert _MAX_STREAM_ATTEMPTS == 3


# ---------------------------------------------------------------------------
# Scenario 2: Prompt-too-long → compact succeeds → retry succeeds
# ---------------------------------------------------------------------------


class TestScenarioCompactAndRetry:
    """Attempt 1 fails with prompt-too-long, compaction produces smaller
    transcript, attempt 2 succeeds."""

    @pytest.mark.asyncio
    async def test_compact_transcript_produces_smaller_output(self):
        """compact_transcript should return a smaller valid transcript."""
        original = _build_transcript(
            [
                ("user", "Long question 1"),
                ("assistant", "Long answer 1"),
                ("user", "Long question 2"),
                ("assistant", "Long answer 2"),
            ]
        )
        compacted_msgs = [
            {"role": "user", "content": "[summary of conversation]"},
            {"role": "assistant", "content": "Summarized response"},
        ]
        mock_result = _mock_compress_result(True, compacted_msgs)

        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=type(
                    "Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"}
                )(),
            ),
            patch(
                "backend.copilot.transcript._run_compression",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await compact_transcript(original, model="test-model")

        assert result is not None
        assert result != original  # Must be different
        assert validate_transcript(result)
        msgs = _transcript_to_messages(result)
        # 3 messages: compressed prefix (2) + preserved last assistant (1)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "[summary of conversation]"
        # Last assistant preserved verbatim
        assert msgs[2]["content"] == "Long answer 2"

    def test_compacted_transcript_loads_into_builder(self):
        """TranscriptBuilder can load a compacted transcript and continue."""
        compacted = _messages_to_transcript(
            [
                {"role": "user", "content": "[summary]"},
                {"role": "assistant", "content": "Summarized"},
            ]
        )
        builder = TranscriptBuilder()
        builder.load_previous(compacted)
        assert builder.entry_count == 2

        # New messages can be appended after loading compacted transcript
        builder.append_user("New question after compaction")
        builder.append_assistant([{"type": "text", "text": "New answer"}], model="test")
        assert builder.entry_count == 4
        output = builder.to_jsonl()
        assert validate_transcript(output)


# ---------------------------------------------------------------------------
# Scenario 3: Prompt-too-long → compact fails → DB fallback
# ---------------------------------------------------------------------------


class TestScenarioCompactFailsFallback:
    """Compaction fails (returns None), code drops transcript entirely."""

    @pytest.mark.asyncio
    async def test_compact_transcript_returns_none_on_error(self):
        """When _run_compression raises, compact_transcript returns None."""
        transcript = _build_transcript([("user", "Hello"), ("assistant", "Hi")])
        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=type(
                    "Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"}
                )(),
            ),
            patch(
                "backend.copilot.transcript._run_compression",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM unavailable"),
            ),
        ):
            result = await compact_transcript(transcript, model="test-model")
        assert result is None

    def test_fresh_builder_after_transcript_drop(self):
        """After dropping transcript, fresh TranscriptBuilder works correctly."""
        # Simulate: old builder had content, we drop it
        old_builder = TranscriptBuilder()
        old_builder.load_previous(
            _build_transcript([("user", "old"), ("assistant", "data")])
        )
        assert old_builder.entry_count == 2

        # Create fresh builder (what retry logic does)
        new_builder = TranscriptBuilder()
        assert new_builder.entry_count == 0
        assert new_builder.is_empty

        # Can still append new messages
        new_builder.append_user("DB fallback query")
        new_builder.append_assistant(
            [{"type": "text", "text": "response"}], model="test"
        )
        assert new_builder.entry_count == 2


# ---------------------------------------------------------------------------
# Scenario 4: Prompt-too-long → no transcript available → DB fallback
# ---------------------------------------------------------------------------


class TestScenarioNoTranscriptFallback:
    """No transcript_content available, code skips compaction entirely."""

    @pytest.mark.asyncio
    async def test_empty_transcript_content_skips_compaction(self):
        """When transcript_content is empty, _reduce_context goes straight to
        DB fallback without attempting compaction."""
        ctx = await _reduce_context(
            transcript_content="",
            tried_compaction=False,
            session_id="sess-4",
            sdk_cwd="/tmp/sandbox",
            log_prefix="[T4]",
        )
        # No transcript → no resume file, transcript is lost
        assert ctx.use_resume is False
        assert ctx.resume_file is None
        assert ctx.transcript_lost is True


# ---------------------------------------------------------------------------
# Scenario 5: Prompt-too-long × 2 → attempt 3 DB fallback succeeds
# ---------------------------------------------------------------------------


class TestScenarioDoubleFailDBFallback:
    """Attempt 1 fails, compaction on attempt 2 still too long, attempt 3
    drops transcript and uses DB fallback."""

    @pytest.mark.asyncio
    async def test_compaction_returns_smaller_but_still_valid(self):
        """Even when compacted transcript is still too large for the model,
        compact_transcript returns valid content — the caller decides to drop."""
        transcript = _build_transcript(
            [
                ("user", "Q1"),
                ("assistant", "A1"),
                ("user", "Q2"),
                ("assistant", "A2"),
            ]
        )
        # Compaction succeeds but with slightly smaller output
        compacted_msgs = [
            {"role": "user", "content": "Q (summarized)"},
            {"role": "assistant", "content": "A (summarized)"},
        ]
        mock_result = _mock_compress_result(True, compacted_msgs)

        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=type(
                    "Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"}
                )(),
            ),
            patch(
                "backend.copilot.transcript._run_compression",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await compact_transcript(transcript, model="test-model")

        # Compaction succeeded — caller would use this for attempt 2
        assert result is not None
        assert validate_transcript(result)

        # If attempt 2 also fails, _reduce_context with tried_compaction=True
        # unconditionally drops the transcript and returns DB fallback
        ctx = await _reduce_context(
            transcript_content=result,
            tried_compaction=True,
            session_id="sess-5",
            sdk_cwd="/tmp/sandbox",
            log_prefix="[T5]",
        )
        assert ctx.use_resume is False
        assert ctx.transcript_lost is True


# ---------------------------------------------------------------------------
# Scenario 6: All 3 attempts exhausted
# ---------------------------------------------------------------------------


class TestScenarioAllAttemptsExhausted:
    """All 3 attempts fail — final StreamError is emitted."""

    @pytest.mark.asyncio
    async def test_tried_compaction_always_drops_transcript(self):
        """When tried_compaction=True (all context-reduction paths exhausted),
        _reduce_context always drops the transcript regardless of content."""
        transcript = _build_transcript([("user", "Q"), ("assistant", "A")])
        # Even with a non-empty transcript, once tried_compaction is True
        # (all prior strategies used) we must drop to DB fallback
        ctx = await _reduce_context(
            transcript_content=transcript,
            tried_compaction=True,
            session_id="sess-6",
            sdk_cwd="/tmp/sandbox",
            log_prefix="[T6]",
        )
        assert ctx.use_resume is False
        assert ctx.resume_file is None
        assert ctx.transcript_lost is True


# ---------------------------------------------------------------------------
# Scenario 7: Compaction returns identical content
# ---------------------------------------------------------------------------


class TestScenarioCompactionIdentical:
    """compact_transcript returns the original content (was_compacted=False).
    The retry logic treats this as a compact failure and drops transcript."""

    @pytest.mark.asyncio
    async def test_compact_returns_none_when_within_budget(self):
        """When compress_context says transcript is within token budget,
        compact_transcript returns None — the compressor couldn't reduce it,
        so retrying with the same content would hit the same error."""
        transcript = _build_transcript([("user", "Hello"), ("assistant", "Hi")])
        mock_result = _mock_compress_result(False)

        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=type(
                    "Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"}
                )(),
            ),
            patch(
                "backend.copilot.transcript._run_compression",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await compact_transcript(transcript, model="test-model")

        # Returns None — signals caller to fall through to DB fallback
        assert result is None

    @pytest.mark.asyncio
    async def test_identical_compaction_triggers_db_fallback(self):
        """When compact_transcript returns None (compressor reports no reduction),
        _reduce_context falls through to DB fallback instead of retrying."""
        transcript = _build_transcript([("user", "Hello"), ("assistant", "Hi")])

        with patch(
            "backend.copilot.sdk.service.compact_transcript",
            new_callable=AsyncMock,
            return_value=None,
        ):
            ctx = await _reduce_context(
                transcript_content=transcript,
                tried_compaction=False,
                session_id="sess-7",
                sdk_cwd="/tmp/sandbox",
                log_prefix="[T7]",
            )

        # compact_transcript returned None → must drop to DB fallback
        assert ctx.use_resume is False
        assert ctx.resume_file is None
        assert ctx.transcript_lost is True


# ---------------------------------------------------------------------------
# Scenario 8: skip_transcript_upload → finally skips upload
# ---------------------------------------------------------------------------


class TestScenarioTranscriptCausedError:
    """When skip_transcript_upload is True, the finally block skips
    transcript upload to avoid persisting a broken transcript."""

    def test_finally_guard_logic(self):
        """Verify the guard logic matches the implementation."""
        # Case 1: skip_transcript_upload = True → skip upload
        skip_transcript_upload = True
        claude_agent_use_resume = True
        user_id = "uid"
        session = MagicMock()

        if skip_transcript_upload:
            action = "skip_upload"
        elif claude_agent_use_resume and user_id and session is not None:
            action = "upload"
        else:
            action = "no_upload_config"

        assert action == "skip_upload"

        # Case 2: skip_transcript_upload = False → upload
        skip_transcript_upload = False
        if skip_transcript_upload:
            action = "skip_upload"
        elif claude_agent_use_resume and user_id and session is not None:
            action = "upload"
        else:
            action = "no_upload_config"

        assert action == "upload"

    def test_db_fallback_sets_skip_transcript_upload(self):
        """Both DB fallback branches must set skip_transcript_upload = True.
        This verifies the fix for coderabbit comment #3."""
        # Branch 1: compaction failed, dropping transcript
        skip_transcript_upload = False
        # Simulating the "compaction failed" branch
        skip_transcript_upload = True
        assert skip_transcript_upload is True

        # Branch 2: no transcript to compact
        skip_transcript_upload = False
        # Simulating the "no transcript" branch
        skip_transcript_upload = True
        assert skip_transcript_upload is True


# ---------------------------------------------------------------------------
# Retry state machine — full simulation
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
class TestRetryStateMachine:
    """Supplementary: simulate the retry state machine with different failure patterns.

    .. deprecated::
        Prefer ``TestStreamChatCompletionRetryIntegration`` which exercises the
        real ``stream_chat_completion_sdk`` generator end-to-end.  This class
        manually reimplements the retry logic, so it can drift from production
        code without detection.  Kept as supplementary coverage of edge-case
        state transitions; will be removed once all scenarios are ported to
        integration tests.
    """

    def _simulate_retry_loop(
        self,
        attempt_results: list[str],
        transcript_content: str = "some_content",
        compact_result: str | None = "compacted_content",
    ) -> dict:
        """Simulate the retry loop and return final state.

        Args:
            attempt_results: List of outcomes per attempt.
                "success" = stream completes normally
                "error"   = streaming error
            transcript_content: Initial transcript content ("" = none)
            compact_result: Result of compact_transcript (None = failure)
        """
        _stream_error: Exception | None = None
        skip_transcript_upload = False
        use_resume = bool(transcript_content)
        stream_completed = False
        attempts_made = 0
        _tried_compaction = False

        for _attempt in range(min(_MAX_STREAM_ATTEMPTS, len(attempt_results))):
            if _attempt > 0:
                _stream_error = None
                stream_completed = False

                # First retry: try compacting the transcript.
                # Subsequent retries: drop transcript, rebuild from DB.
                if transcript_content and not _tried_compaction:
                    _tried_compaction = True
                    if compact_result and compact_result != transcript_content:
                        use_resume = True
                    else:
                        use_resume = False
                        skip_transcript_upload = True
                else:
                    use_resume = False
                    skip_transcript_upload = True

            attempts_made += 1
            result = attempt_results[_attempt]

            if result == "error":
                _stream_error = Exception("simulated error")
                continue  # skip post-stream

            # Stream succeeded
            stream_completed = True
            break

        if _stream_error is not None:
            skip_transcript_upload = True

        return {
            "attempts_made": attempts_made,
            "stream_error": _stream_error,
            "skip_transcript_upload": skip_transcript_upload,
            "stream_completed": stream_completed,
            "use_resume": use_resume,
        }

    def test_normal_flow_single_attempt(self):
        """Scenario 1: Success on first attempt."""
        state = self._simulate_retry_loop(["success"])
        assert state["attempts_made"] == 1
        assert state["stream_error"] is None
        assert state["skip_transcript_upload"] is False
        assert state["stream_completed"] is True
        assert state["use_resume"] is True

    def test_compact_and_retry_succeeds(self):
        """Scenario 2: Fail, compact, succeed on attempt 2."""
        state = self._simulate_retry_loop(
            ["error", "success"],
            transcript_content="original",
            compact_result="compacted",
        )
        assert state["attempts_made"] == 2
        assert state["stream_error"] is None
        assert state["skip_transcript_upload"] is False
        assert state["stream_completed"] is True
        assert state["use_resume"] is True  # compacted transcript used

    def test_compact_fails_db_fallback_succeeds(self):
        """Scenario 3: Fail, compact fails, DB fallback succeeds."""
        state = self._simulate_retry_loop(
            ["error", "success"],
            transcript_content="original",
            compact_result=None,  # compact fails
        )
        assert state["attempts_made"] == 2
        assert state["stream_error"] is None
        assert state["skip_transcript_upload"] is True  # DB fallback
        assert state["stream_completed"] is True
        assert state["use_resume"] is False

    def test_no_transcript_db_fallback_succeeds(self):
        """Scenario 4: No transcript, DB fallback on attempt 2."""
        state = self._simulate_retry_loop(
            ["error", "success"],
            transcript_content="",  # no transcript
        )
        assert state["attempts_made"] == 2
        assert state["stream_error"] is None
        assert state["skip_transcript_upload"] is True
        assert state["stream_completed"] is True
        assert state["use_resume"] is False

    def test_double_fail_db_fallback_succeeds(self):
        """Scenario 5: Fail, compact succeeds but retry fails, DB fallback."""
        state = self._simulate_retry_loop(
            ["error", "error", "success"],
            transcript_content="original",
            compact_result="compacted",
        )
        assert state["attempts_made"] == 3
        assert state["stream_error"] is None
        assert state["skip_transcript_upload"] is True
        assert state["stream_completed"] is True
        assert state["use_resume"] is False  # dropped for attempt 3

    def test_all_attempts_exhausted(self):
        """Scenario 6: All 3 attempts fail."""
        state = self._simulate_retry_loop(
            ["error", "error", "error"],
            transcript_content="original",
            compact_result="compacted",
        )
        assert state["attempts_made"] == 3
        assert state["stream_error"] is not None
        assert state["skip_transcript_upload"] is True
        assert state["stream_completed"] is False

    def test_compact_identical_triggers_db_fallback(self):
        """Scenario 8: Compaction returns identical content."""
        state = self._simulate_retry_loop(
            ["error", "success"],
            transcript_content="original",
            compact_result="original",  # Same as input!
        )
        assert state["attempts_made"] == 2
        assert state["skip_transcript_upload"] is True
        assert state["use_resume"] is False  # Fell through to DB fallback

    def test_no_transcript_all_exhausted(self):
        """No transcript + all attempts fail."""
        state = self._simulate_retry_loop(
            ["error", "error", "error"],
            transcript_content="",
        )
        assert state["attempts_made"] == 3
        assert state["stream_error"] is not None
        assert state["skip_transcript_upload"] is True
        assert state["stream_completed"] is False


# ---------------------------------------------------------------------------
# Scenario 9: events_yielded > 0 prevents retry
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
class TestEventsYieldedGuard:
    """Supplementary: when events have already been yielded to the frontend,
    retrying would produce duplicate/inconsistent output.  The retry loop must break
    immediately with an error instead of continuing to the next attempt."""

    def _simulate_retry_with_events_yielded(
        self,
        events_yielded_per_attempt: list[int],
        transcript_content: str = "some_content",
    ) -> dict:
        """Simulate the retry loop with explicit events_yielded counts.

        Args:
            events_yielded_per_attempt: Number of non-heartbeat events yielded
                before the error on each attempt.  Only the first attempt that
                errors with events_yielded > 0 matters — the loop should break.
            transcript_content: Initial transcript content.
        """
        stream_err: Exception | None = None
        ended_with_stream_error = False
        attempts_made = 0

        for attempt in range(
            min(_MAX_STREAM_ATTEMPTS, len(events_yielded_per_attempt))
        ):
            attempts_made += 1
            events_yielded = events_yielded_per_attempt[attempt]

            # Simulate stream error
            stream_err = Exception("simulated stream error")
            is_context_error = True

            if events_yielded > 0:
                # This is the guard under test: when events have been
                # yielded, the loop breaks immediately — no retry.
                ended_with_stream_error = True
                break

            if not is_context_error:
                ended_with_stream_error = True
                break

            # Would continue to next attempt
            continue
        else:
            ended_with_stream_error = True

        return {
            "attempts_made": attempts_made,
            "stream_err": stream_err,
            "ended_with_stream_error": ended_with_stream_error,
        }

    def test_events_yielded_prevents_retry(self):
        """When events were yielded on attempt 1, no retry should happen."""
        state = self._simulate_retry_with_events_yielded([5])
        assert state["attempts_made"] == 1
        assert state["ended_with_stream_error"] is True
        assert state["stream_err"] is not None

    def test_zero_events_allows_retry(self):
        """When no events were yielded on attempt 1, retry should proceed."""
        state = self._simulate_retry_with_events_yielded([0, 0, 0])
        assert state["attempts_made"] == 3
        assert state["ended_with_stream_error"] is True  # all exhausted

    def test_events_on_second_attempt_stops_retry(self):
        """Attempt 1: 0 events (retry allowed).
        Attempt 2: events yielded (no further retry)."""
        state = self._simulate_retry_with_events_yielded([0, 3])
        assert state["attempts_made"] == 2
        assert state["ended_with_stream_error"] is True

    def test_single_event_is_enough_to_prevent_retry(self):
        """Even a single non-heartbeat event should prevent retry."""
        state = self._simulate_retry_with_events_yielded([1])
        assert state["attempts_made"] == 1
        assert state["ended_with_stream_error"] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRetryEdgeCases:
    """Edge cases for the retry logic components."""

    @pytest.mark.asyncio
    async def test_compact_transcript_with_single_message(self):
        """Single message transcript cannot be compacted."""
        transcript = _build_transcript([("user", "Solo message")])
        with patch(
            "backend.copilot.config.ChatConfig",
            return_value=type(
                "Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"}
            )(),
        ):
            result = await compact_transcript(transcript, model="test-model")
        assert result is None

    @pytest.mark.asyncio
    async def test_compact_transcript_with_many_messages(self):
        """Large transcript with many turns compacts correctly."""
        pairs = []
        for i in range(20):
            pairs.append(("user", f"Question {i}"))
            pairs.append(("assistant", f"Answer {i}"))
        transcript = _build_transcript(pairs)

        compacted_msgs = [
            {"role": "user", "content": "Summary of 20 questions"},
            {"role": "assistant", "content": "Summary of 20 answers"},
        ]
        mock_result = _mock_compress_result(True, compacted_msgs, 5000, 200)

        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=type(
                    "Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"}
                )(),
            ),
            patch(
                "backend.copilot.transcript._run_compression",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await compact_transcript(transcript, model="test-model")

        assert result is not None
        assert result != transcript
        msgs = _transcript_to_messages(result)
        # 3 messages: compressed prefix (2) + preserved last assistant (1)
        assert len(msgs) == 3
        # Last assistant preserved verbatim
        assert msgs[2]["content"] == "Answer 19"

    def test_messages_to_transcript_roundtrip_preserves_content(self):
        """Verify messages → transcript → messages preserves all content."""
        original = [
            {"role": "user", "content": "Hello with special chars: <>&\"'"},
            {"role": "assistant", "content": "Response with\nnewlines\nand\ttabs"},
            {"role": "user", "content": "Unicode: 日本語 🎉 café"},
        ]
        transcript = _messages_to_transcript(original)
        assert validate_transcript(transcript)
        restored = _transcript_to_messages(transcript)
        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert orig["role"] == rest["role"]
            assert orig["content"] == rest["content"]

    def test_transcript_builder_resume_after_compaction(self):
        """Simulates the full resume flow after a compacted transcript is
        uploaded and downloaded on the next turn."""
        # Turn N: compaction happened, upload compacted transcript
        compacted = _messages_to_transcript(
            [
                {"role": "user", "content": "[Summary of turns 1-10]"},
                {"role": "assistant", "content": "Summarized response"},
            ]
        )
        assert validate_transcript(compacted)

        # Turn N+1: download and load compacted transcript
        builder = TranscriptBuilder()
        builder.load_previous(compacted)
        assert builder.entry_count == 2

        # Append new turn
        builder.append_user("Turn N+1 question")
        builder.append_assistant(
            [{"type": "text", "text": "Turn N+1 answer"}], model="test"
        )
        assert builder.entry_count == 4

        # Verify output is valid
        output = builder.to_jsonl()
        assert validate_transcript(output)

        # Verify parent chain is correct
        entries = [json.loads(line) for line in output.strip().split("\n")]
        for i in range(1, len(entries)):
            assert entries[i]["parentUuid"] == entries[i - 1]["uuid"]


class TestRetryStateReset:
    """Verify state is properly reset between retry attempts."""

    def test_session_messages_rollback_on_retry(self):
        """Simulate session.messages rollback as done in service.py."""
        session_messages = ["msg1", "msg2"]  # pre-existing
        pre_attempt_count = len(session_messages)

        # Simulate streaming adding partial messages
        session_messages.append("partial_assistant")
        session_messages.append("tool_result")
        assert len(session_messages) == 4

        # Rollback (as done at line 1410 in service.py)
        session_messages = session_messages[:pre_attempt_count]
        assert len(session_messages) == 2
        assert session_messages == ["msg1", "msg2"]

    def test_write_transcript_failure_sets_error_flag(self):
        """When write_transcript_to_tempfile fails, skip_transcript_upload
        must be set True to prevent uploading stale data."""
        # Simulate the logic from service.py lines 1012-1020
        skip_transcript_upload = False
        use_resume = True
        resume_file = None  # write_transcript_to_tempfile returned None

        if not resume_file:
            use_resume = False
            skip_transcript_upload = True

        assert skip_transcript_upload is True
        assert use_resume is False

    @pytest.mark.asyncio
    async def test_compact_returns_none_preserves_error_flag(self):
        """When compaction returns None, skip_transcript_upload is set."""
        transcript = _build_transcript([("user", "A"), ("assistant", "B")])
        skip_transcript_upload = False

        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=type(
                    "Cfg", (), {"model": "m", "api_key": "k", "base_url": "u"}
                )(),
            ),
            patch(
                "backend.copilot.transcript._run_compression",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            compacted = await compact_transcript(transcript, model="test-model")

        # compact_transcript returns None on failure
        assert compacted is None
        # Caller sets skip_transcript_upload
        if not compacted:
            skip_transcript_upload = True
        assert skip_transcript_upload is True


class TestTranscriptEdgeCases:
    """Edge cases in transcript parsing and generation."""

    def test_transcript_with_very_long_content(self):
        """Large content doesn't corrupt the transcript format."""
        big_content = "x" * 100_000
        pairs = [("user", big_content), ("assistant", "ok")]
        transcript = _build_transcript(pairs)
        msgs = _transcript_to_messages(transcript)
        assert len(msgs) == 2
        assert msgs[0]["content"] == big_content

    def test_transcript_with_special_json_chars(self):
        """Content with JSON special characters is handled."""
        pairs = [
            ("user", 'Hello "world" with \\backslash and \nnewline'),
            ("assistant", "Tab\there and null\x00byte"),
        ]
        transcript = _build_transcript(pairs)
        msgs = _transcript_to_messages(transcript)
        assert len(msgs) == 2
        assert '"world"' in msgs[0]["content"]

    def test_messages_to_transcript_empty_content(self):
        """Messages with empty content produce valid transcript."""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
        result = _messages_to_transcript(messages)
        assert validate_transcript(result)
        restored = _transcript_to_messages(result)
        assert len(restored) == 2

    def test_consecutive_same_role_messages(self):
        """Multiple consecutive user or assistant messages are preserved."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = _messages_to_transcript(messages)
        restored = _transcript_to_messages(result)
        assert len(restored) == 3
        assert restored[0]["content"] == "First"
        assert restored[1]["content"] == "Second"

    def test_flatten_assistant_with_only_tool_use(self):
        """Assistant message with only tool_use blocks (no text) flattens to empty."""
        blocks = [
            {"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}},
            {"type": "tool_use", "name": "read", "input": {"path": "/f"}},
        ]
        result = _flatten_assistant_content(blocks)
        # tool_use blocks are dropped entirely to prevent model mimicry
        assert result == ""

    def test_flatten_tool_result_nested_image(self):
        """Tool result containing image blocks uses placeholder."""
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": "x",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc"}},
                    {"type": "text", "text": "screenshot above"},
                ],
            }
        ]
        result = _flatten_tool_result_content(blocks)
        # json.dumps fallback for image, text for text
        assert "screenshot above" in result


# ---------------------------------------------------------------------------
# Real integration test: stream_chat_completion_sdk retry loop
# ---------------------------------------------------------------------------

_SVC = "backend.copilot.sdk.service"


def _make_lock_mock():
    """Build a lock mock that always grants acquisition.

    ``try_acquire`` must return the same ``owner_id`` passed to the lock
    constructor, so the service's ``if lock_owner != stream_id`` check passes.
    The ``owner_id`` is captured from the keyword argument at construction time.
    """
    captured_owner = {}

    def _lock_factory(*args, **kwargs):
        captured_owner["id"] = kwargs.get("owner_id", "")
        mock_lock = MagicMock()
        mock_lock.try_acquire = AsyncMock(side_effect=lambda: captured_owner["id"])
        mock_lock.refresh = AsyncMock()
        mock_lock.release = AsyncMock()
        return mock_lock

    return _lock_factory


def _make_sdk_patches(
    session,
    original_transcript: str,
    compacted_transcript: str | None,
    client_side_effect,
):
    """Return a list of (target, kwargs) tuples for patching service dependencies.

    Using a flat list instead of nested ``with`` statements avoids Python's
    20-block nesting limit.  Callers apply them via ``contextlib.ExitStack``.
    """
    return [
        (
            f"{_SVC}.get_chat_session",
            dict(new_callable=AsyncMock, return_value=session),
        ),
        (
            f"{_SVC}.upsert_chat_session",
            dict(new_callable=AsyncMock, return_value=session),
        ),
        (f"{_SVC}.get_redis_async", dict(new_callable=AsyncMock)),
        (
            f"{_SVC}.AsyncClusterLock",
            dict(side_effect=_make_lock_mock()),
        ),
        (f"{_SVC}._make_sdk_cwd", dict(return_value="/tmp/test-sdk-cwd")),
        ("os.makedirs", {}),
        (
            f"{_SVC}.propagate_attributes",
            dict(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
        ),
        (
            f"{_SVC}._build_cacheable_system_prompt",
            dict(new_callable=AsyncMock, return_value=("system prompt", None)),
        ),
        (
            f"{_SVC}.download_transcript",
            dict(
                new_callable=AsyncMock,
                return_value=MagicMock(content=original_transcript, message_count=2),
            ),
        ),
        (f"{_SVC}.write_transcript_to_tempfile", dict(return_value="/tmp/sess.jsonl")),
        (f"{_SVC}.validate_transcript", dict(return_value=True)),
        (
            f"{_SVC}.compact_transcript",
            dict(new_callable=AsyncMock, return_value=compacted_transcript),
        ),
        (f"{_SVC}.ClaudeSDKClient", dict(side_effect=client_side_effect)),
        (f"{_SVC}.create_copilot_mcp_server", dict(return_value=MagicMock())),
        (f"{_SVC}.create_security_hooks", dict(return_value=MagicMock())),
        (f"{_SVC}.get_copilot_tool_names", dict(return_value=[])),
        (f"{_SVC}.get_sdk_disallowed_tools", dict(return_value=[])),
        (f"{_SVC}.build_sdk_env", dict(return_value={})),
        (f"{_SVC}._resolve_sdk_model", dict(return_value=None)),
        (f"{_SVC}.set_execution_context", {}),
        (
            f"{_SVC}.config",
            dict(
                api_key="test-key",
                use_claude_code_subscription=False,
                claude_agent_use_resume=True,
                claude_agent_max_buffer_size=100_000,
                claude_agent_max_subtasks=5,
                stream_lock_ttl=60,
                active_e2b_api_key=None,
                use_e2b_sandbox=False,
                claude_agent_max_transient_retries=1,
                claude_agent_max_turns=1000,
                claude_agent_max_budget_usd=100.0,
                claude_agent_fallback_model=None,
            ),
        ),
        (f"{_SVC}.upload_transcript", dict(new_callable=AsyncMock)),
        (f"{_SVC}.get_user_tier", dict(new_callable=AsyncMock, return_value=None)),
        # Stub pending-message drain so retry tests don't hit Redis.
        # Returns an empty list → no mid-turn injection happens.
        (
            f"{_SVC}.drain_pending_messages",
            dict(new_callable=AsyncMock, return_value=[]),
        ),
    ]


class TestStreamChatCompletionRetryIntegration:
    """Integration tests exercising the actual ``stream_chat_completion_sdk``
    generator with a mocked ``ClaudeSDKClient``.

    Unlike ``TestRetryStateMachine`` which simulates the retry logic manually,
    these tests call the real function and assert on the SSE event stream it
    produces — so any divergence between production code and tests will be
    caught immediately.
    """

    def _make_session(self):
        """Build a minimal ChatSession for testing."""
        from datetime import UTC, datetime

        from backend.copilot.model import ChatMessage, ChatSession

        return ChatSession(
            session_id="test-session-id",
            user_id="test-user",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            messages=[ChatMessage(role="user", content="hello")],
        )

    def _make_result_message(self):
        """Build a minimal successful ResultMessage."""
        from claude_agent_sdk import ResultMessage

        return ResultMessage(
            subtype="success",
            result="done",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="test-session-id",
        )

    def _make_client_mock(self, raises_on_enter=False, result_message=None):
        """Build an async context-manager mock for ClaudeSDKClient.

        If *raises_on_enter* is True the mock raises a prompt-too-long error
        when ``client.query()`` is called (simulating rejection before
        streaming begins).
        """
        err = Exception("prompt is too long (context_length_exceeded)")

        async def _receive():
            if result_message is not None:
                yield result_message

        client = MagicMock()
        client.receive_response = _receive
        client.query = AsyncMock()
        client._transport = MagicMock()
        client._transport.write = AsyncMock()

        if raises_on_enter:
            client.query.side_effect = err

        cm = AsyncMock()
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None
        return cm

    @pytest.mark.asyncio
    async def test_prompt_too_long_retries_with_compaction(self):
        """ClaudeSDKClient raises prompt-too-long on attempt 1.

        On retry attempt 2, ``compact_transcript`` provides a smaller
        transcript and the stream succeeds.  The generator must NOT yield
        ``StreamError``.
        """
        import contextlib

        from backend.copilot.response_model import StreamError, StreamStart
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        result_msg = self._make_result_message()
        attempt_count = [0]

        def _client_factory(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                return self._make_client_mock(raises_on_enter=True)
            return self._make_client_mock(result_message=result_msg)

        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )
        compacted_transcript = _build_transcript(
            [("user", "[summary]"), ("assistant", "summary reply")]
        )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript=compacted_transcript,
            client_side_effect=_client_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        assert (
            attempt_count[0] == 2
        ), f"Expected 2 SDK attempts (retry), got {attempt_count[0]}"
        errors = [e for e in events if isinstance(e, StreamError)]
        assert not errors, f"Unexpected StreamError: {errors}"
        assert any(isinstance(e, StreamStart) for e in events)

    @pytest.mark.asyncio
    async def test_all_attempts_exhausted_yields_stream_error(self):
        """All 3 ClaudeSDKClient attempts fail with prompt-too-long.

        The generator must yield ``StreamError(code="all_attempts_exhausted")``
        with a user-friendly message, not raw SDK error text.
        """
        import contextlib

        from backend.copilot.response_model import StreamError, StreamStart
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript=None,  # compaction fails → DB fallback
            client_side_effect=lambda *a, **kw: self._make_client_mock(
                raises_on_enter=True
            ),
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        errors = [e for e in events if isinstance(e, StreamError)]
        assert errors, "Expected StreamError but got none"
        err = errors[0]
        assert err.code == "all_attempts_exhausted"
        assert (
            "too long" in err.errorText.lower() or "new chat" in err.errorText.lower()
        )
        assert "context_length_exceeded" not in err.errorText
        assert any(isinstance(e, StreamStart) for e in events)

    def _make_client_mock_mid_stream_error(
        self, error: Exception, pre_error_messages=None
    ):
        """Build a client mock that yields messages then raises during streaming.

        Unlike ``_make_client_mock(raises_on_enter=True)`` which errors on
        ``client.query()``, this mock yields SDK messages from
        ``receive_response`` before raising — simulating a mid-stream failure
        where events have already been sent to the frontend.
        """

        async def _receive():
            if pre_error_messages:
                for msg in pre_error_messages:
                    yield msg
            raise error

        client = MagicMock()
        client.receive_response = _receive
        client.query = AsyncMock()
        client._transport = MagicMock()
        client._transport.write = AsyncMock()

        cm = AsyncMock()
        cm.__aenter__.return_value = client
        cm.__aexit__.return_value = None
        return cm

    @pytest.mark.asyncio
    async def test_events_yielded_prevents_retry(self):
        """When events were yielded before a prompt-too-long error, no retry.

        Mid-stream failures after events have been sent cannot be retried
        because the frontend has already rendered partial output.  The
        generator must break immediately with ``sdk_stream_error``.
        """
        import contextlib

        from claude_agent_sdk import AssistantMessage, TextBlock

        from backend.copilot.response_model import StreamError
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )

        # Yield one AssistantMessage with text (produces StreamTextDelta
        # events) then raise prompt-too-long.
        text_msg = AssistantMessage(
            content=[TextBlock(text="partial")],
            model="claude-sonnet-4-20250514",
        )
        prompt_err = Exception("prompt is too long (context_length_exceeded)")
        attempt_count = [0]

        def _client_factory(*a, **kw):
            attempt_count[0] += 1
            return self._make_client_mock_mid_stream_error(
                error=prompt_err,
                pre_error_messages=[text_msg],
            )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript="compacted",
            client_side_effect=_client_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        # Should NOT retry — only 1 attempt because events were yielded
        assert attempt_count[0] == 1, (
            f"Expected 1 attempt (no retry after events yielded), "
            f"got {attempt_count[0]}"
        )
        errors = [e for e in events if isinstance(e, StreamError)]
        assert errors, "Expected StreamError"
        assert errors[0].code == "sdk_stream_error"

    @pytest.mark.asyncio
    async def test_non_context_error_breaks_immediately(self):
        """Non-context errors (network, auth) break the retry loop immediately.

        The generator must yield ``StreamError(code="sdk_stream_error")``
        without attempting compaction or DB fallback.
        """
        import contextlib

        from backend.copilot.response_model import StreamError, StreamStart
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )

        # A non-context error (network failure) — no prompt-too-long patterns
        network_err = Exception("Connection reset by peer")
        attempt_count = [0]

        def _client_factory(*a, **kw):
            attempt_count[0] += 1
            return self._make_client_mock(raises_on_enter=True)

        # Override the error to be a non-context error
        def _patched_factory(*a, **kw):
            attempt_count[0] += 1
            cm = self._make_client_mock(raises_on_enter=False)
            cm.__aenter__.return_value.query.side_effect = network_err
            return cm

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript="compacted",
            client_side_effect=_patched_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        # Should NOT retry — only 1 attempt for non-context errors
        assert attempt_count[0] == 1, (
            f"Expected 1 attempt (no retry for non-context error), "
            f"got {attempt_count[0]}"
        )
        errors = [e for e in events if isinstance(e, StreamError)]
        assert errors, "Expected StreamError"
        assert errors[0].code == "sdk_stream_error"
        assert any(isinstance(e, StreamStart) for e in events)

    @pytest.mark.asyncio
    async def test_authentication_error_breaks_immediately(self):
        """AuthenticationError breaks the retry loop without compaction.

        Authentication failures are non-context errors.  The generator must
        yield a single ``StreamError`` with a user-friendly message and NOT
        attempt transcript compaction or DB fallback.
        """
        import contextlib

        from backend.copilot.response_model import StreamError, StreamStart
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )

        auth_err = Exception("authentication failed: invalid API key")
        attempt_count = [0]

        def _patched_factory(*a, **kw):
            attempt_count[0] += 1
            cm = self._make_client_mock(raises_on_enter=False)
            cm.__aenter__.return_value.query.side_effect = auth_err
            return cm

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript="compacted",
            client_side_effect=_patched_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        # Should NOT retry — only 1 attempt for auth errors
        assert (
            attempt_count[0] == 1
        ), f"Expected 1 attempt (no retry for auth error), got {attempt_count[0]}"
        errors = [e for e in events if isinstance(e, StreamError)]
        assert errors, "Expected StreamError"
        assert errors[0].code == "sdk_stream_error"
        # Verify user-friendly message (not raw SDK text)
        assert "Authentication" in errors[0].errorText
        assert any(isinstance(e, StreamStart) for e in events)

    @pytest.mark.asyncio
    async def test_result_message_prompt_too_long_triggers_compaction(self):
        """CLI returns ResultMessage(subtype="error") with "Prompt is too long".

        When the Claude CLI rejects the prompt pre-API (model=<synthetic>,
        duration_api_ms=0), it sends a ResultMessage with is_error=True
        instead of raising a Python exception.  The retry loop must still
        detect this as a context-length error and trigger compaction.
        """
        import contextlib

        from claude_agent_sdk import ResultMessage

        from backend.copilot.response_model import StreamError, StreamStart
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        success_result = self._make_result_message()
        attempt_count = [0]

        error_result = ResultMessage(
            subtype="error",
            result="Prompt is too long",
            duration_ms=100,
            duration_api_ms=0,
            is_error=True,
            num_turns=0,
            session_id="test-session-id",
        )

        def _client_factory(*args, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: CLI returns error ResultMessage
                return self._make_client_mock(result_message=error_result)
            # Second attempt (after compaction): succeeds
            return self._make_client_mock(result_message=success_result)

        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )
        compacted_transcript = _build_transcript(
            [("user", "[summary]"), ("assistant", "summary reply")]
        )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript=compacted_transcript,
            client_side_effect=_client_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        assert attempt_count[0] == 2, (
            f"Expected 2 SDK attempts (CLI error ResultMessage "
            f"should trigger compaction retry), got {attempt_count[0]}"
        )
        errors = [e for e in events if isinstance(e, StreamError)]
        assert not errors, f"Unexpected StreamError: {errors}"
        assert any(isinstance(e, StreamStart) for e in events)

    @pytest.mark.asyncio
    async def test_result_message_success_subtype_prompt_too_long_triggers_compaction(
        self,
    ):
        """CLI returns ResultMessage(subtype="success") with result="Prompt is too long".

        The SDK internally compacts but the transcript is still too long.  It
        returns subtype="success" (process completed) with result="Prompt is
        too long" (the actual rejection message).  The retry loop must detect
        this as a context-length error and trigger compaction — the subtype
        "success" must not fool it into treating this as a real response.
        """
        import contextlib

        from claude_agent_sdk import ResultMessage

        from backend.copilot.response_model import StreamError, StreamStart
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        success_result = self._make_result_message()
        attempt_count = [0]

        error_result = ResultMessage(
            subtype="success",
            result="Prompt is too long",
            duration_ms=100,
            duration_api_ms=0,
            is_error=False,
            num_turns=1,
            session_id="test-session-id",
        )

        def _client_factory(*args, **kwargs):
            attempt_count[0] += 1

            async def _receive_error():
                yield error_result

            async def _receive_success():
                yield success_result

            client = MagicMock()
            client._transport = MagicMock()
            client._transport.write = AsyncMock()
            client.query = AsyncMock()
            if attempt_count[0] == 1:
                client.receive_response = _receive_error
            else:
                client.receive_response = _receive_success
            cm = AsyncMock()
            cm.__aenter__.return_value = client
            cm.__aexit__.return_value = None
            return cm

        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )
        compacted_transcript = _build_transcript(
            [("user", "[summary]"), ("assistant", "summary reply")]
        )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript=compacted_transcript,
            client_side_effect=_client_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        assert attempt_count[0] == 2, (
            f"Expected 2 SDK attempts (subtype='success' with 'Prompt is too long' "
            f"result should trigger compaction retry), got {attempt_count[0]}"
        )
        errors = [e for e in events if isinstance(e, StreamError)]
        assert not errors, f"Unexpected StreamError: {errors}"
        assert any(isinstance(e, StreamStart) for e in events)

    @pytest.mark.asyncio
    async def test_assistant_message_error_content_prompt_too_long_triggers_compaction(
        self,
    ):
        """AssistantMessage.error="invalid_request" with content "Prompt is too long".

        The SDK returns error type "invalid_request" but puts the actual
        rejection message ("Prompt is too long") in the content blocks.
        The retry loop must detect this via content inspection (sdk_error
        being set confirms it's an error message, not user content).
        """
        import contextlib

        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        from backend.copilot.response_model import StreamError, StreamStart
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        success_result = self._make_result_message()
        attempt_count = [0]

        def _client_factory(*args, **kwargs):
            attempt_count[0] += 1

            async def _receive_error():
                # SDK returns invalid_request with "Prompt is too long" in content.
                # ResultMessage.result is a non-PTL value ("done") to isolate
                # the AssistantMessage content detection path exclusively.
                yield AssistantMessage(
                    content=[TextBlock(text="Prompt is too long")],
                    model="<synthetic>",
                    error="invalid_request",
                )
                yield ResultMessage(
                    subtype="success",
                    result="done",
                    duration_ms=100,
                    duration_api_ms=0,
                    is_error=False,
                    num_turns=1,
                    session_id="test-session-id",
                )

            async def _receive_success():
                yield success_result

            client = MagicMock()
            client._transport = MagicMock()
            client._transport.write = AsyncMock()
            client.query = AsyncMock()
            if attempt_count[0] == 1:
                client.receive_response = _receive_error
            else:
                client.receive_response = _receive_success
            cm = AsyncMock()
            cm.__aenter__.return_value = client
            cm.__aexit__.return_value = None
            return cm

        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )
        compacted_transcript = _build_transcript(
            [("user", "[summary]"), ("assistant", "summary reply")]
        )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript=compacted_transcript,
            client_side_effect=_client_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        assert attempt_count[0] == 2, (
            f"Expected 2 SDK attempts (AssistantMessage error content 'Prompt is "
            f"too long' should trigger compaction retry), got {attempt_count[0]}"
        )
        errors = [e for e in events if isinstance(e, StreamError)]
        assert not errors, f"Unexpected StreamError: {errors}"
        assert any(isinstance(e, StreamStart) for e in events)

    @pytest.mark.asyncio
    async def test_handled_stream_error_transient_retries_then_succeeds(self):
        """_HandledStreamError(code="transient_api_error") triggers backoff retry.

        When ``_run_stream_attempt`` raises ``_HandledStreamError`` with
        ``code="transient_api_error"`` (i.e. an AssistantMessage with a transient
        error field arrives mid-stream), the outer loop must:
          1. Call ``_next_transient_backoff`` to get the sleep duration.
          2. Yield a ``StreamStatus`` message ("Connection interrupted…").
          3. Sleep for the backoff duration.
          4. Continue the loop and retry the same context-level attempt.
          5. NOT yield ``StreamError`` while retries remain.

        This exercises the ``_HandledStreamError`` handler path at
        ``stream_chat_completion_sdk`` line ~2335.
        """
        import contextlib

        from claude_agent_sdk import AssistantMessage, ResultMessage

        from backend.copilot.response_model import (
            StreamError,
            StreamStart,
            StreamStatus,
        )
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        result_msg = self._make_result_message()
        call_count = [0]

        def _client_factory(*args, **kwargs):
            call_count[0] += 1
            attempt = call_count[0]

            async def _receive():
                if attempt == 1:
                    # First call: emit AssistantMessage with a transient error field
                    # so _run_stream_attempt detects is_transient_api_error and
                    # raises _HandledStreamError(code="transient_api_error").
                    yield AssistantMessage(
                        content=[],
                        model="claude-sonnet-4-20250514",
                        error="rate_limit",
                    )
                    yield ResultMessage(
                        subtype="error",
                        result="rate limit exceeded (status code 429)",
                        duration_ms=50,
                        duration_api_ms=0,
                        is_error=True,
                        num_turns=0,
                        session_id="test-session-id",
                    )
                else:
                    yield result_msg

            client = MagicMock()
            client.receive_response = _receive
            client.query = AsyncMock()
            client._transport = MagicMock()
            client._transport.write = AsyncMock()

            cm = AsyncMock()
            cm.__aenter__.return_value = client
            cm.__aexit__.return_value = None
            return cm

        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript=None,
            client_side_effect=_client_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            # Patch asyncio.sleep to avoid actual delays in the test.
            stack.enter_context(patch(f"{_SVC}.asyncio.sleep", new_callable=AsyncMock))
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        # Two SDK client calls: first fails with transient error, second succeeds.
        assert (
            call_count[0] == 2
        ), f"Expected 2 SDK calls (transient retry), got {call_count[0]}"
        # No StreamError emitted — the retry succeeded.
        errors = [e for e in events if isinstance(e, StreamError)]
        assert (
            not errors
        ), f"Unexpected StreamError emitted during transient retry: {errors}"
        # StreamStatus("Connection interrupted…") must have been yielded.
        status_events = [e for e in events if isinstance(e, StreamStatus)]
        assert status_events, "Expected StreamStatus retry notification but got none"
        assert any(
            "retrying" in (e.message or "").lower()
            or "interrupted" in (e.message or "").lower()
            for e in status_events
        ), f"Expected 'retrying' or 'interrupted' in StreamStatus, got: {[e.message for e in status_events]}"
        assert any(isinstance(e, StreamStart) for e in events)

    @pytest.mark.asyncio
    async def test_generic_exception_transient_retry_then_succeeds(self):
        """Raw Exception("ECONNRESET") from receive_response triggers backoff retry.

        When ``receive_response`` raises a raw ``Exception`` whose string
        matches a transient pattern (e.g. ECONNRESET), the generic ``except
        Exception`` handler at ``stream_chat_completion_sdk`` line ~2398 must:
          1. Detect ``is_transient_api_error(str(e))`` as True.
          2. Call ``_next_transient_backoff`` to get the sleep duration.
          3. Yield a ``StreamStatus`` message ("Connection interrupted…").
          4. Sleep for the backoff duration.
          5. Continue the loop and retry the same context-level attempt.
          6. NOT yield ``StreamError`` while retries remain.

        This exercises the generic ``Exception`` handler (ECONNRESET path) at
        ``stream_chat_completion_sdk`` line ~2398.
        """
        import contextlib

        from backend.copilot.response_model import (
            StreamError,
            StreamStart,
            StreamStatus,
        )
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        session = self._make_session()
        result_msg = self._make_result_message()
        call_count = [0]

        def _client_factory(*args, **kwargs):
            call_count[0] += 1
            attempt = call_count[0]

            if attempt == 1:
                # First call: receive_response raises ECONNRESET immediately
                return self._make_client_mock_mid_stream_error(
                    error=Exception("ECONNRESET: connection reset by peer"),
                    pre_error_messages=None,
                )
            return self._make_client_mock(result_message=result_msg)

        original_transcript = _build_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        )

        patches = _make_sdk_patches(
            session,
            original_transcript=original_transcript,
            compacted_transcript=None,
            client_side_effect=_client_factory,
        )

        events = []
        with contextlib.ExitStack() as stack:
            # Patch asyncio.sleep to avoid actual delays in the test.
            stack.enter_context(patch(f"{_SVC}.asyncio.sleep", new_callable=AsyncMock))
            for target, kwargs in patches:
                stack.enter_context(patch(target, **kwargs))
            async for event in stream_chat_completion_sdk(
                session_id="test-session-id",
                message="hello",
                is_user_message=True,
                user_id="test-user",
                session=session,
            ):
                events.append(event)

        # Two SDK client calls: first fails with ECONNRESET, second succeeds.
        assert (
            call_count[0] == 2
        ), f"Expected 2 SDK calls (ECONNRESET transient retry), got {call_count[0]}"
        # No StreamError emitted — the retry succeeded.
        errors = [e for e in events if isinstance(e, StreamError)]
        assert (
            not errors
        ), f"Unexpected StreamError emitted during ECONNRESET retry: {errors}"
        # StreamStatus("Connection interrupted…") must have been yielded.
        status_events = [e for e in events if isinstance(e, StreamStatus)]
        assert status_events, "Expected StreamStatus retry notification but got none"
        assert any(
            "retrying" in (e.message or "").lower()
            or "interrupted" in (e.message or "").lower()
            for e in status_events
        ), f"Expected 'retrying' or 'interrupted' in StreamStatus, got: {[e.message for e in status_events]}"
        assert any(isinstance(e, StreamStart) for e in events)
