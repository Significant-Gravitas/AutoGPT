"""Unit tests for extracted service helpers.

Covers ``_is_prompt_too_long``, ``_reduce_context``, ``_iter_sdk_messages``,
``ReducedContext``, and the ``is_parallel_continuation`` logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock

from .conftest import build_test_transcript as _build_transcript
from .service import (
    _RETRY_TARGET_TOKENS,
    ReducedContext,
    _is_prompt_too_long,
    _is_tool_only_message,
    _iter_sdk_messages,
    _normalize_model_name,
    _reduce_context,
    _TokenUsage,
)

# ---------------------------------------------------------------------------
# _is_prompt_too_long
# ---------------------------------------------------------------------------


class TestIsPromptTooLong:
    def test_direct_match(self) -> None:
        assert _is_prompt_too_long(Exception("prompt is too long")) is True

    def test_case_insensitive(self) -> None:
        assert _is_prompt_too_long(Exception("PROMPT IS TOO LONG")) is True

    def test_no_match(self) -> None:
        assert _is_prompt_too_long(Exception("network timeout")) is False

    def test_request_too_large(self) -> None:
        assert _is_prompt_too_long(Exception("request too large for model")) is True

    def test_context_length_exceeded(self) -> None:
        assert _is_prompt_too_long(Exception("context_length_exceeded")) is True

    def test_max_tokens_exceeded_not_matched(self) -> None:
        """'max_tokens_exceeded' is intentionally excluded (too broad)."""
        assert _is_prompt_too_long(Exception("max_tokens_exceeded")) is False

    def test_max_tokens_config_error_no_match(self) -> None:
        """'max_tokens must be at least 1' should NOT match."""
        assert _is_prompt_too_long(Exception("max_tokens must be at least 1")) is False

    def test_chained_cause(self) -> None:
        inner = Exception("prompt is too long")
        outer = RuntimeError("SDK error")
        outer.__cause__ = inner
        assert _is_prompt_too_long(outer) is True

    def test_chained_context(self) -> None:
        inner = Exception("request too large")
        outer = RuntimeError("wrapped")
        outer.__context__ = inner
        assert _is_prompt_too_long(outer) is True

    def test_deep_chain(self) -> None:
        bottom = Exception("maximum context length")
        middle = RuntimeError("middle")
        middle.__cause__ = bottom
        top = ValueError("top")
        top.__cause__ = middle
        assert _is_prompt_too_long(top) is True

    def test_chain_no_match(self) -> None:
        inner = Exception("rate limit exceeded")
        outer = RuntimeError("wrapped")
        outer.__cause__ = inner
        assert _is_prompt_too_long(outer) is False

    def test_cycle_detection(self) -> None:
        """Exception chain with a cycle should not infinite-loop."""
        a = Exception("error a")
        b = Exception("error b")
        a.__cause__ = b
        b.__cause__ = a  # cycle
        assert _is_prompt_too_long(a) is False

    def test_all_patterns(self) -> None:
        patterns = [
            "prompt is too long",
            "request too large",
            "maximum context length",
            "context_length_exceeded",
            "input tokens exceed",
            "input is too long",
            "content length exceeds",
        ]
        for pattern in patterns:
            assert _is_prompt_too_long(Exception(pattern)) is True, pattern


# ---------------------------------------------------------------------------
# _reduce_context
# ---------------------------------------------------------------------------


class TestReduceContext:
    @pytest.mark.asyncio
    async def test_first_retry_compaction_success(self) -> None:
        # After compaction the retry runs WITHOUT --resume because we cannot
        # inject the compacted content into the CLI's native session file format.
        # The compacted builder state is still set for future upload_transcript.
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])
        compacted = _build_transcript([("user", "hi"), ("assistant", "[summary]")])

        with (
            patch(
                "backend.copilot.sdk.service.compact_transcript",
                new_callable=AsyncMock,
                return_value=compacted,
            ),
            patch(
                "backend.copilot.sdk.service.validate_transcript",
                return_value=True,
            ),
        ):
            ctx = await _reduce_context(
                transcript, False, "sess-123", "/tmp/cwd", "[test]"
            )

        assert isinstance(ctx, ReducedContext)
        assert ctx.use_resume is False
        assert ctx.resume_file is None
        assert ctx.transcript_lost is False
        assert ctx.tried_compaction is True

    @pytest.mark.asyncio
    async def test_compaction_fails_drops_transcript(self) -> None:
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])

        with patch(
            "backend.copilot.sdk.service.compact_transcript",
            new_callable=AsyncMock,
            return_value=None,
        ):
            ctx = await _reduce_context(
                transcript, False, "sess-123", "/tmp/cwd", "[test]"
            )

        assert ctx.use_resume is False
        assert ctx.resume_file is None
        assert ctx.transcript_lost is True
        assert ctx.tried_compaction is True

    @pytest.mark.asyncio
    async def test_already_tried_compaction_skips(self) -> None:
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])

        ctx = await _reduce_context(transcript, True, "sess-123", "/tmp/cwd", "[test]")

        assert ctx.use_resume is False
        assert ctx.transcript_lost is True
        assert ctx.tried_compaction is True

    @pytest.mark.asyncio
    async def test_empty_transcript_drops(self) -> None:
        ctx = await _reduce_context("", False, "sess-123", "/tmp/cwd", "[test]")

        assert ctx.use_resume is False
        assert ctx.transcript_lost is True

    @pytest.mark.asyncio
    async def test_compaction_returns_same_content_drops(self) -> None:
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])

        with patch(
            "backend.copilot.sdk.service.compact_transcript",
            new_callable=AsyncMock,
            return_value=transcript,  # same content
        ):
            ctx = await _reduce_context(
                transcript, False, "sess-123", "/tmp/cwd", "[test]"
            )

        assert ctx.transcript_lost is True

    @pytest.mark.asyncio
    async def test_compaction_invalid_transcript_drops(self) -> None:
        # When validate_transcript returns False for compacted content, drop transcript.
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])
        compacted = _build_transcript([("user", "hi"), ("assistant", "[summary]")])

        with (
            patch(
                "backend.copilot.sdk.service.compact_transcript",
                new_callable=AsyncMock,
                return_value=compacted,
            ),
            patch(
                "backend.copilot.sdk.service.validate_transcript",
                return_value=False,
            ),
        ):
            ctx = await _reduce_context(
                transcript, False, "sess-123", "/tmp/cwd", "[test]"
            )

        assert ctx.transcript_lost is True

    @pytest.mark.asyncio
    async def test_drop_returns_target_tokens_attempt_1(self) -> None:
        ctx = await _reduce_context("", False, "sess-1", "/tmp", "[t]", attempt=1)
        assert ctx.transcript_lost is True
        assert ctx.target_tokens == _RETRY_TARGET_TOKENS[0]

    @pytest.mark.asyncio
    async def test_drop_returns_target_tokens_attempt_2(self) -> None:
        ctx = await _reduce_context("", False, "sess-1", "/tmp", "[t]", attempt=2)
        assert ctx.transcript_lost is True
        assert ctx.target_tokens == _RETRY_TARGET_TOKENS[1]

    @pytest.mark.asyncio
    async def test_drop_clamps_attempt_beyond_limits(self) -> None:
        ctx = await _reduce_context("", False, "sess-1", "/tmp", "[t]", attempt=99)
        assert ctx.transcript_lost is True
        assert ctx.target_tokens == _RETRY_TARGET_TOKENS[-1]


# ---------------------------------------------------------------------------
# _iter_sdk_messages
# ---------------------------------------------------------------------------


class TestIterSdkMessages:
    @pytest.mark.asyncio
    async def test_yields_messages(self) -> None:
        messages = ["msg1", "msg2", "msg3"]
        client = AsyncMock()

        async def _fake_receive() -> AsyncGenerator[str]:
            for m in messages:
                yield m

        client.receive_response = _fake_receive
        result = [msg async for msg in _iter_sdk_messages(client)]
        assert result == messages

    @pytest.mark.asyncio
    async def test_heartbeat_on_timeout(self) -> None:
        """Yields None when asyncio.wait times out."""
        client = AsyncMock()
        received: list = []

        async def _slow_receive() -> AsyncGenerator[str]:
            await asyncio.sleep(100)  # never completes
            yield "never"  # pragma: no cover — unreachable, yield makes this an async generator

        client.receive_response = _slow_receive

        with patch("backend.copilot.sdk.service._HEARTBEAT_INTERVAL", 0.01):
            count = 0
            async for msg in _iter_sdk_messages(client):
                received.append(msg)
                count += 1
                if count >= 3:
                    break

        assert all(m is None for m in received)

    @pytest.mark.asyncio
    async def test_exception_propagates(self) -> None:
        client = AsyncMock()

        async def _error_receive() -> AsyncGenerator[str]:
            raise RuntimeError("SDK crash")
            yield  # pragma: no cover — unreachable, yield makes this an async generator

        client.receive_response = _error_receive

        with pytest.raises(RuntimeError, match="SDK crash"):
            async for _ in _iter_sdk_messages(client):
                pass

    @pytest.mark.asyncio
    async def test_task_cleanup_on_break(self) -> None:
        """Pending task is cancelled when generator is closed."""
        client = AsyncMock()

        async def _slow_receive() -> AsyncGenerator[str]:
            yield "first"
            await asyncio.sleep(100)
            yield "second"

        client.receive_response = _slow_receive

        gen = _iter_sdk_messages(client)
        first = await gen.__anext__()
        assert first == "first"
        await gen.aclose()  # should cancel pending task cleanly


# ---------------------------------------------------------------------------
# is_parallel_continuation logic
# ---------------------------------------------------------------------------


class TestIsParallelContinuation:
    """Unit tests for the is_parallel_continuation expression in the streaming loop.

    Verifies the vacuous-truth guard (empty content must return False) and the
    boundary cases for mixed TextBlock+ToolUseBlock messages.
    """

    def _make_tool_block(self) -> MagicMock:
        block = MagicMock(spec=ToolUseBlock)
        return block

    def test_all_tool_use_blocks_is_parallel(self):
        """AssistantMessage with only ToolUseBlocks is a parallel continuation."""
        msg = MagicMock(spec=AssistantMessage)
        msg.content = [self._make_tool_block(), self._make_tool_block()]
        assert _is_tool_only_message(msg) is True

    def test_empty_content_is_not_parallel(self):
        """AssistantMessage with empty content must NOT be treated as parallel.

        Without the bool(sdk_msg.content) guard, all() on an empty iterable
        returns True via vacuous truth — this test ensures the guard is present.
        """
        msg = MagicMock(spec=AssistantMessage)
        msg.content = []
        assert _is_tool_only_message(msg) is False

    def test_mixed_text_and_tool_blocks_not_parallel(self):
        """AssistantMessage with text + tool blocks is NOT a parallel continuation."""
        msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        msg.content = [text_block, self._make_tool_block()]
        assert _is_tool_only_message(msg) is False

    def test_non_assistant_message_not_parallel(self):
        """Non-AssistantMessage types are never parallel continuations."""
        assert _is_tool_only_message("not a message") is False
        assert _is_tool_only_message(None) is False
        assert _is_tool_only_message(42) is False

    def test_single_tool_block_is_parallel(self):
        """Single ToolUseBlock AssistantMessage is a parallel continuation."""
        msg = MagicMock(spec=AssistantMessage)
        msg.content = [self._make_tool_block()]
        assert _is_tool_only_message(msg) is True


# ---------------------------------------------------------------------------
# _normalize_model_name — used by per-request model override
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    """Unit tests for the model-name normalisation helper.

    The per-request model toggle calls _normalize_model_name with either
    ``"anthropic/claude-opus-4-6"`` (for 'advanced') or ``config.model`` (for
    'standard').  These tests verify the OpenRouter/provider-prefix stripping
    that keeps the value compatible with the Claude CLI.
    """

    def test_strips_anthropic_prefix(self):
        assert _normalize_model_name("anthropic/claude-opus-4-6") == "claude-opus-4-6"

    def test_strips_openai_prefix(self):
        assert _normalize_model_name("openai/gpt-4o") == "gpt-4o"

    def test_strips_google_prefix(self):
        assert _normalize_model_name("google/gemini-2.5-flash") == "gemini-2.5-flash"

    def test_already_normalized_unchanged(self):
        assert (
            _normalize_model_name("claude-sonnet-4-20250514")
            == "claude-sonnet-4-20250514"
        )

    def test_empty_string_unchanged(self):
        assert _normalize_model_name("") == ""

    def test_opus_model_roundtrip(self):
        """The exact string used for the 'opus' toggle strips correctly."""
        assert _normalize_model_name("anthropic/claude-opus-4-6") == "claude-opus-4-6"

    def test_sonnet_openrouter_model(self):
        """Sonnet model as stored in config (OpenRouter-prefixed) strips cleanly."""
        assert _normalize_model_name("anthropic/claude-sonnet-4") == "claude-sonnet-4"


# ---------------------------------------------------------------------------
# _TokenUsage — null-safe accumulation (OpenRouter initial-stream-event bug)
# ---------------------------------------------------------------------------


class TestTokenUsageNullSafety:
    """Verify that ResultMessage.usage dicts with null-valued cache fields
    (as emitted by OpenRouter for the initial streaming event before real
    token counts are available) do not crash the accumulator.

    Before the fix, dict.get("cache_read_input_tokens", 0) returned None
    when the key existed with a null value, causing 'int += None' TypeError.
    """

    def _apply_usage(self, usage: dict, acc: _TokenUsage) -> None:
        """Null-safe accumulation: ``or 0`` treats missing/None as zero.

        Uses ``usage.get("key") or 0`` rather than ``usage.get("key", 0)``
        because the latter returns ``None`` when the key exists with a null
        value, which would raise ``TypeError`` on ``int += None``.  This is
        the intentional pattern that fixes the OpenRouter initial-stream-event
        bug described in the class docstring.
        """
        acc.prompt_tokens += usage.get("input_tokens") or 0
        acc.cache_read_tokens += usage.get("cache_read_input_tokens") or 0
        acc.cache_creation_tokens += usage.get("cache_creation_input_tokens") or 0
        acc.completion_tokens += usage.get("output_tokens") or 0

    def test_null_cache_tokens_do_not_crash(self):
        """OpenRouter initial event: cache keys present with null value."""
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
        }
        acc = _TokenUsage()
        self._apply_usage(usage, acc)  # must not raise TypeError
        assert acc.prompt_tokens == 0
        assert acc.cache_read_tokens == 0
        assert acc.cache_creation_tokens == 0
        assert acc.completion_tokens == 0

    def test_real_cache_tokens_are_accumulated(self):
        """OpenRouter final event: real cache token counts are captured."""
        usage = {
            "input_tokens": 10,
            "output_tokens": 349,
            "cache_read_input_tokens": 16600,
            "cache_creation_input_tokens": 512,
        }
        acc = _TokenUsage()
        self._apply_usage(usage, acc)
        assert acc.prompt_tokens == 10
        assert acc.cache_read_tokens == 16600
        assert acc.cache_creation_tokens == 512
        assert acc.completion_tokens == 349

    def test_absent_cache_keys_default_to_zero(self):
        """Minimal usage dict without cache keys defaults correctly."""
        usage = {"input_tokens": 5, "output_tokens": 20}
        acc = _TokenUsage()
        self._apply_usage(usage, acc)
        assert acc.prompt_tokens == 5
        assert acc.cache_read_tokens == 0
        assert acc.cache_creation_tokens == 0
        assert acc.completion_tokens == 20

    def test_multi_turn_accumulation(self):
        """Null event followed by real event: only real tokens counted."""
        null_event = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
        }
        real_event = {
            "input_tokens": 10,
            "output_tokens": 349,
            "cache_read_input_tokens": 16600,
            "cache_creation_input_tokens": 512,
        }
        acc = _TokenUsage()
        self._apply_usage(null_event, acc)
        self._apply_usage(real_event, acc)
        assert acc.prompt_tokens == 10
        assert acc.cache_read_tokens == 16600
        assert acc.cache_creation_tokens == 512
        assert acc.completion_tokens == 349


# ---------------------------------------------------------------------------
# session_id / resume selection logic
# ---------------------------------------------------------------------------


def _build_sdk_options(
    use_resume: bool,
    resume_file: str | None,
    session_id: str,
) -> dict:
    """Mirror the session_id/resume selection in stream_chat_completion_sdk.

    This helper encodes the exact branching so the unit tests stay in sync
    with the production code without needing to invoke the full generator.
    """
    kwargs: dict = {}
    if use_resume and resume_file:
        kwargs["resume"] = resume_file
    else:
        kwargs["session_id"] = session_id
    return kwargs


def _build_retry_sdk_options(
    initial_kwargs: dict,
    ctx_use_resume: bool,
    ctx_resume_file: str | None,
    session_id: str,
) -> dict:
    """Mirror the retry branch in stream_chat_completion_sdk."""
    retry: dict = dict(initial_kwargs)
    if ctx_use_resume and ctx_resume_file:
        retry["resume"] = ctx_resume_file
        retry.pop("session_id", None)
    elif "session_id" in initial_kwargs:
        retry.pop("resume", None)
        retry["session_id"] = session_id
    else:
        retry.pop("resume", None)
        retry.pop("session_id", None)
    return retry


class TestSdkSessionIdSelection:
    """Verify that session_id is set for all non-resume turns.

    Regression test for the mode-switch T1 bug: when a user switches from
    baseline mode (fast) to SDK mode (extended_thinking) mid-session, the
    first SDK turn has has_history=True but no CLI session file.  The old
    code gated session_id on ``not has_history``, so mode-switch T1 never
    got a session_id — the CLI used a random ID that couldn't be found on
    the next turn, causing --resume to fail for the whole session.
    """

    SESSION_ID = "sess-abc123"

    def test_t1_fresh_sets_session_id(self):
        """T1 of a fresh session always gets session_id."""
        opts = _build_sdk_options(
            use_resume=False,
            resume_file=None,
            session_id=self.SESSION_ID,
        )
        assert opts.get("session_id") == self.SESSION_ID
        assert "resume" not in opts

    def test_mode_switch_t1_sets_session_id(self):
        """Mode-switch T1 (has_history=True, no CLI session) gets session_id.

        Before the fix, the ``elif not has_history`` guard prevented this
        case from setting session_id, causing all subsequent turns to run
        without --resume.
        """
        # Mode-switch T1: use_resume=False (no prior CLI session) and
        # has_history=True (prior baseline turns in DB). The old code
        # (``elif not has_history``) silently skipped this case.
        opts = _build_sdk_options(
            use_resume=False,
            resume_file=None,
            session_id=self.SESSION_ID,
        )
        assert opts.get("session_id") == self.SESSION_ID
        assert "resume" not in opts

    def test_t2_with_resume_uses_resume(self):
        """T2+ with a restored CLI session uses --resume, not session_id."""
        opts = _build_sdk_options(
            use_resume=True,
            resume_file=self.SESSION_ID,
            session_id=self.SESSION_ID,
        )
        assert opts.get("resume") == self.SESSION_ID
        assert "session_id" not in opts

    def test_t2_without_resume_sets_session_id(self):
        """T2+ when restore failed still gets session_id (no prior file on disk)."""
        opts = _build_sdk_options(
            use_resume=False,
            resume_file=None,
            session_id=self.SESSION_ID,
        )
        assert opts.get("session_id") == self.SESSION_ID
        assert "resume" not in opts

    def test_retry_keeps_session_id_for_t1(self):
        """Retry for T1 (or mode-switch T1) preserves session_id."""
        initial = _build_sdk_options(False, None, self.SESSION_ID)
        retry = _build_retry_sdk_options(initial, False, None, self.SESSION_ID)
        assert retry.get("session_id") == self.SESSION_ID
        assert "resume" not in retry

    def test_retry_removes_session_id_for_t2_plus(self):
        """Retry for T2+ (initial used --resume) removes session_id to avoid conflict."""
        initial = _build_sdk_options(True, self.SESSION_ID, self.SESSION_ID)
        # T2+ retry where context reduction dropped --resume
        retry = _build_retry_sdk_options(initial, False, None, self.SESSION_ID)
        assert "session_id" not in retry
        assert "resume" not in retry

    def test_retry_t2_with_resume_sets_resume(self):
        """Retry that still uses --resume keeps --resume and drops session_id."""
        initial = _build_sdk_options(True, self.SESSION_ID, self.SESSION_ID)
        retry = _build_retry_sdk_options(
            initial, True, self.SESSION_ID, self.SESSION_ID
        )
        assert retry.get("resume") == self.SESSION_ID
        assert "session_id" not in retry
