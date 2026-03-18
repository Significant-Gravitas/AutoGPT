"""Unit tests for extracted service helpers.

Covers ``_is_prompt_too_long``, ``_reduce_context``, ``_iter_sdk_messages``,
and the ``ReducedContext`` named tuple.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

from .conftest import build_test_transcript as _build_transcript
from .service import (
    ReducedContext,
    _is_prompt_too_long,
    _iter_sdk_messages,
    _reduce_context,
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
            patch(
                "backend.copilot.sdk.service.write_transcript_to_tempfile",
                return_value="/tmp/resume.jsonl",
            ),
        ):
            ctx = await _reduce_context(
                transcript, False, "sess-123", "/tmp/cwd", "[test]"
            )

        assert isinstance(ctx, ReducedContext)
        assert ctx.use_resume is True
        assert ctx.resume_file == "/tmp/resume.jsonl"
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
    async def test_write_tempfile_fails_drops(self) -> None:
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
            patch(
                "backend.copilot.sdk.service.write_transcript_to_tempfile",
                return_value=None,
            ),
        ):
            ctx = await _reduce_context(
                transcript, False, "sess-123", "/tmp/cwd", "[test]"
            )

        assert ctx.transcript_lost is True


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
