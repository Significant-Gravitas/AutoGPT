"""Tests for <internal_reasoning> / <thinking> tag stripping in the SDK path.

Covers the ThinkingStripper integration in ``_dispatch_response`` — verifying
that reasoning tags emitted by non-extended-thinking models (e.g. Sonnet) are
stripped from the SSE stream and the persisted assistant message.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamTextDelta
from backend.copilot.sdk.service import _dispatch_response, _StreamAccumulator

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_ctx() -> MagicMock:
    """Build a minimal _StreamContext mock."""
    ctx = MagicMock()
    ctx.session = ChatSession(
        session_id="test",
        user_id="test-user",
        title="test",
        messages=[],
        usage=[],
        started_at=_NOW,
        updated_at=_NOW,
    )
    ctx.log_prefix = "[test]"
    return ctx


def _make_state() -> MagicMock:
    """Build a minimal _RetryState mock."""
    state = MagicMock()
    state.transcript_builder = MagicMock()
    return state


def _make_acc() -> _StreamAccumulator:
    return _StreamAccumulator(
        assistant_response=ChatMessage(role="assistant", content=""),
        accumulated_tool_calls=[],
    )


class TestDispatchResponseThinkingStrip:
    """Verify _dispatch_response strips reasoning tags from text deltas."""

    def test_internal_reasoning_stripped_from_delta(self) -> None:
        """Full <internal_reasoning> block in one delta is stripped."""
        acc = _make_acc()
        ctx = _make_ctx()
        state = _make_state()

        response = StreamTextDelta(
            id="t1",
            delta="<internal_reasoning>step by step</internal_reasoning>The answer is 42",
        )
        result = _dispatch_response(response, acc, ctx, state, False, "[test]")

        assert result is not None
        assert isinstance(result, StreamTextDelta)
        assert "internal_reasoning" not in result.delta
        assert result.delta == "The answer is 42"
        assert acc.assistant_response.content == "The answer is 42"

    def test_thinking_tag_stripped(self) -> None:
        """<thinking> blocks are also stripped."""
        acc = _make_acc()
        ctx = _make_ctx()
        state = _make_state()

        response = StreamTextDelta(
            id="t1",
            delta="<thinking>hmm</thinking>Hello!",
        )
        result = _dispatch_response(response, acc, ctx, state, False, "[test]")

        assert result is not None
        assert result.delta == "Hello!"
        assert acc.assistant_response.content == "Hello!"

    def test_partial_tag_buffers(self) -> None:
        """A partial opening tag causes the delta to be suppressed."""
        acc = _make_acc()
        ctx = _make_ctx()
        state = _make_state()

        # First chunk ends mid-tag — stripper buffers, nothing to emit.
        r1 = _dispatch_response(
            StreamTextDelta(id="t1", delta="Hello <inter"),
            acc,
            ctx,
            state,
            False,
            "[test]",
        )
        # The stripper emits "Hello " but buffers "<inter".
        # With "Hello " the dispatch should still yield.
        if r1 is None:
            # If the entire chunk was buffered, the accumulated content is empty.
            assert acc.assistant_response.content == ""
        else:
            assert "inter" not in r1.delta

        # Second chunk completes the tag + provides visible text.
        _dispatch_response(
            StreamTextDelta(
                id="t1", delta="nal_reasoning>secret</internal_reasoning> world"
            ),
            acc,
            ctx,
            state,
            False,
            "[test]",
        )
        content = acc.assistant_response.content or ""
        tail = acc.thinking_stripper.flush()
        full = content + tail
        assert "secret" not in full
        assert "world" in full

    def test_plain_text_unchanged(self) -> None:
        """Text without reasoning tags passes through unmodified."""
        acc = _make_acc()
        ctx = _make_ctx()
        state = _make_state()

        response = StreamTextDelta(id="t1", delta="Just normal text")
        result = _dispatch_response(response, acc, ctx, state, False, "[test]")

        assert result is not None
        # The stripper may buffer trailing chars that look like tag starts.
        # Flush to get everything.
        flushed = acc.thinking_stripper.flush()
        full = (result.delta or "") + flushed
        assert full == "Just normal text"

    def test_multi_delta_accumulation(self) -> None:
        """Multiple clean deltas accumulate correctly."""
        acc = _make_acc()
        ctx = _make_ctx()
        state = _make_state()

        _dispatch_response(
            StreamTextDelta(id="t1", delta="Hello "),
            acc,
            ctx,
            state,
            False,
            "[test]",
        )
        _dispatch_response(
            StreamTextDelta(id="t1", delta="world"),
            acc,
            ctx,
            state,
            False,
            "[test]",
        )
        tail = acc.thinking_stripper.flush()
        full = (acc.assistant_response.content or "") + tail
        assert full == "Hello world"

    def test_reasoning_only_delta_suppressed(self) -> None:
        """A delta containing only reasoning content emits nothing."""
        acc = _make_acc()
        ctx = _make_ctx()
        state = _make_state()

        result = _dispatch_response(
            StreamTextDelta(
                id="t1",
                delta="<internal_reasoning>all hidden</internal_reasoning>",
            ),
            acc,
            ctx,
            state,
            False,
            "[test]",
        )
        assert result is None
        assert acc.assistant_response.content == ""
