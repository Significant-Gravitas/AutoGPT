"""Unit tests for extracted service helpers.

Covers ``_is_prompt_too_long``, ``_reduce_context``, ``_iter_sdk_messages``
and ``ReducedContext``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import RouteProfile, codex_profile, platform_profile
from backend.util.prompt import (
    DEFAULT_COMPRESSION_RESERVE,
    compress_context,
    estimate_token_count,
    get_compression_target,
)

from ..constants import COPILOT_ERROR_PREFIX
from ..model import ChatMessage, ChatSession
from ..transcript_builder import TranscriptBuilder
from .compaction import CompactionStats, CompactionTracker
from .conftest import build_test_transcript as _build_transcript
from .service import (
    ReducedContext,
    _build_query_message,
    _compress_messages,
    _compression_model,
    _context_limit_without_compaction,
    _expect_pre_query_compaction,
    _is_prompt_too_long,
    _iter_sdk_messages,
    _reduce_context,
    _restore_cli_session_for_turn,
    _retry_reduced_context,
    _TokenUsage,
    _will_compact,
)


def _profile(
    model: str = "claude-sonnet-4-6", route: str = "direct_anthropic"
) -> RouteProfile:
    """A turn's route profile for *model* on the platform *route*."""
    cfg = ChatConfig(
        platform_route=route, anthropic_api_key="k", openrouter_api_key="k"
    )
    return platform_profile(cfg, model, "env")


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
        ):
            ctx = await _reduce_context(transcript, "[test]", profile=_profile())

        assert isinstance(ctx, ReducedContext)
        assert ctx.transcript_lost is False

    @pytest.mark.asyncio
    async def test_compaction_success_keeps_the_pinned_budget(self) -> None:
        # The retry still rebuilds its history from the DB rows, held to the
        # budget the transcript was compacted to: the pinned window's, never
        # the compressor's catalog default (far past the pin on Codex).
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])
        compacted = _build_transcript([("user", "hi"), ("assistant", "[summary]")])
        compact = AsyncMock(return_value=compacted)
        profile = _profile("anthropic/claude-sonnet-5-5", "openrouter")

        with (
            patch("backend.copilot.sdk.service.compact_transcript", new=compact),
            patch(
                "backend.copilot.sdk.service.validate_transcript",
                return_value=True,
            ),
        ):
            ctx = await _reduce_context(transcript, "[test]", profile=profile)

        assert ctx.transcript_lost is False
        assert ctx.target_tokens == profile.compaction_target_tokens()
        assert ctx.target_tokens == compact.await_args.kwargs["target_tokens"]

    @pytest.mark.asyncio
    async def test_compaction_fails_drops_transcript(self) -> None:
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])

        with patch(
            "backend.copilot.sdk.service.compact_transcript",
            new_callable=AsyncMock,
            return_value=None,
        ):
            ctx = await _reduce_context(transcript, "[test]", profile=_profile())

        assert ctx.transcript_lost is True

    @pytest.mark.asyncio
    async def test_a_later_retry_skips_compaction(self) -> None:
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])
        compact = AsyncMock()

        with patch("backend.copilot.sdk.service.compact_transcript", new=compact):
            ctx = await _reduce_context(
                transcript, "[test]", profile=_profile(), attempt=2
            )

        compact.assert_not_awaited()
        assert ctx.transcript_lost is True

    @pytest.mark.asyncio
    async def test_empty_transcript_drops(self) -> None:
        ctx = await _reduce_context("", "[test]", profile=_profile())

        assert ctx.transcript_lost is True

    @pytest.mark.asyncio
    async def test_compaction_returns_same_content_drops(self) -> None:
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])

        with patch(
            "backend.copilot.sdk.service.compact_transcript",
            new_callable=AsyncMock,
            return_value=transcript,  # same content
        ):
            ctx = await _reduce_context(transcript, "[test]", profile=_profile())

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
            ctx = await _reduce_context(transcript, "[test]", profile=_profile())

        assert ctx.transcript_lost is True

    @pytest.mark.asyncio
    async def test_drop_returns_target_tokens_attempt_1(self) -> None:
        ctx = await _reduce_context("", "[t]", profile=_profile(), attempt=1)
        assert ctx.transcript_lost is True
        assert ctx.target_tokens == _profile().retry_target_tokens()[0]

    @pytest.mark.asyncio
    async def test_drop_returns_target_tokens_attempt_2(self) -> None:
        ctx = await _reduce_context("", "[t]", profile=_profile(), attempt=2)
        assert ctx.transcript_lost is True
        assert ctx.target_tokens == _profile().retry_target_tokens()[1]

    @pytest.mark.asyncio
    async def test_drop_clamps_attempt_beyond_limits(self) -> None:
        ctx = await _reduce_context("", "[t]", profile=_profile(), attempt=99)
        assert ctx.transcript_lost is True
        assert ctx.target_tokens == _profile().retry_target_tokens()[-1]


class TestCompactionTargetTokens:
    """The retry's compaction target sits below the CLI's own threshold for
    the turn's window and trigger, on every route."""

    @pytest.mark.parametrize(
        ("trigger", "expected"),
        [
            # 200K window with a 50% trigger -> CLI threshold 100K -> 80K
            (50, 80_000),
            # No trigger: the CLI's default (window - 13K = 187K) -> 167K
            (None, 167_000),
            # An aggressive 30% -> CLI threshold 60K -> 40K
            (30, 40_000),
        ],
    )
    def test_target_below_the_cli_threshold(self, trigger, expected) -> None:
        profile = _profile().model_copy(
            update={"window": 200_000, "trigger_pct": trigger}
        )
        assert profile.compaction_target_tokens() == expected

    def test_openrouter_sonnet_compacts_at_half_its_window(self) -> None:
        profile = _profile("anthropic/claude-sonnet-4-6", "openrouter")
        assert (profile.window, profile.trigger_pct) == (200_000, 50)
        assert profile.compaction_target_tokens() == 80_000

    def test_moonshot_uses_the_cli_default_threshold(self) -> None:
        # A Moonshot SKU keeps the CLI's own trigger on OpenRouter.
        profile = _profile("moonshotai/kimi-k2.6", "openrouter")
        assert profile.trigger_pct is None
        assert profile.compaction_target_tokens() == profile.window - 13_000 - 20_000

    @pytest.mark.parametrize("model", ["gpt-6-astra", "moonshotai/kimi-k2.6"])
    def test_codex_compacts_at_its_engine_default(self, model) -> None:
        # min(272000*90//100, 272000-13000) - 20000 = 224800, whatever the
        # slug looks like: the Codex route keeps its own trigger.
        profile = codex_profile(
            model, effort=None, routing_source="env", codex_engine=None
        )
        assert profile.compaction_target_tokens() == 224_800

    def test_floor_at_10k_for_an_extremely_aggressive_trigger(self) -> None:
        # 1% of a 50K window -> CLI threshold 500 -> negative without the floor.
        profile = _profile().model_copy(update={"window": 50_000, "trigger_pct": 1})
        assert profile.compaction_target_tokens() == 10_000

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "profile",
        [
            _profile("moonshotai/kimi-k2.6", "openrouter"),
            codex_profile(
                "gpt-6-astra", effort=None, routing_source="env", codex_engine=None
            ),
        ],
        ids=["kimi", "codex"],
    )
    async def test_reduce_context_sizes_the_target_for_the_turns_route(
        self, profile
    ) -> None:
        """The compactor LLM is fixed, but the target is sized for the route
        the CLI is actually serving: its window and trigger."""
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])
        captured: dict = {}

        async def fake_compact(content, *, model, log_prefix, target_tokens):
            captured["target_tokens"] = target_tokens
            return None

        with patch(
            "backend.copilot.sdk.service.compact_transcript", side_effect=fake_compact
        ):
            await _reduce_context(transcript, "[t]", profile=profile)

        assert captured["target_tokens"] == profile.compaction_target_tokens()


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
    async def test_wake_event_cuts_heartbeat_short(self) -> None:
        """A set ``wake`` yields the sentinel without waiting out the interval."""
        client = AsyncMock()
        wake = asyncio.Event()

        async def _slow_receive() -> AsyncGenerator[str]:
            await asyncio.sleep(100)  # never completes
            yield "never"  # pragma: no cover — unreachable, yield makes this an async generator

        client.receive_response = _slow_receive

        with patch("backend.copilot.sdk.service._HEARTBEAT_INTERVAL", 100):
            gen = _iter_sdk_messages(client, wake=wake)
            asyncio.get_running_loop().call_later(0.01, wake.set)
            first = await asyncio.wait_for(gen.__anext__(), timeout=1)
            await gen.aclose()

        assert first is None
        assert not wake.is_set()

    @pytest.mark.asyncio
    async def test_wake_sentinel_precedes_concurrent_message(self) -> None:
        """A message landing with the wake is delivered after the sentinel."""
        client = AsyncMock()
        wake = asyncio.Event()

        async def _receive() -> AsyncGenerator[str]:
            await wake.wait()
            yield "msg"

        client.receive_response = _receive

        with patch("backend.copilot.sdk.service._HEARTBEAT_INTERVAL", 100):
            gen = _iter_sdk_messages(client, wake=wake)
            asyncio.get_running_loop().call_later(0.01, wake.set)
            received = [
                await asyncio.wait_for(gen.__anext__(), timeout=1) for _ in range(2)
            ]
            await gen.aclose()

        assert received == [None, "msg"]

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
# _normalize_model_name — used by per-request model override
# ---------------------------------------------------------------------------


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


def _build_retry_sdk_options(initial_kwargs: dict, session_id: str) -> dict:
    """Mirror the retry branch in stream_chat_completion_sdk.

    Production-side companion: ``delete_stale_cli_session_file`` is invoked
    on every retry so the CLI doesn't trip "Session ID already in use"
    when we re-attach ``session_id``.  This helper only mirrors the kwarg
    shape (file-system side effect is tested separately).
    """
    retry: dict = dict(initial_kwargs)
    retry.pop("resume", None)
    retry["session_id"] = session_id
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
        retry = _build_retry_sdk_options(initial, self.SESSION_ID)
        assert retry.get("session_id") == self.SESSION_ID
        assert "resume" not in retry

    def test_retry_keeps_session_id_for_t2_plus(self):
        """Retry for T2+ now keeps session_id so the recovery turn writes to
        the predictable ``cli_session_path`` and gets uploaded.  Production
        clears the stale local file via ``delete_stale_cli_session_file``
        before this branch runs to dodge "Session ID already in use".

        Regression guard for SENTRY-1207: previously this branch dropped
        session_id, the CLI wrote to a random path, and the post-turn
        upload silently grabbed the stale pre-failure file — so GCS stayed
        bloated and every subsequent turn re-tripped prompt-too-long.
        """
        initial = _build_sdk_options(True, self.SESSION_ID, self.SESSION_ID)
        # T2+ retry: context reduction drops --resume
        retry = _build_retry_sdk_options(initial, self.SESSION_ID)
        assert retry.get("session_id") == self.SESSION_ID
        assert "resume" not in retry


# ---------------------------------------------------------------------------
# _restore_cli_session_for_turn — mode check
# ---------------------------------------------------------------------------


class TestRestoreCliSessionModeCheck:
    """SDK skips --resume when the transcript was written by the old fast path."""

    @pytest.mark.asyncio
    async def test_a_fast_path_transcript_is_treated_as_none(self, tmp_path):
        """A transcript with mode='baseline' (a local chat from before the
        one-engine upgrade) is neither resumed nor read: the turn's context is
        the chat's own rows.
        """
        from datetime import UTC, datetime

        from backend.copilot.model import ChatMessage, ChatSession
        from backend.copilot.transcript import TranscriptDownload
        from backend.copilot.transcript_builder import TranscriptBuilder

        session = ChatSession(
            session_id="test-session",
            user_id="user-1",
            messages=[
                ChatMessage(role="user", content="hello-unique-marker", sequence=0),
                ChatMessage(
                    role="assistant", content="world-unique-marker", sequence=1
                ),
                ChatMessage(role="user", content="follow up", sequence=2),
            ],
            title="test",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        builder = TranscriptBuilder()
        # Baseline content with a sentinel that must NOT appear in the final transcript
        baseline_restore = TranscriptDownload(
            content=b'{"type":"user","uuid":"bad-uuid","message":{"role":"user","content":"BASELINE_SENTINEL"}}\n',
            message_count=1,
            mode="baseline",
        )

        download_mock = AsyncMock(return_value=baseline_restore)
        with (
            patch(
                "backend.copilot.sdk.service.download_transcript",
                new=download_mock,
            ),
        ):
            result = await _restore_cli_session_for_turn(
                user_id="user-1",
                session_id="test-session",
                session=session,
                sdk_cwd=str(tmp_path),
                transcript_builder=builder,
                log_prefix="[Test]",
            )

        # download_transcript was called (attempted GCS restore)
        download_mock.assert_awaited_once()
        assert result.use_resume is False
        assert result.context_messages is not None
        assert [m.content for m in result.context_messages] == [
            "hello-unique-marker",
            "world-unique-marker",
        ]
        assert result.transcript_content == ""
        assert result.transcript_msg_count == 2

    @pytest.mark.asyncio
    async def test_sdk_mode_transcript_allows_resume(self, tmp_path):
        """A valid SDK-written transcript is accepted for --resume."""
        import json as stdlib_json
        from datetime import UTC, datetime

        from backend.copilot.model import ChatMessage, ChatSession
        from backend.copilot.transcript import STOP_REASON_END_TURN, TranscriptDownload
        from backend.copilot.transcript_builder import TranscriptBuilder

        lines = [
            stdlib_json.dumps(
                {
                    "type": "user",
                    "uuid": "uid-0",
                    "parentUuid": "",
                    "message": {"role": "user", "content": "hi"},
                }
            ),
            stdlib_json.dumps(
                {
                    "type": "assistant",
                    "uuid": "uid-1",
                    "parentUuid": "uid-0",
                    "message": {
                        "role": "assistant",
                        "id": "msg_1",
                        "model": "test",
                        "type": "message",
                        "stop_reason": STOP_REASON_END_TURN,
                        "content": [{"type": "text", "text": "hello"}],
                    },
                }
            ),
        ]
        content = ("\n".join(lines) + "\n").encode("utf-8")

        session = ChatSession(
            session_id="test-session",
            user_id="user-1",
            messages=[
                ChatMessage(role="user", content="hi", sequence=0),
                ChatMessage(role="assistant", content="hello", sequence=1),
                ChatMessage(role="user", content="follow up", sequence=2),
            ],
            title="test",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        builder = TranscriptBuilder()
        sdk_restore = TranscriptDownload(
            content=content,
            message_count=2,
            mode="sdk",
        )

        with (
            patch(
                "backend.copilot.sdk.service.download_transcript",
                new=AsyncMock(return_value=sdk_restore),
            ),
        ):
            result = await _restore_cli_session_for_turn(
                user_id="user-1",
                session_id="test-session",
                session=session,
                sdk_cwd=str(tmp_path),
                transcript_builder=builder,
                log_prefix="[Test]",
            )

        assert result.use_resume is True


# ---------------------------------------------------------------------------
# delete_stale_cli_session_file — clears a leftover local session file so a
# subsequent --session-id (no --resume) invocation doesn't trip "Session ID
# already in use".  Critical for the prompt-too-long retry path.
# ---------------------------------------------------------------------------


class TestDeleteStaleCliSessionFile:
    def test_deletes_file_when_present(self, tmp_path) -> None:
        from backend.copilot.sdk.service import delete_stale_cli_session_file

        sdk_cwd = str(tmp_path / "cwd")
        session_id = "sess-deadbeef"

        with (
            patch(
                "backend.copilot.sdk.service.cli_session_path",
                return_value=str(tmp_path / "session.jsonl"),
            ),
            patch(
                "backend.copilot.sdk.service.projects_base",
                return_value=str(tmp_path),
            ),
        ):
            target = tmp_path / "session.jsonl"
            target.write_text("{}\n")

            removed = delete_stale_cli_session_file(sdk_cwd, session_id, "[t]")

            assert removed is True
            assert not target.exists()

    def test_returns_false_when_file_missing(self, tmp_path) -> None:
        from backend.copilot.sdk.service import delete_stale_cli_session_file

        with (
            patch(
                "backend.copilot.sdk.service.cli_session_path",
                return_value=str(tmp_path / "missing.jsonl"),
            ),
            patch(
                "backend.copilot.sdk.service.projects_base",
                return_value=str(tmp_path),
            ),
        ):
            removed = delete_stale_cli_session_file("/cwd", "sess", "[t]")

        assert removed is False

    def test_path_traversal_guard_rejects_outside_projects_base(self, tmp_path) -> None:
        """Refuse to delete files outside the projects base, even if they exist."""
        from backend.copilot.sdk.service import delete_stale_cli_session_file

        outside = tmp_path / "outside.jsonl"
        outside.write_text("data")
        projects = tmp_path / "projects"
        projects.mkdir()

        with (
            patch(
                "backend.copilot.sdk.service.cli_session_path",
                return_value=str(outside),
            ),
            patch(
                "backend.copilot.sdk.service.projects_base",
                return_value=str(projects),
            ),
        ):
            removed = delete_stale_cli_session_file("/cwd", "sess", "[t]")

        # File was outside projects base — guard rejected, file untouched.
        assert removed is False
        assert outside.exists()


# ---------------------------------------------------------------------------
# Empty-tool-call circuit breaker — must NOT false-positive on no-arg tools
# ---------------------------------------------------------------------------


class TestEmptyToolCallNoArgException:
    """No-arg copilot tools (e.g. ``get_agent_building_guide``) legitimately
    invoke with ``input == {}``. The breaker must NOT count them as the
    saturation-failure mode it targets — that would be a false positive
    every time the model calls such a tool, polluting Sentry and (after 5
    consecutive) aborting the stream wrongly.
    """

    def test_no_arg_tool_names_includes_mcp_prefixed_form(self) -> None:
        from backend.copilot.sdk.service import _no_arg_tool_names

        names = _no_arg_tool_names()
        # Both bare and MCP-prefixed forms — the SDK's ToolUseBlock.name
        # carries the MCP prefix when registered through the copilot
        # MCP server.
        assert "get_agent_building_guide" in names
        assert "mcp__copilot__get_agent_building_guide" in names

    def test_no_arg_tool_with_empty_input_does_not_trip_counter(self) -> None:
        """Regression guard: a single empty-input call to a registered
        no-arg tool must reset the consecutive counter (returning 0),
        same as a non-empty AssistantMessage. This proves
        ``get_agent_building_guide`` mid-session won't trip the
        circuit-breaker after 5 invocations across a long session."""
        from claude_agent_sdk import AssistantMessage, ToolUseBlock

        from backend.copilot.sdk.service import _check_empty_tool_breaker

        msg = AssistantMessage(
            content=[
                ToolUseBlock(
                    id="tu_1",
                    name="mcp__copilot__get_agent_building_guide",
                    input={},
                )
            ],
            model="anthropic/claude-sonnet-4-6",
        )
        # ctx + state aren't read on the no-arg fast-path — pass MagicMocks.
        ctx = MagicMock()
        ctx.log_prefix = "[t]"
        state = MagicMock()

        result = _check_empty_tool_breaker(msg, consecutive=4, ctx=ctx, state=state)

        # Consecutive counter MUST reset (no-arg tool is a normal action,
        # not the saturation failure).
        assert result.count == 0
        assert result.tripped is False


# ---------------------------------------------------------------------------
# _stamp_turn_messages — turn-bounded model/routing_source stamping
# ---------------------------------------------------------------------------


class TestStampTurnMessages:
    """The SDK path stamps at persist time; the stamp must be bounded to
    the turn (never back-stamping pre-feature NULL history) and must not
    overwrite already-stamped rows."""

    def _messages(self):
        from backend.copilot.model import ChatMessage

        return [
            ChatMessage(role="assistant", content="old", sequence=0),
            ChatMessage(role="user", content="q", sequence=1),
            ChatMessage(role="assistant", content="new", sequence=2),
        ]

    def test_stamps_only_this_turns_assistant_rows(self):
        from backend.copilot.sdk.service import _stamp_turn_messages

        msgs = self._messages()
        _stamp_turn_messages(
            msgs,
            start_index=1,
            requested_model="claude-sonnet-4-6",
            actual_model=None,
            routing_source="env",
        )
        assert msgs[0].model is None  # pre-turn history untouched
        assert msgs[1].model is None  # user rows untouched
        assert msgs[2].model == "claude-sonnet-4-6"
        assert msgs[2].routing_source == "env"

    def test_never_overwrites_an_existing_stamp(self):
        from backend.copilot.sdk.service import _stamp_turn_messages

        msgs = self._messages()
        msgs[2].model = "already/stamped"
        msgs[2].routing_source = "ld"
        _stamp_turn_messages(
            msgs,
            start_index=0,
            requested_model="claude-sonnet-4-6",
            actual_model=None,
            routing_source="env",
        )
        assert msgs[2].model == "already/stamped"
        assert msgs[2].routing_source == "ld"

    def test_fallback_divergence_restamps_source(self):
        """When the CLI's overload fallback served a different model than
        the routed one, the stamp records the ACTUAL model and marks the
        source as "fallback" — never attributing the turn to a layer that
        routed a different model."""
        from backend.copilot.sdk.service import _stamp_turn_messages

        msgs = self._messages()
        _stamp_turn_messages(
            msgs,
            start_index=0,
            requested_model="moonshotai/kimi-k2.6",
            actual_model="claude-sonnet-4-6",
            routing_source="ld",
        )
        assert msgs[0].model == "claude-sonnet-4-6"
        assert msgs[0].routing_source == "fallback"

    def test_matching_observed_model_keeps_source(self):
        from backend.copilot.sdk.service import _stamp_turn_messages

        msgs = self._messages()
        _stamp_turn_messages(
            msgs,
            start_index=0,
            requested_model="claude-sonnet-4-6",
            actual_model="claude-sonnet-4-6",
            routing_source="ld",
        )
        assert msgs[0].routing_source == "ld"

    def test_subscription_default_never_marks_fallback(self):
        """Subscription mode requests NO model (requested_model=None) — the
        CLI's default choice is the resolution itself, not a divergence."""
        from backend.copilot.sdk.service import _stamp_turn_messages

        msgs = self._messages()
        _stamp_turn_messages(
            msgs,
            start_index=0,
            requested_model=None,
            actual_model="claude-opus-4-6",
            routing_source="env",
        )
        assert msgs[0].model == "claude-opus-4-6"
        assert msgs[0].routing_source == "env"  # NOT "fallback"

    def test_flushed_rows_flagged_for_stamp_backfill(self):
        """A row the mid-turn flush already persisted (has a sequence) gets
        flagged so the save path back-fills its columns — the insert path
        only covers unsequenced rows."""
        from backend.copilot.sdk.service import _stamp_turn_messages

        msgs = self._messages()
        msgs[2].sequence = 7  # flushed mid-turn
        _stamp_turn_messages(
            msgs,
            start_index=1,
            requested_model="claude-sonnet-4-6",
            actual_model=None,
            routing_source="env",
        )
        assert msgs[2].model == "claude-sonnet-4-6"
        assert msgs[2].stamps_pending_save is True
        assert msgs[1].stamps_pending_save is False  # user row untouched

    def test_turn_boundary_captured_after_message_cleanup(self):
        """The stamp's ``start_index`` (``pre_turn_message_count``) MUST be
        taken after the turn-start cleanup that pops trailing error markers
        and prunes orphan tool rows. Captured before, it overshoots by the
        number of popped rows and leaves this turn's first assistant row(s)
        below ``start_index`` — unstamped — precisely on error-recovery
        turns. Guard the ordering at the source level: the enormous turn
        handler can't be driven in a unit test, so pin the invariant that
        the count assignment follows the last cleanup call."""
        import inspect

        from backend.copilot.sdk.service import stream_chat_completion_sdk

        src = inspect.getsource(stream_chat_completion_sdk)
        assign_at = src.index("pre_turn_message_count = len(session.messages)")
        prune_at = src.index("prune_orphan_tool_calls(session.messages")
        marker_pop_at = src.index("Removing stale error marker")
        assert (
            assign_at > prune_at
        ), "pre_turn_message_count must be captured AFTER prune_orphan_tool_calls"
        assert (
            assign_at > marker_pop_at
        ), "pre_turn_message_count must be captured AFTER error-marker cleanup"


class TestStampTurnTraceId:
    """A thumbs up/down on a reply is scored against the trace recorded on
    it, so the stamp must name this turn's trace on this turn's replies."""

    TRACE_ID = "1edf31f11b1693cc6103f358c1481694"

    def _messages(self):
        from backend.copilot.model import ChatMessage

        return [
            ChatMessage(role="assistant", content="old", sequence=0),
            ChatMessage(role="user", content="q", sequence=1),
            ChatMessage(role="assistant", content="working", sequence=2),
            ChatMessage(role="tool", content="{}", tool_call_id="t1"),
            ChatMessage(role="assistant", content="done"),
            # The UI can name the reply bubble after a trailing reasoning row.
            ChatMessage(role="reasoning", content="checking the totals"),
        ]

    def test_stamps_this_turns_replies_only(self):
        from backend.copilot.sdk.service import _stamp_turn_trace_id

        msgs = self._messages()
        _stamp_turn_trace_id(msgs, start_index=1, trace_id=self.TRACE_ID)

        assert msgs[0].langfuse_trace_id is None  # an earlier turn's reply
        assert msgs[1].langfuse_trace_id is None  # user row
        assert msgs[2].langfuse_trace_id == self.TRACE_ID
        assert msgs[3].langfuse_trace_id is None  # tool row
        assert msgs[4].langfuse_trace_id == self.TRACE_ID
        assert msgs[5].langfuse_trace_id == self.TRACE_ID  # reasoning row

    def test_flushed_rows_flagged_for_backfill(self):
        from backend.copilot.sdk.service import _stamp_turn_trace_id

        msgs = self._messages()
        _stamp_turn_trace_id(msgs, start_index=1, trace_id=self.TRACE_ID)

        assert msgs[2].stamps_pending_save is True  # flushed mid-turn
        assert msgs[4].stamps_pending_save is False  # inserted by this save

    def test_never_overwrites_an_existing_trace(self):
        from backend.copilot.sdk.service import _stamp_turn_trace_id

        msgs = self._messages()
        msgs[2].langfuse_trace_id = "a" * 32
        _stamp_turn_trace_id(msgs, start_index=0, trace_id=self.TRACE_ID)

        assert msgs[2].langfuse_trace_id == "a" * 32
        assert msgs[2].stamps_pending_save is False

    def test_no_trace_stamps_nothing(self):
        """Langfuse off, or its span failed to open: nothing to record."""
        from backend.copilot.sdk.service import _stamp_turn_trace_id

        msgs = self._messages()
        _stamp_turn_trace_id(msgs, start_index=0, trace_id=None)

        assert all(m.langfuse_trace_id is None for m in msgs)
        assert not any(m.stamps_pending_save for m in msgs)

    def test_stamped_in_the_finally_after_the_trace_is_captured(self):
        """The turn handler is too large to drive in a unit test, so pin at
        the source that the stamp runs after the trace id is captured and
        before the end-of-turn save that persists it."""
        import inspect

        from backend.copilot.sdk.service import stream_chat_completion_sdk

        src = inspect.getsource(stream_chat_completion_sdk)
        captured_at = src.index(
            "langfuse_trace_id = get_client().get_current_trace_id()"
        )
        stamped_at = src.index("_stamp_turn_trace_id(")
        saved_at = src.index("await asyncio.shield(upsert_chat_session(session))")
        assert captured_at < stamped_at < saved_at


class TestSdkServingSegment:
    def test_platform_turn_is_not_read_from_mutable_session_metadata(self):
        from backend.copilot.sdk.service import _sdk_serving_segment

        segment = _sdk_serving_segment(None)

        assert segment.auth_provider == "platform"
        assert segment.credential_id is None
        assert segment.is_segment_zero is False

    def test_codex_turn_is_attributed_to_the_credential_that_served_it(self):
        from backend.copilot.sdk.service import _sdk_serving_segment

        lease = MagicMock()
        lease.credentials.id = "cred-that-served-the-turn"

        segment = _sdk_serving_segment(lease)

        assert segment.auth_provider == "codex"
        assert segment.credential_id == "cred-that-served-the-turn"
        assert segment.is_segment_zero is False

    def test_finalizer_stamps_the_captured_serving_segment(self):
        import inspect

        from backend.copilot.sdk.service import stream_chat_completion_sdk

        source = inspect.getsource(stream_chat_completion_sdk)
        flat = " ".join(source.split())

        assert "turn_segment = _sdk_serving_segment(credential_lease)" in source
        assert "pre_turn_message_count, turn_segment," in flat
        assert "session_segment(session)" not in source


class TestChatMessageStampRoundTrip:
    """model survives client-facing serialization; routing_source is
    deliberately EXCLUDED from payloads (clients could infer LD cohort
    membership from "ld" vs "env") while persisting via the explicit
    field-mapped DB path."""

    def test_model_serializes_routing_source_excluded(self):
        from backend.copilot.model import ChatMessage

        msg = ChatMessage(
            role="assistant",
            content="x",
            sequence=3,
            model="claude-sonnet-4-6",
            routing_source="catalog",
        )
        dumped = msg.model_dump()
        assert dumped["model"] == "claude-sonnet-4-6"
        assert "routing_source" not in dumped
        # The DB path reads the attribute, not the dump.
        assert msg.routing_source == "catalog"


class TestSameModelCanonicalization:
    """Spelling differences between the requested resolution and the CLI's
    report must never fake a fallback; real model changes must."""

    def test_spelling_variants_are_same_model(self):
        from backend.copilot.sdk.service import _same_model

        assert _same_model("anthropic/claude-opus-4.7", "claude-opus-4-7")
        assert _same_model("claude-haiku-4-5-20251001", "anthropic/claude-haiku-4-5")
        assert _same_model("anthropic.claude-sonnet-4-6", "claude-sonnet-4.6")

    def test_different_models_diverge(self):
        from backend.copilot.sdk.service import _same_model

        assert not _same_model("claude-sonnet-4-6", "moonshotai/kimi-k2.6")
        assert not _same_model("claude-opus-4-7", "claude-sonnet-4-6")

    def test_fallback_not_faked_by_spelling(self):
        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _stamp_turn_messages

        msgs = [ChatMessage(role="assistant", content="x", sequence=None)]
        _stamp_turn_messages(
            msgs,
            start_index=0,
            requested_model="claude-opus-4-7",
            actual_model="anthropic/claude-opus-4.7",
            routing_source="ld",
        )
        assert msgs[0].routing_source == "ld"  # same model, not "fallback"


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


def _limit(model: str = "gpt-4o") -> int:
    """The threshold ``_will_compact`` mirrors: target minus the reserve."""
    return get_compression_target(model) - DEFAULT_COMPRESSION_RESERVE


# The budget the turn would hand ``_will_compact``; the tests size their
# payloads against ``_limit`` so the two stay in step.
_TARGET = get_compression_target("gpt-4o")

_FILLER = "the quick brown fox jumps over the lazy dog "


def _band_history(
    fraction: float, model: str = "gpt-4o", rows: int = 8
) -> list[ChatMessage]:
    """A history whose token estimate is *fraction* of the threshold.

    Sized with the real tokenizer so the payload lands inside the ambiguous
    band, which is where ``_will_compact`` actually tokenizes — and the only
    region where its threshold can disagree with the compressor's.  Keep
    *fraction* within roughly 0.5-1.2: outside that the character prefilter
    answers on its own and the tokenizer never runs.
    """
    probe = 100
    unit = (
        estimate_token_count(
            [{"role": "user", "content": _FILLER * probe}], model=model
        )
        / probe
    )
    per_row = max(1, int(_limit(model) * fraction) // rows)
    body = _FILLER * max(1, round(per_row / unit))
    return [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=body)
        for i in range(rows)
    ]


class TestWillCompact:
    def test_false_for_short_history(self):
        assert (
            _will_compact(
                [_msg("user", "hi")], "gpt-4o", target_tokens=_TARGET
            ).expected
            is False
        )

    def test_false_for_empty_history(self):
        assert _will_compact([], "gpt-4o", target_tokens=_TARGET).expected is False

    def test_true_when_history_exceeds_the_compression_target(self):
        # get_compression_target for gpt-4o is well under 1M tokens; a
        # megabyte of prose comfortably clears it.
        big = [_msg("user", "word " * 200_000), _msg("assistant", "ok")]
        forecast = _will_compact(big, "gpt-4o", target_tokens=_TARGET)
        assert forecast.expected is True
        # The size rides along so the client can pace its progress curve
        # against the real size of the work instead of a constant floor.
        assert forecast.tokens_before is not None
        assert forecast.tokens_before > 100_000

    def test_ignores_reasoning_and_compaction_rows(self):
        rows = [
            _msg("reasoning", "word " * 200_000),
            _msg("user", "hi"),
        ]
        assert _will_compact(rows, "gpt-4o", target_tokens=_TARGET).expected is False

    def test_counts_tool_call_arguments(self):
        """Tool-call payloads must be counted, not silently dropped.

        Copilot assistant rows carry whole graph JSON in ``create_agent`` /
        ``edit_agent`` arguments.  An estimate built from role+content alone
        reads these rows as nearly empty, so the sessions that need the
        compaction row most are exactly the ones that never open one.
        """
        graph_json = json.dumps({"nodes": ["n" * 40] * 40_000})
        rows = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "create_agent",
                            "arguments": graph_json,
                        },
                    }
                ],
            ),
            _msg("user", "ship it"),
        ]
        assert _will_compact(rows, "gpt-4o", target_tokens=_TARGET).expected is True

    def test_tokenizer_failure_degrades_to_a_false_negative(self):
        """The prediction is cosmetic; nothing it does may kill the stream.

        tiktoken downloads its BPE table on first use and models come and go,
        so this call can fail for reasons that have nothing to do with the
        conversation.  Giving up returns False — a false negative, which
        ``emit_pre_query_end`` covers with a self-contained row.
        """
        over_threshold = _band_history(1.1)
        assert (
            _will_compact(over_threshold, "gpt-4o", target_tokens=_TARGET).expected
            is True
        )
        with patch(
            "backend.copilot.sdk.service.estimate_token_count",
            side_effect=RuntimeError("tiktoken download failed"),
        ):
            assert (
                _will_compact(over_threshold, "gpt-4o", target_tokens=_TARGET).expected
                is False
            )

    def test_huge_history_skips_the_tokenizer_entirely(self):
        """The expensive path must not run where it costs the most.

        Above ``limit * _MAX_CHARS_PER_TOKEN`` no history can be under the
        threshold whatever it contains, so the answer is arithmetic — which
        is what keeps the tokenizer's input bounded.
        """
        huge = [_msg("user", "word " * 400_000), _msg("assistant", "ok")]
        with patch(
            "backend.copilot.sdk.service.estimate_token_count",
            side_effect=AssertionError("tokenizer must not run on a huge history"),
        ):
            forecast = _will_compact(huge, "gpt-4o", target_tokens=_TARGET)
        assert forecast.expected is True
        # No estimate was taken, but the client still needs something to pace
        # against, so the character count stands in for one.
        assert forecast.tokens_before is not None

    def test_tiny_history_skips_the_tokenizer_entirely(self):
        small = [_msg("user", "hi there"), _msg("assistant", "hello")]
        with patch(
            "backend.copilot.sdk.service.estimate_token_count",
            side_effect=AssertionError("tokenizer must not run on a tiny history"),
        ):
            assert (
                _will_compact(small, "gpt-4o", target_tokens=_TARGET).expected is False
            )


def _seq_msg(role: str, content: str, sequence: int) -> ChatMessage:
    return ChatMessage(role=role, content=content, sequence=sequence)


def _big_history(n: int = 10) -> list[ChatMessage]:
    """Alternating user/assistant rows whose total comfortably exceeds the
    gpt-4o compression target (see TestWillCompact)."""
    return [
        _seq_msg("user" if i % 2 == 0 else "assistant", "word " * 30_000, i)
        for i in range(n)
    ]


class TestExpectPreQueryCompaction:
    def test_steady_state_resume_never_predicts(self):
        # The transcript covers the full history — _build_query_message's
        # Scenario A compresses nothing, so the pre-check must stay False
        # even though the cumulative history exceeds the target.
        messages = _big_history()
        assert _will_compact(messages, "gpt-4o", target_tokens=_TARGET).expected is True
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=len(messages) - 1,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is False

    def test_no_resume_predicts_over_full_history(self):
        messages = _big_history()
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=False,
                transcript_msg_count=0,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is True

    def test_no_resume_predicts_over_prior_messages_when_given(self):
        """The no-resume branch compresses ``prior_messages`` when it is set.

        ``prior_messages`` is the compacted transcript + hole + gap — normally
        far smaller than the raw session and non-None on essentially every
        no-resume turn with a baseline transcript.  Predicting over the full
        session there opens a row for compression that never runs.
        """
        messages = _big_history()
        small_prior = [_msg("user", "hi"), _msg("assistant", "hello")]
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=False,
                transcript_msg_count=0,
                session_msg_ceiling=len(messages),
                prior_messages=small_prior,
                target_tokens=_TARGET,
            )
        ).expected is False

    def test_no_resume_single_message_never_predicts(self):
        messages = [_msg("user", "word " * 200_000)]
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=False,
                transcript_msg_count=0,
                session_msg_ceiling=1,
                target_tokens=_TARGET,
            )
        ).expected is False

    def test_resume_with_zero_watermark_never_predicts(self):
        messages = _big_history()
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=0,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is False

    def test_resume_with_a_big_stale_gap_predicts(self):
        # Watermark covers rows 0-1 (last covered row is the assistant at
        # index 1); the remaining gap is large enough to compress.
        messages = _big_history()
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=2,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is True

    def test_resume_with_a_small_gap_never_predicts(self):
        messages = [
            _seq_msg("user", "word " * 30_000, 0),
            _seq_msg("assistant", "word " * 30_000, 1),
            _seq_msg("user", "hi", 2),
            _seq_msg("assistant", "ok", 3),
            _seq_msg("user", "thanks", 4),
        ]
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=2,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is False

    def test_resume_watermark_after_a_user_row_predicts_from_the_rest(self):
        # The watermark is a sequence: rows 1.. are uncovered whatever role
        # row 0 has, and _build_query_message compresses them.
        messages = _big_history()
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=1,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is True

    def test_resume_gap_error_markers_never_predict(self):
        # Markers are never sent, so only the small row is left to compress;
        # counted, the two large markers would predict a compaction.
        big_marker = f"{COPILOT_ERROR_PREFIX} " + "word " * 100_000
        messages = [
            _seq_msg("user", "hi", 0),
            _seq_msg("assistant", "ok", 1),
            _seq_msg("user", "small question", 2),
            _seq_msg("assistant", big_marker, 3),
            _seq_msg("assistant", big_marker, 4),
            _seq_msg("user", "again", 5),
        ]
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=2,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is False

    def test_resume_forecast_measures_the_gap_against_the_turn_budget(self):
        # The same uncovered rows predict a compaction under a tight budget and
        # none under a generous one: the resume branch reads the target the
        # turn hands it, the pinned window's, not a constant of its own.
        messages = _big_history()

        def forecast(target_tokens: int) -> bool:
            return _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=1,
                session_msg_ceiling=len(messages),
                target_tokens=target_tokens,
            ).expected

        assert forecast(_TARGET) is True
        assert forecast(10_000_000) is False

    def test_cap_engaged_window_predicts_from_post_watermark_rows(self):
        # The windowed view starts above absolute sequence 0, so the
        # sequence-based gap path runs; rows at/after the watermark form
        # the compressible slice.
        messages = [
            _seq_msg("user" if i % 2 == 0 else "assistant", "word " * 30_000, 100 + i)
            for i in range(10)
        ]
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=102,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is True

    def test_cap_engaged_with_no_post_watermark_rows_never_predicts(self):
        messages = [
            _seq_msg("user" if i % 2 == 0 else "assistant", "word " * 30_000, 100 + i)
            for i in range(10)
        ]
        assert (
            _expect_pre_query_compaction(
                messages,
                "gpt-4o",
                use_resume=True,
                transcript_msg_count=200,
                session_msg_ceiling=len(messages),
                target_tokens=_TARGET,
            )
        ).expected is False


# ---------------------------------------------------------------------------
# Predictor vs. compressor — the two must never disagree
# ---------------------------------------------------------------------------


def _make_session(messages: list[ChatMessage]) -> ChatSession:
    now = datetime.now(UTC)
    return ChatSession(
        session_id="differential-session",
        user_id="user-1",
        messages=messages,
        title="t",
        usage=[],
        started_at=now,
        updated_at=now,
    )


_DIFFERENTIAL_CASES: list[dict] = [
    # name, use_resume, transcript_msg_count, messages, prior_messages
    {
        "name": "no-resume, full db history",
        "use_resume": False,
        "transcript_msg_count": 0,
        "messages": _big_history(),
        "prior_messages": None,
    },
    {
        "name": "no-resume, small transcript+gap prior",
        "use_resume": False,
        "transcript_msg_count": 4,
        "messages": _big_history(),
        "prior_messages": [_msg("user", "hi"), _msg("assistant", "hello")],
    },
    {
        "name": "no-resume, large transcript+gap prior",
        "use_resume": False,
        "transcript_msg_count": 4,
        "messages": _big_history(),
        "prior_messages": _big_history(6),
    },
    {
        "name": "no-resume, single message",
        "use_resume": False,
        "transcript_msg_count": 0,
        "messages": [_msg("user", "word " * 200_000)],
        "prior_messages": None,
    },
    {
        "name": "resume, watermark aligned with a big gap",
        "use_resume": True,
        "transcript_msg_count": 2,
        "messages": _big_history(),
        "prior_messages": None,
    },
    {
        "name": "resume, watermark right after a user row",
        "use_resume": True,
        "transcript_msg_count": 1,
        "messages": _big_history(),
        "prior_messages": None,
    },
    {
        "name": "resume, big gap with reasoning rows and an error marker",
        "use_resume": True,
        "transcript_msg_count": 3,
        "messages": [
            _seq_msg("user", "hi", 0),
            _seq_msg("reasoning", "word " * 30_000, 1),
            _seq_msg("assistant", "ok", 2),
            _seq_msg("user", "word " * 30_000, 3),
            _seq_msg("reasoning", "word " * 30_000, 4),
            _seq_msg("assistant", "word " * 30_000, 5),
            _seq_msg("assistant", f"{COPILOT_ERROR_PREFIX} " + "word " * 30_000, 6),
            _seq_msg("user", "again", 7),
        ],
        "prior_messages": None,
    },
    {
        "name": "resume, zero watermark",
        "use_resume": True,
        "transcript_msg_count": 0,
        "messages": _big_history(),
        "prior_messages": None,
    },
    {
        "name": "resume, full coverage",
        "use_resume": True,
        "transcript_msg_count": 9,
        "messages": _big_history(),
        "prior_messages": None,
    },
    {
        "name": "resume, cap engaged with post-watermark rows",
        "use_resume": True,
        "transcript_msg_count": 102,
        "messages": [
            _seq_msg("user" if i % 2 == 0 else "assistant", "word " * 30_000, 100 + i)
            for i in range(10)
        ],
        "prior_messages": None,
    },
    {
        "name": "resume, cap engaged with no post-watermark rows",
        "use_resume": True,
        "transcript_msg_count": 200,
        "messages": [
            _seq_msg("user" if i % 2 == 0 else "assistant", "word " * 30_000, 100 + i)
            for i in range(10)
        ],
        "prior_messages": None,
    },
]


class TestPredictorMatchesCompressor:
    """``_expect_pre_query_compaction`` is a hand-written mirror of
    ``_build_query_message``'s branch tree.  Nothing but a differential test
    keeps the two honest: every divergence so far has shipped as either a bar
    that flashes and aborts, or expensive work with no row open at all.

    This pins branch SELECTION — that the predictor and the compressor look at
    the same slice.  The threshold applied to that slice is pinned separately
    by ``TestWillCompactThresholdMatchesCompressContext``, which drives the
    real ``compress_context``; measuring it here would mean re-calling
    ``_will_compact`` to grade ``_will_compact``.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "case", _DIFFERENTIAL_CASES, ids=[c["name"] for c in _DIFFERENTIAL_CASES]
    )
    async def test_predictor_agrees_with_the_slice_that_gets_compressed(
        self, case, monkeypatch
    ):
        messages = case["messages"]
        session = _make_session(messages)
        ceiling = len(messages)

        forecast = _expect_pre_query_compaction(
            messages,
            _compression_model(),
            use_resume=case["use_resume"],
            transcript_msg_count=case["transcript_msg_count"],
            session_msg_ceiling=ceiling,
            prior_messages=case["prior_messages"],
            target_tokens=_TARGET,
        )

        # Spy on the compressor: run the real pre-check against the slice
        # _build_query_message actually hands it, so the two see identical
        # inputs by construction rather than by a hand-copied estimate.
        compressed_slices: list[list[ChatMessage]] = []

        async def _spy(msgs, target_tokens=None):
            compressed_slices.append(list(msgs))
            return msgs, False, None

        monkeypatch.setattr("backend.copilot.sdk.service._compress_messages", _spy)
        await _build_query_message(
            "now what?",
            session,
            case["use_resume"],
            case["transcript_msg_count"],
            "differential-session",
            session_msg_ceiling=ceiling,
            prior_messages=case["prior_messages"],
        )

        would_compact = any(
            _will_compact(slice_, _compression_model(), target_tokens=_TARGET).expected
            for slice_ in compressed_slices
        )

        # Both directions are enforced.  A false positive flashes a bar that
        # aborts; a false negative is survivable (``emit_pre_query_end`` still
        # emits a self-contained row) but leaves the user staring at a frozen
        # screen for the whole summarization, which is the bug this PR exists
        # to fix — so neither is allowed to creep back in through the branch
        # tree.  Cases whose slice the estimator genuinely cannot see (the
        # cap-engaged hole rows fetched from the DB) are not in the matrix.
        assert forecast.expected == would_compact, (
            f"{case['name']}: predicted {forecast.expected} but "
            f"_build_query_message compressed {len(compressed_slices)} slice(s) "
            f"of which would_compact={would_compact}"
        )


class TestWillCompactThresholdMatchesCompressContext:
    """``_will_compact``'s threshold is a copy of ``compress_context``'s early
    return (``original_count + reserve <= target_tokens``).  Grading it against
    itself proves nothing, so these cases drive the real compressor with real
    payloads and read ``was_compacted`` back.

    ``client=None`` keeps the LLM out of it: the summarization step is skipped
    and the remaining strategies still only run when the early return doesn't
    fire, which is exactly the boundary under test.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fraction", "expected"),
        [(0.5, False), (0.9, False), (1.1, True), (1.6, True)],
        ids=["half", "just-under", "just-over", "well-over"],
    )
    async def test_prediction_matches_the_compressor(self, fraction, expected):
        rows = _band_history(fraction)
        payload = [{"role": m.role, "content": m.content} for m in rows]

        result = await compress_context(payload, model="gpt-4o", client=None)

        assert result.was_compacted is expected
        assert (
            _will_compact(rows, "gpt-4o", target_tokens=_TARGET).expected
            is result.was_compacted
        )

    @pytest.mark.asyncio
    async def test_threshold_reads_the_compressors_own_constants(self):
        """A payload sized off ``get_compression_target`` and
        ``DEFAULT_COMPRESSION_RESERVE`` must land on the compressor's own
        boundary — if either constant is read differently on the two sides,
        one of these two cases flips.
        """
        under = [{"role": "user", "content": _FILLER}] * 2
        assert (
            await compress_context(under, model="gpt-4o", client=None)
        ).was_compacted is False

        at_limit = _band_history(1.0)
        payload = [{"role": m.role, "content": m.content} for m in at_limit]
        estimated = estimate_token_count(payload, model="gpt-4o")
        result = await compress_context(payload, model="gpt-4o", client=None)
        # ``compress_context`` sums per-message, the estimator scales the sum,
        # so the two differ by at most a rounding step per message.
        assert abs(result.original_token_count - estimated) <= len(payload)
        assert result.was_compacted is (
            result.original_token_count + DEFAULT_COMPRESSION_RESERVE
            > get_compression_target("gpt-4o")
        )


# ---------------------------------------------------------------------------
# _retry_reduced_context
# ---------------------------------------------------------------------------


def _summarized() -> ReducedContext:
    """``_reduce_context``'s success exit: the transcript came back compacted."""
    return ReducedContext(TranscriptBuilder(), False)


def _dropped() -> ReducedContext:
    """``_reduce_context``'s fallback exit: the transcript was discarded.

    Reached both when ``compact_transcript`` fails on the first retry and on
    every later retry.
    """
    return ReducedContext(TranscriptBuilder(), True, 50_000)


class TestRetryReducedContext:
    """The retry row must only persist when the conversation was summarized.

    ``_reduce_context`` is called after the row is already open, so the
    close is a claim about work that may not have happened.
    """

    def test_summarized_transcript_counts_even_without_compressed_history(self):
        # First retry, compaction succeeded — a real summarization the row
        # can honestly report.
        assert (
            _retry_reduced_context(reduced=_summarized(), compaction_stats=None) is True
        )

    def test_compressed_history_counts_without_a_transcript(self):
        assert (
            _retry_reduced_context(
                reduced=_dropped(),
                compaction_stats=CompactionStats(tokens_before=100, tokens_after=10),
            )
            is True
        )

    def test_nothing_reduced_retires_the_row(self):
        # A later retry whose transcript was already dropped and whose
        # history already fits reduces nothing.
        assert (
            _retry_reduced_context(reduced=_dropped(), compaction_stats=None) is False
        )

    @pytest.mark.asyncio
    async def test_failed_compaction_is_not_a_summarization(self):
        """A live transcript that FAILS to compact takes the drop branch.

        The row is durable, so honouring it here tells the user — permanently,
        on every refresh — that a conversation which was truncated had been
        summarized.  Driven through the real ``_reduce_context`` because the
        distinction only exists in its return value: the inputs (a live
        transcript, first retry) are identical to the success case.
        """
        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])

        with patch(
            "backend.copilot.sdk.service.compact_transcript",
            new_callable=AsyncMock,
            return_value=None,
        ):
            ctx = await _reduce_context(transcript, "[test]", profile=_profile())

        assert ctx.transcript_lost is True
        assert _retry_reduced_context(reduced=ctx, compaction_stats=None) is False

    def test_both_signals_still_reports_a_reduction(self):
        assert (
            _retry_reduced_context(
                reduced=_summarized(),
                compaction_stats=CompactionStats(messages_before=9, messages_after=2),
            )
            is True
        )


# ---------------------------------------------------------------------------
# The compressor's model is the predictor's model
# ---------------------------------------------------------------------------


class TestCompressionModelIsShared:
    """The predictor sizes an estimate against a context window; the
    compressor compresses against one.  When those were two different models
    — the runtime model (Codex, 400K) versus the compressor's default
    (Sonnet, 200K) — every turn in the band between them either compacted
    with no row open or opened a row that instantly aborted.
    """

    @pytest.mark.asyncio
    async def test_compress_messages_uses_the_shared_model(self):
        from backend.util.prompt import CompressResult

        captured: dict[str, str] = {}

        async def _fake_run(messages, model, log_prefix, target_tokens=None, **_):
            captured["model"] = model
            return CompressResult(
                messages=list(messages),
                token_count=1,
                was_compacted=False,
                original_token_count=1,
            )

        with patch("backend.copilot.sdk.service._run_compression", new=_fake_run):
            await _compress_messages([_msg("user", "q"), _msg("assistant", "a")])

        assert captured["model"] == _compression_model()


# ---------------------------------------------------------------------------
# _seed_transcript
# ---------------------------------------------------------------------------


class TestSeedTranscript:
    @pytest.mark.asyncio
    async def test_seed_excludes_the_current_turn_and_the_compaction_row(
        self, monkeypatch
    ):
        """The pre-query row is persisted BEFORE seeding runs.

        Slicing off only the last entry then leaves the current user message
        inside the seed; the turn appends it again via ``append_user``, so the
        next turn's ``--resume`` context shows the live message twice and the
        watermark is two rows ahead of reality.
        """
        from backend.copilot.transcript_builder import TranscriptBuilder

        from .compaction import _persist
        from .service import _seed_transcript

        session = _make_session(
            [
                _msg("user", "q1"),
                _msg("assistant", "a1"),
                _msg("user", "current turn"),
            ]
        )
        ceiling = len(session.messages)
        _persist(session, "compaction-abc", '{"summary": "x"}')
        assert len(session.messages) == ceiling + 2

        seen: list[list[ChatMessage]] = []

        async def _passthrough(msgs, target_tokens=None):
            seen.append(list(msgs))
            return list(msgs), False, None

        monkeypatch.setattr(
            "backend.copilot.sdk.service._compress_messages", _passthrough
        )

        content, count = await _seed_transcript(
            session,
            TranscriptBuilder(),
            0,
            "[test]",
            ceiling,
            seed_target=30_000,
        )

        assert [(m.role, m.content) for m in seen[0]] == [
            ("user", "q1"),
            ("assistant", "a1"),
        ]
        assert "current turn" not in content
        assert count == 2


# ---------------------------------------------------------------------------
# A failed compression drops history — and says so
# ---------------------------------------------------------------------------


class TestCompressionFailureIsReportedAsADrop:
    @pytest.mark.asyncio
    async def test_failure_returns_dropped_stats(self):
        """Both compression tiers failing drops the history to the bare
        message.  That is a real, user-visible loss, so the stats say
        ``dropped`` — enough for the row to close honestly — while claiming
        none of the counts a summarize would have produced."""
        rows = [_msg("user", "q"), _msg("assistant", "a"), _msg("user", "q2")]
        with patch(
            "backend.copilot.sdk.service._run_compression",
            new_callable=AsyncMock,
            side_effect=RuntimeError("summarizer down"),
        ):
            compressed, reduced, stats = await _compress_messages(rows)
        assert compressed == []
        assert reduced is True
        assert stats is not None
        assert stats.dropped is True
        assert stats.messages_before == 3
        assert stats.tokens_after is None


class TestContextLimitWithoutCompaction:
    def test_first_attempt_with_nothing_attempted(self):
        alarm = _context_limit_without_compaction(0, CompactionTracker())
        assert alarm is not None
        assert "before any compaction was attempted" in alarm

    @pytest.mark.asyncio
    async def test_first_attempt_after_a_cycle_that_did_not_land(self):
        tracker = CompactionTracker()
        tracker.on_compact("/tmp/session.jsonl")
        await tracker.emit_end_if_ready(
            ChatSession.new(user_id="test-user", dry_run=False),
            CompactionStats(tokens_before=191_536, messages_before=92),
            after_source="no_summary_line",
        )
        alarm = _context_limit_without_compaction(0, tracker)
        assert alarm is not None
        assert "1 compaction attempt(s) and none landed" in alarm
        assert "sdk_internal" in alarm

    @pytest.mark.asyncio
    async def test_quiet_once_a_compaction_landed(self):
        tracker = CompactionTracker()
        tracker.on_compact("/tmp/session.jsonl")
        await tracker.emit_end_if_ready(
            ChatSession.new(user_id="test-user", dry_run=False),
            CompactionStats(
                tokens_before=191_536,
                tokens_after=31_000,
                messages_before=92,
                messages_after=12,
            ),
            after_source="read",
        )
        assert _context_limit_without_compaction(0, tracker) is None

    def test_quiet_on_our_own_retries(self):
        assert _context_limit_without_compaction(1, CompactionTracker()) is None


class TestMeasureSdkCompactionWaitsForTheSummary:
    """Measured against CLI 2.1.274: the summary lands ~30 s after PreCompact."""

    @staticmethod
    def _ctx_state():
        tracker = CompactionTracker()
        tracker.on_compact("/projects/s/s.jsonl")
        ctx = MagicMock()
        ctx.compaction = tracker
        state = MagicMock()
        state.transcript_builder.entries_as_dicts.return_value = []
        return tracker, ctx, state

    @pytest.mark.asyncio
    async def test_no_summary_yet_keeps_the_cycle_open_until_final(self):
        from backend.copilot.sdk import service as svc

        _, ctx, state = self._ctx_state()
        with (
            patch.object(
                svc,
                "read_compacted_entries_detailed",
                return_value=(None, "no_summary_line"),
            ),
            patch.object(svc, "sdk_compaction_stats", return_value=None),
        ):
            assert await svc._measure_sdk_compaction(ctx, state, final=False) == (
                False,
                None,
                None,
                None,
            )
            measured, compacted, _, after = await svc._measure_sdk_compaction(
                ctx, state, final=True
            )
        assert measured is True and compacted is None
        assert after == "no_summary_line"

    @pytest.mark.asyncio
    async def test_previous_cycles_summary_is_stale_not_this_cycles(self):
        from backend.copilot.sdk import service as svc

        tracker, ctx, state = self._ctx_state()
        old = [{"uuid": "cs1", "isCompactSummary": True}]
        tracker.note_summary(old)
        new = [{"uuid": "cs2", "isCompactSummary": True}]
        with patch.object(svc, "sdk_compaction_stats", return_value=None):
            with patch.object(
                svc, "read_compacted_entries_detailed", return_value=(old, "read")
            ):
                assert await svc._measure_sdk_compaction(ctx, state, final=False) == (
                    False,
                    None,
                    None,
                    None,
                )
            with patch.object(
                svc, "read_compacted_entries_detailed", return_value=(new, "read")
            ):
                measured, compacted, _, after = await svc._measure_sdk_compaction(
                    ctx, state, final=False
                )
        assert measured is True and compacted == new and after == "read"
        assert tracker.resolve_after_source(new, "read") == "stale_summary"
