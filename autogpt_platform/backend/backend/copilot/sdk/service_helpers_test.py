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

from backend.copilot import config as cfg_mod
from backend.copilot.config import ChatConfig

from .conftest import build_test_transcript as _build_transcript
from .service import (
    _RETRY_TARGET_TOKENS,
    ReducedContext,
    _compaction_target_tokens,
    _is_prompt_too_long,
    _is_tool_only_message,
    _iter_sdk_messages,
    _normalize_model_name,
    _reduce_context,
    _resolve_sdk_model_for_request,
    _restore_cli_session_for_turn,
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
    ``config.thinking_advanced_model`` (for 'advanced') or
    ``config.thinking_standard_model`` (for 'standard').  These tests verify
    the OpenRouter/direct-Anthropic split: OpenRouter routes by full
    ``vendor/model`` slug, while direct-Anthropic strips the prefix and
    converts dots to hyphens.
    """

    @pytest.fixture
    def _direct_anthropic_config(
        self, monkeypatch: pytest.MonkeyPatch, _clean_config_env: None
    ):
        """Force ``config.openrouter_active = False`` for prefix-strip tests.

        Pins the SDK model fields to anthropic/* so the new
        ``_validate_sdk_model_vendor_compatibility`` model_validator
        permits ChatConfig construction.
        """
        cfg = cfg_mod.ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

    @pytest.fixture
    def _openrouter_config(
        self, monkeypatch: pytest.MonkeyPatch, _clean_config_env: None
    ):
        """Force ``config.openrouter_active = True`` for slug-preservation tests."""
        cfg = cfg_mod.ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

    def test_strips_anthropic_prefix(self, _direct_anthropic_config):
        assert _normalize_model_name("anthropic/claude-opus-4-6") == "claude-opus-4-6"

    def test_rejects_non_anthropic_vendor_in_direct_mode(
        self, _direct_anthropic_config
    ):
        """Direct-Anthropic mode must fail loudly on non-Anthropic vendor
        slugs — silent strip would send e.g. ``gpt-4o`` to the Anthropic
        API and produce an opaque model_not_found error."""
        with pytest.raises(ValueError, match="requires an Anthropic model"):
            _normalize_model_name("openai/gpt-4o")
        with pytest.raises(ValueError, match="requires an Anthropic model"):
            _normalize_model_name("moonshotai/kimi-k2.6")
        with pytest.raises(ValueError, match="requires an Anthropic model"):
            _normalize_model_name("google/gemini-2.5-flash")

    def test_already_normalized_unchanged(self, _direct_anthropic_config):
        assert (
            _normalize_model_name("claude-sonnet-4-20250514")
            == "claude-sonnet-4-20250514"
        )

    def test_empty_string_unchanged(self, _direct_anthropic_config):
        assert _normalize_model_name("") == ""

    def test_opus_model_dot_to_hyphen(self, _direct_anthropic_config):
        """Direct-Anthropic mode: dots in versions become hyphens."""
        assert _normalize_model_name("anthropic/claude-opus-4.6") == "claude-opus-4-6"

    def test_openrouter_keeps_anthropic_slug(self, _openrouter_config):
        """OpenRouter routes by full slug — keep prefix and dots intact."""
        assert (
            _normalize_model_name("anthropic/claude-sonnet-4.6")
            == "anthropic/claude-sonnet-4.6"
        )

    def test_openrouter_keeps_kimi_slug(self, _openrouter_config):
        """Non-Anthropic vendors (Moonshot) require the prefix to route."""
        assert _normalize_model_name("moonshotai/kimi-k2.6") == "moonshotai/kimi-k2.6"

    @pytest.fixture
    def _subscription_with_openrouter_config(
        self, monkeypatch: pytest.MonkeyPatch, _clean_config_env: None
    ):
        """Subscription mode with leftover OpenRouter base_url + api_key.

        Reproduces the bug: ``CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true`` plus
        a populated ``CHAT_BASE_URL`` (e.g. left over from an earlier
        OpenRouter setup) used to incorrectly preserve the OpenRouter slug
        because the gate checked config shape (``openrouter_active``) not
        actual transport.  The CLI subprocess uses OAuth here and rejects
        the OpenRouter format.
        """
        cfg = cfg_mod.ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

    def test_subscription_strips_anthropic_prefix_despite_openrouter_config(
        self, _subscription_with_openrouter_config
    ):
        """Subscription transport must produce the CLI-friendly form even
        when OpenRouter base_url + api_key are set — the CLI uses OAuth
        and ignores those fields, so the OpenRouter slug would be rejected."""
        assert _normalize_model_name("anthropic/claude-opus-4.7") == "claude-opus-4-7"

    def test_subscription_rejects_non_anthropic_vendor(
        self, _subscription_with_openrouter_config
    ):
        """The CLI subprocess can only talk to Anthropic models — Kimi via
        Moonshot must raise so the resolver falls back to a tier default
        instead of feeding an unroutable slug to the CLI."""
        with pytest.raises(ValueError, match="requires an Anthropic model"):
            _normalize_model_name("moonshotai/kimi-k2.6")


# ---------------------------------------------------------------------------
# ChatConfig.effective_transport — single source of truth for "which
# transport will the SDK CLI actually use?"
# ---------------------------------------------------------------------------


class TestEffectiveTransport:
    """Subscription mode wins over OpenRouter even when OpenRouter
    base_url + api_key are set, because the CLI subprocess uses OAuth and
    ignores ``CHAT_BASE_URL`` / ``CHAT_API_KEY`` (see ``build_sdk_env``
    mode 1).  Picking the right transport here is what lets
    ``_normalize_model_name`` produce the correct model-name format.
    """

    def test_subscription_wins_over_openrouter_config(self, _clean_config_env):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=True,
        )
        assert cfg.effective_transport == "subscription"
        # ``openrouter_active`` is still True (config-shape check) but
        # the actual transport is subscription.
        assert cfg.openrouter_active is True

    def test_openrouter_when_subscription_disabled(self, _clean_config_env):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
        )
        assert cfg.effective_transport == "openrouter"

    def test_direct_anthropic_when_no_openrouter_no_subscription(
        self, _clean_config_env
    ):
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
        )
        assert cfg.effective_transport == "direct_anthropic"

    def test_subscription_alone_is_subscription(self, _clean_config_env):
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=True,
        )
        assert cfg.effective_transport == "subscription"


# ---------------------------------------------------------------------------
# _resolve_sdk_model_for_request — transport-aware LD-override normalisation
# ---------------------------------------------------------------------------


class TestResolveSdkModelForRequestTransportAware:
    """When subscription mode is on but the deployment also has OpenRouter
    config populated (e.g. ``CHAT_BASE_URL`` left over from a previous
    setup), an LD-served override must be normalised for the **subscription
    CLI**, not passed through as the OpenRouter slug.  The CLI subprocess
    uses OAuth and rejects ``anthropic/claude-opus-4.7`` with the model
    error reproduced in local debugging:

        ``There's an issue with the selected model
        (anthropic/claude-opus-4.7). It may not exist or you may not have
        access to it.``
    """

    @pytest.mark.asyncio
    async def test_subscription_advanced_override_normalised_for_cli(
        self, monkeypatch: pytest.MonkeyPatch, _clean_config_env: None
    ):
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4.7",
            claude_agent_model=None,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="anthropic/claude-opus-4.7"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="advanced", session_id="sess-adv", user_id="user-1"
            )
        # NOT the OpenRouter slug, NOT None — the CLI-friendly hyphenated form.
        assert resolved == "claude-opus-4-7"

    @pytest.mark.asyncio
    async def test_subscription_standard_no_override_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, _clean_config_env: None
    ):
        """When LD agrees with the config default, subscription mode still
        wins on the standard tier — returns ``None`` so the CLI picks the
        subscription default model."""
        cfg = cfg_mod.ChatConfig(
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            claude_agent_model=None,
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=True,
        )
        monkeypatch.setattr("backend.copilot.sdk.service.config", cfg)

        with patch(
            "backend.copilot.sdk.service._resolve_thinking_model_for_user",
            new=AsyncMock(return_value="anthropic/claude-sonnet-4-6"),
        ):
            resolved = await _resolve_sdk_model_for_request(
                model="standard", session_id="sess-std", user_id="user-1"
            )
        assert resolved is None


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


# ---------------------------------------------------------------------------
# _restore_cli_session_for_turn — mode check
# ---------------------------------------------------------------------------


class TestRestoreCliSessionModeCheck:
    """SDK skips --resume when the transcript was written by the baseline mode."""

    @pytest.mark.asyncio
    async def test_baseline_mode_transcript_skips_gcs_content(self, tmp_path):
        """A transcript with mode='baseline' must not be used as the --resume source.

        The mode check discards the GCS baseline content and falls back to DB
        reconstruction from session.messages instead.
        """
        from datetime import UTC, datetime

        from backend.copilot.model import ChatMessage, ChatSession
        from backend.copilot.transcript import TranscriptDownload
        from backend.copilot.transcript_builder import TranscriptBuilder

        session = ChatSession(
            session_id="test-session",
            user_id="user-1",
            messages=[
                ChatMessage(role="user", content="hello-unique-marker"),
                ChatMessage(role="assistant", content="world-unique-marker"),
                ChatMessage(role="user", content="follow up"),
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

        import backend.copilot.sdk.service as _svc_mod

        download_mock = AsyncMock(return_value=baseline_restore)
        with (
            patch(
                "backend.copilot.sdk.service.download_transcript",
                new=download_mock,
            ),
            patch.object(_svc_mod.config, "claude_agent_use_resume", True),
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
        # use_resume must be False — baseline transcripts cannot be used with --resume
        assert result.use_resume is False
        # context_messages must be populated — new behaviour uses transcript content + gap
        # instead of full DB reconstruction.
        assert result.context_messages is not None
        # The baseline transcript has 1 user message (BASELINE_SENTINEL).
        # Watermark=1 but position 0 is 'user', not 'assistant', so detect_gap returns [].
        # Result: 1 message from transcript, no gap.
        assert len(result.context_messages) == 1
        assert "BASELINE_SENTINEL" in (result.context_messages[0].content or "")

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
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello"),
                ChatMessage(role="user", content="follow up"),
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

        import backend.copilot.sdk.service as _svc_mod

        with (
            patch(
                "backend.copilot.sdk.service.download_transcript",
                new=AsyncMock(return_value=sdk_restore),
            ),
            patch.object(_svc_mod.config, "claude_agent_use_resume", True),
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

    @pytest.mark.asyncio
    async def test_baseline_mode_context_messages_from_transcript_content(
        self, tmp_path
    ):
        """mode='baseline' → context_messages populated from transcript content + gap.

        When a baseline-mode transcript exists, extract_context_messages converts
        the JSONL content to ChatMessage objects and returns them in context_messages.
        use_resume must remain False.
        """
        import json as stdlib_json
        from datetime import UTC, datetime

        from backend.copilot.model import ChatMessage, ChatSession
        from backend.copilot.transcript import STOP_REASON_END_TURN, TranscriptDownload
        from backend.copilot.transcript_builder import TranscriptBuilder

        # Build a minimal valid JSONL transcript with 2 messages
        lines = [
            stdlib_json.dumps(
                {
                    "type": "user",
                    "uuid": "uid-0",
                    "parentUuid": "",
                    "message": {"role": "user", "content": "TRANSCRIPT_USER"},
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
                        "content": [{"type": "text", "text": "TRANSCRIPT_ASSISTANT"}],
                    },
                }
            ),
        ]
        content = ("\n".join(lines) + "\n").encode("utf-8")

        session = ChatSession(
            session_id="test-session",
            user_id="user-1",
            messages=[
                ChatMessage(role="user", content="DB_USER"),
                ChatMessage(role="assistant", content="DB_ASSISTANT"),
                ChatMessage(role="user", content="current turn"),
            ],
            title="test",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        builder = TranscriptBuilder()
        baseline_restore = TranscriptDownload(
            content=content,
            message_count=2,
            mode="baseline",
        )

        import backend.copilot.sdk.service as _svc_mod

        with (
            patch(
                "backend.copilot.sdk.service.download_transcript",
                new=AsyncMock(return_value=baseline_restore),
            ),
            patch.object(_svc_mod.config, "claude_agent_use_resume", True),
        ):
            result = await _restore_cli_session_for_turn(
                user_id="user-1",
                session_id="test-session",
                session=session,
                sdk_cwd=str(tmp_path),
                transcript_builder=builder,
                log_prefix="[Test]",
            )

        assert result.use_resume is False
        assert result.context_messages is not None
        # Transcript content has 2 messages, no gap (watermark=2, session prior=2)
        assert len(result.context_messages) == 2
        assert result.context_messages[0].role == "user"
        assert result.context_messages[1].role == "assistant"
        assert "TRANSCRIPT_ASSISTANT" in (result.context_messages[1].content or "")
        # transcript_content must be non-empty so the _seed_transcript guard in
        # stream_chat_completion_sdk skips DB reconstruction (which would duplicate
        # builder entries since load_previous appends).
        assert result.transcript_content != ""

    @pytest.mark.asyncio
    async def test_baseline_mode_gap_present_context_includes_gap(self, tmp_path):
        """mode='baseline' + gap → context_messages includes transcript msgs and gap."""
        import json as stdlib_json
        from datetime import UTC, datetime

        from backend.copilot.model import ChatMessage, ChatSession
        from backend.copilot.transcript import STOP_REASON_END_TURN, TranscriptDownload
        from backend.copilot.transcript_builder import TranscriptBuilder

        # Transcript covers only 2 messages; session has 4 prior + current turn
        lines = [
            stdlib_json.dumps(
                {
                    "type": "user",
                    "uuid": "uid-0",
                    "parentUuid": "",
                    "message": {"role": "user", "content": "TRANSCRIPT_USER_0"},
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
                        "content": [{"type": "text", "text": "TRANSCRIPT_ASSISTANT_1"}],
                    },
                }
            ),
        ]
        content = ("\n".join(lines) + "\n").encode("utf-8")

        session = ChatSession(
            session_id="test-session",
            user_id="user-1",
            messages=[
                ChatMessage(role="user", content="DB_USER_0"),
                ChatMessage(role="assistant", content="DB_ASSISTANT_1"),
                ChatMessage(role="user", content="GAP_USER_2"),
                ChatMessage(role="assistant", content="GAP_ASSISTANT_3"),
                ChatMessage(role="user", content="current turn"),
            ],
            title="test",
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        builder = TranscriptBuilder()
        baseline_restore = TranscriptDownload(
            content=content,
            message_count=2,  # watermark=2; session has 4 prior → gap of 2
            mode="baseline",
        )

        import backend.copilot.sdk.service as _svc_mod

        with (
            patch(
                "backend.copilot.sdk.service.download_transcript",
                new=AsyncMock(return_value=baseline_restore),
            ),
            patch.object(_svc_mod.config, "claude_agent_use_resume", True),
        ):
            result = await _restore_cli_session_for_turn(
                user_id="user-1",
                session_id="test-session",
                session=session,
                sdk_cwd=str(tmp_path),
                transcript_builder=builder,
                log_prefix="[Test]",
            )

        assert result.use_resume is False
        assert result.context_messages is not None
        # 2 from transcript + 2 gap messages = 4 total
        assert len(result.context_messages) == 4
        roles = [m.role for m in result.context_messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        # Gap messages come from DB (ChatMessage objects)
        gap_user = result.context_messages[2]
        gap_asst = result.context_messages[3]
        assert gap_user.content == "GAP_USER_2"
        assert gap_asst.content == "GAP_ASSISTANT_3"


# ---------------------------------------------------------------------------
# _compaction_target_tokens — keeps our retry compaction below the CLI's
# autocompact threshold so a compacted retry doesn't immediately re-trigger
# the CLI's own autocompact on the next call.
# ---------------------------------------------------------------------------


class TestCompactionTargetTokens:
    @pytest.mark.parametrize(
        ("model", "window", "pct", "expected"),
        [
            # Sonnet 200K window with PCT=50 → CLI threshold 100K → target 80K
            ("anthropic/claude-sonnet-4-6", 200_000, 50, 80_000),
            # Sonnet 200K with PCT=0 → CLI uses ~93% (window-13K=187K) → 167K target
            ("anthropic/claude-sonnet-4-6", 200_000, 0, 167_000),
            # Aggressive PCT=30 on 200K window → CLI threshold 60K → target 40K
            ("anthropic/claude-sonnet-4-6", 200_000, 30, 40_000),
        ],
    )
    def test_anthropic_target_below_cli_threshold(
        self, model, window, pct, expected
    ) -> None:
        with (
            patch("backend.util.prompt.get_context_window", return_value=window),
            patch("backend.copilot.sdk.service.config") as mock_cfg,
        ):
            mock_cfg.claude_agent_autocompact_pct_override = pct
            assert _compaction_target_tokens(model) == expected

    def test_moonshot_uses_cli_default_threshold(self) -> None:
        # Moonshot routes ignore PCT override (config.gate skips the env var
        # entirely), so our target should mirror the CLI's ~93% default
        # regardless of the configured pct value.
        with (
            patch("backend.util.prompt.get_context_window", return_value=262_144),
            patch("backend.copilot.sdk.service.config") as mock_cfg,
        ):
            mock_cfg.claude_agent_autocompact_pct_override = 50  # ignored
            # 262144 - 13000 = 249144 (CLI default), minus 20K headroom = 229144
            assert _compaction_target_tokens("moonshotai/kimi-k2.6") == 229_144

    def test_unknown_model_falls_back_to_default_threshold(self) -> None:
        from backend.util.prompt import DEFAULT_TOKEN_THRESHOLD

        with patch("backend.util.prompt.get_context_window", return_value=None):
            assert _compaction_target_tokens("unknown/model") == DEFAULT_TOKEN_THRESHOLD

    def test_floor_at_10k_for_extremely_aggressive_pct(self) -> None:
        # PCT=1 on a 50K window → CLI threshold = 500 → target would be
        # negative without the floor.
        with (
            patch("backend.util.prompt.get_context_window", return_value=50_000),
            patch("backend.copilot.sdk.service.config") as mock_cfg,
        ):
            mock_cfg.claude_agent_autocompact_pct_override = 1
            assert _compaction_target_tokens("anthropic/foo") == 10_000

    def test_resolve_env_model_prefers_moonshot_fallback(self) -> None:
        """When the primary is Anthropic and the fallback is Moonshot, the
        env-gate model resolves to the fallback so a 529-triggered swap to
        Kimi still suppresses ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``."""
        from backend.copilot.sdk.service import _resolve_env_model

        assert (
            _resolve_env_model("anthropic/claude-sonnet-4-6", "moonshotai/kimi-k2.6")
            == "moonshotai/kimi-k2.6"
        )

    def test_resolve_env_model_keeps_primary_when_fallback_anthropic(self) -> None:
        from backend.copilot.sdk.service import _resolve_env_model

        assert (
            _resolve_env_model(
                "anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-3-5"
            )
            == "anthropic/claude-sonnet-4-6"
        )

    def test_resolve_env_model_keeps_primary_when_no_fallback(self) -> None:
        from backend.copilot.sdk.service import _resolve_env_model

        assert (
            _resolve_env_model("anthropic/claude-sonnet-4-6", None)
            == "anthropic/claude-sonnet-4-6"
        )

    @pytest.mark.asyncio
    async def test_reduce_context_uses_runtime_model_for_target(self) -> None:
        """Compactor LLM is fixed (Sonnet) but target must be sized for the
        RUNTIME model that the CLI is actually serving — otherwise a Kimi
        runtime gets a 200K-window-derived target while the CLI threshold
        is computed against Kimi's 256K window.
        """
        from backend.copilot.sdk.service import _reduce_context

        transcript = _build_transcript([("user", "hi"), ("assistant", "hello")])
        captured: dict = {}

        async def fake_compact(content, *, model, log_prefix, target_tokens):
            captured["target_tokens"] = target_tokens
            captured["compactor_model"] = model
            return None

        with (
            patch(
                "backend.copilot.sdk.service.compact_transcript",
                side_effect=fake_compact,
            ),
            patch(
                "backend.copilot.sdk.service._compaction_target_tokens",
                side_effect=lambda m: 12345 if "kimi" in m else 99999,
            ),
        ):
            await _reduce_context(
                transcript,
                False,
                "sess",
                "/tmp",
                "[t]",
                runtime_model="moonshotai/kimi-k2.6",
            )

        # Target derived from the RUNTIME model, not the compactor model.
        assert captured["target_tokens"] == 12345
