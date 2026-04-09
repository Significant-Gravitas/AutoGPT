"""Tests for P0 guardrails: _resolve_fallback_model, security env vars, TMPDIR."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.copilot.config import ChatConfig
from backend.copilot.constants import is_transient_api_error


def _make_config(**overrides) -> ChatConfig:
    """Create a ChatConfig with safe defaults, applying *overrides*."""
    defaults = {
        "use_claude_code_subscription": False,
        "use_openrouter": False,
        "api_key": None,
        "base_url": None,
    }
    defaults.update(overrides)
    return ChatConfig(**defaults)


# ---------------------------------------------------------------------------
# _resolve_fallback_model
# ---------------------------------------------------------------------------

_SVC = "backend.copilot.sdk.service"
_ENV = "backend.copilot.sdk.env"


class TestResolveFallbackModel:
    """Provider-aware fallback model resolution."""

    def test_returns_none_when_empty(self):
        cfg = _make_config(claude_agent_fallback_model="")
        with patch(f"{_SVC}.config", cfg):
            from backend.copilot.sdk.service import _resolve_fallback_model

            assert _resolve_fallback_model() is None

    def test_strips_provider_prefix(self):
        """OpenRouter-style 'anthropic/claude-sonnet-4-...' is stripped."""
        cfg = _make_config(
            claude_agent_fallback_model="anthropic/claude-sonnet-4-20250514",
            use_openrouter=True,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(f"{_SVC}.config", cfg):
            from backend.copilot.sdk.service import _resolve_fallback_model

            result = _resolve_fallback_model()

        assert result == "claude-sonnet-4-20250514"
        assert "/" not in result

    def test_dots_replaced_for_direct_anthropic(self):
        """Direct Anthropic requires hyphen-separated versions."""
        cfg = _make_config(
            claude_agent_fallback_model="claude-sonnet-4.5-20250514",
            use_openrouter=False,
        )
        with patch(f"{_SVC}.config", cfg):
            from backend.copilot.sdk.service import _resolve_fallback_model

            result = _resolve_fallback_model()

        assert result is not None
        assert "." not in result
        assert result == "claude-sonnet-4-5-20250514"

    def test_dots_preserved_for_openrouter(self):
        """OpenRouter uses dot-separated versions — don't normalise."""
        cfg = _make_config(
            claude_agent_fallback_model="claude-sonnet-4.5-20250514",
            use_openrouter=True,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(f"{_SVC}.config", cfg):
            from backend.copilot.sdk.service import _resolve_fallback_model

            result = _resolve_fallback_model()

        assert result == "claude-sonnet-4.5-20250514"

    def test_default_value(self):
        """Default fallback model resolves to a valid string."""
        cfg = _make_config()
        with patch(f"{_SVC}.config", cfg):
            from backend.copilot.sdk.service import _resolve_fallback_model

            result = _resolve_fallback_model()

        assert result is not None
        assert "sonnet" in result.lower() or "claude" in result.lower()


# ---------------------------------------------------------------------------
# Security & isolation env vars
# ---------------------------------------------------------------------------


_SECURITY_VARS = (
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
)


class TestSecurityEnvVars:
    """Verify security env vars are set in the returned dict for every auth mode.

    Tests call ``build_sdk_env()`` directly and assert the vars are present
    in the returned dict — not just present somewhere in the source file.
    """

    def test_security_vars_set_in_openrouter_mode(self):
        """Mode 3 (OpenRouter): security vars must be in the returned env."""
        cfg = _make_config(
            use_claude_code_subscription=False,
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env(session_id="s1", user_id="u1")

        for var in _SECURITY_VARS:
            assert env.get(var) == "1", f"{var} not set in OpenRouter mode"

    def test_security_vars_set_in_direct_anthropic_mode(self):
        """Mode 2 (direct Anthropic): security vars must be in the returned env."""
        cfg = _make_config(use_claude_code_subscription=False, use_openrouter=False)
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        for var in _SECURITY_VARS:
            assert env.get(var) == "1", f"{var} not set in direct Anthropic mode"

    def test_security_vars_set_in_subscription_mode(self):
        """Mode 1 (subscription): security vars must be in the returned env."""
        cfg = _make_config(use_claude_code_subscription=True)
        with (
            patch(f"{_ENV}.config", cfg),
            patch(f"{_ENV}.validate_subscription"),
        ):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env(session_id="s1", user_id="u1")

        for var in _SECURITY_VARS:
            assert env.get(var) == "1", f"{var} not set in subscription mode"

    def test_tmpdir_set_when_sdk_cwd_provided(self):
        """CLAUDE_CODE_TMPDIR must be set when sdk_cwd is provided."""
        cfg = _make_config(use_openrouter=False)
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env(sdk_cwd="/workspace/session-1")

        assert env.get("CLAUDE_CODE_TMPDIR") == "/workspace/session-1"

    def test_tmpdir_absent_when_sdk_cwd_not_provided(self):
        """CLAUDE_CODE_TMPDIR must NOT be set when sdk_cwd is None."""
        cfg = _make_config(use_openrouter=False)
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        assert "CLAUDE_CODE_TMPDIR" not in env

    def test_home_not_overridden(self):
        """HOME must NOT be overridden — would break git/ssh/npm in subprocesses."""
        cfg = _make_config(use_openrouter=False)
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        assert "HOME" not in env


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Verify ChatConfig P0 fields have correct defaults."""

    def test_fallback_model_default(self):
        cfg = _make_config()
        assert cfg.claude_agent_fallback_model
        assert "sonnet" in cfg.claude_agent_fallback_model.lower()

    def test_max_turns_default(self):
        cfg = _make_config()
        assert cfg.claude_agent_max_turns == 1000

    def test_max_budget_usd_default(self):
        cfg = _make_config()
        assert cfg.claude_agent_max_budget_usd == 100.0

    def test_max_transient_retries_default(self):
        cfg = _make_config()
        assert cfg.claude_agent_max_transient_retries == 3


# ---------------------------------------------------------------------------
# build_sdk_env — all 3 auth modes
# ---------------------------------------------------------------------------


class TestBuildSdkEnv:
    """Verify build_sdk_env returns correct dicts for each auth mode."""

    def test_subscription_mode_clears_keys(self):
        """Mode 1: subscription clears API key / auth token / base URL."""
        cfg = _make_config(use_claude_code_subscription=True)
        with (
            patch(f"{_ENV}.config", cfg),
            patch(f"{_ENV}.validate_subscription"),
        ):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env(session_id="s1", user_id="u1")

        assert env["ANTHROPIC_API_KEY"] == ""
        assert env["ANTHROPIC_AUTH_TOKEN"] == ""
        assert env["ANTHROPIC_BASE_URL"] == ""

    def test_direct_anthropic_inherits_api_key(self):
        """Mode 2: direct Anthropic doesn't set ANTHROPIC_* keys (inherits from parent)."""
        cfg = _make_config(
            use_claude_code_subscription=False,
            use_openrouter=False,
        )
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert "ANTHROPIC_BASE_URL" not in env

    def test_openrouter_sets_base_url_and_auth(self):
        """Mode 3: OpenRouter sets base URL, auth token, and clears API key."""
        cfg = _make_config(
            use_claude_code_subscription=False,
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env(session_id="sess-1", user_id="user-1")

        assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-or-test"
        assert env["ANTHROPIC_API_KEY"] == ""
        assert "x-session-id: sess-1" in env["ANTHROPIC_CUSTOM_HEADERS"]
        assert "x-user-id: user-1" in env["ANTHROPIC_CUSTOM_HEADERS"]

    def test_openrouter_no_headers_when_ids_empty(self):
        """Mode 3: No custom headers when session_id/user_id are not given."""
        cfg = _make_config(
            use_claude_code_subscription=False,
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        assert "ANTHROPIC_CUSTOM_HEADERS" not in env

    def test_openrouter_clears_oauth_tokens(self):
        """Mode 3: OAuth tokens are explicitly cleared to prevent CLI preferring subscription auth."""
        cfg = _make_config(
            use_claude_code_subscription=False,
            use_openrouter=True,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert env["CLAUDE_CODE_REFRESH_TOKEN"] == ""

    def test_direct_anthropic_clears_oauth_tokens(self):
        """Mode 2: OAuth tokens are cleared so CLI uses ANTHROPIC_API_KEY from parent env."""
        cfg = _make_config(
            use_claude_code_subscription=False,
            use_openrouter=False,
        )
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == ""
        assert env["CLAUDE_CODE_REFRESH_TOKEN"] == ""

    def test_all_modes_return_mutable_dict(self):
        """build_sdk_env must return a mutable dict (not None) in every mode."""
        for cfg in (
            _make_config(use_claude_code_subscription=True),
            _make_config(use_openrouter=False),
            _make_config(
                use_openrouter=True,
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
            ),
        ):
            with (
                patch(f"{_ENV}.config", cfg),
                patch(f"{_ENV}.validate_subscription"),
            ):
                from backend.copilot.sdk.env import build_sdk_env

                env = build_sdk_env()

            assert isinstance(env, dict)
            env["CLAUDE_CODE_TMPDIR"] = "/tmp/test"
            assert env["CLAUDE_CODE_TMPDIR"] == "/tmp/test"


# ---------------------------------------------------------------------------
# is_transient_api_error
# ---------------------------------------------------------------------------


class TestIsTransientApiError:
    """Verify that is_transient_api_error detects all transient patterns."""

    @pytest.mark.parametrize(
        "error_text",
        [
            "socket connection was closed unexpectedly",
            "ECONNRESET",
            "connection was forcibly closed",
            "network socket disconnected",
        ],
    )
    def test_connection_level_errors(self, error_text: str):
        assert is_transient_api_error(error_text)

    @pytest.mark.parametrize(
        "error_text",
        [
            "rate limit exceeded",
            "rate_limit_error",
            "Too Many Requests",
            "status code 429",
        ],
    )
    def test_429_rate_limit_errors(self, error_text: str):
        assert is_transient_api_error(error_text)

    @pytest.mark.parametrize(
        "error_text",
        [
            # Status-code-specific patterns (preferred — no false-positive risk)
            "status code 529",
            "status code 500",
            "status code 502",
            "status code 503",
            "status code 504",
        ],
    )
    def test_5xx_server_errors(self, error_text: str):
        assert is_transient_api_error(error_text)

    @pytest.mark.parametrize(
        "error_text",
        [
            "invalid_api_key",
            "Authentication failed",
            "prompt is too long",
            "model not found",
            "",
            # Natural-language phrases intentionally NOT matched — they are too
            # broad and could appear in application-level SDK messages unrelated
            # to Anthropic API transient conditions.
            "API is overloaded",
            "Internal Server Error",
            "Bad Gateway",
            "Service Unavailable",
            "Gateway Timeout",
        ],
    )
    def test_non_transient_errors(self, error_text: str):
        assert not is_transient_api_error(error_text)

    def test_case_insensitive(self):
        assert is_transient_api_error("SOCKET CONNECTION WAS CLOSED UNEXPECTEDLY")
        assert is_transient_api_error("econnreset")


# ---------------------------------------------------------------------------
# _HandledStreamError.already_yielded contract
# ---------------------------------------------------------------------------


class TestHandledStreamErrorAlreadyYielded:
    """Verify the already_yielded semantics on _HandledStreamError."""

    def test_default_already_yielded_is_true(self):
        """Non-transient callers (circuit-breaker, idle timeout) don't pass the flag —
        the default True means the outer loop won't yield a duplicate StreamError."""
        from backend.copilot.sdk.service import _HandledStreamError

        exc = _HandledStreamError("some error", code="circuit_breaker_empty_tool_calls")
        assert exc.already_yielded is True

    def test_transient_error_sets_already_yielded_false(self):
        """Transient errors pass already_yielded=False so the outer loop
        yields StreamError only once (when retries are exhausted)."""
        from backend.copilot.sdk.service import _HandledStreamError

        exc = _HandledStreamError(
            "transient",
            code="transient_api_error",
            already_yielded=False,
        )
        assert exc.already_yielded is False

    def test_backoff_capped_at_30s(self):
        """_compute_transient_backoff must be capped at _MAX_TRANSIENT_BACKOFF_SECONDS.

        With max_transient_retries=10, uncapped 2^9=512s would stall users
        for 8+ minutes.  _compute_transient_backoff caps at 30s.

        Full-jitter (0.5 … 1.0 × base) is applied for thundering-herd
        prevention, so each call returns a value in [base//2, base] rather
        than an exact integer.  We verify bounds instead of exact values.
        """
        from backend.copilot.sdk.service import (
            _MAX_TRANSIENT_BACKOFF_SECONDS,
            _compute_transient_backoff,
        )

        # attempt=1: base=1, jitter range [1, 1] (max(1, round(1 * [0.5,1.0])))
        v1 = _compute_transient_backoff(1)
        assert v1 >= 1

        # attempt=2: base=2, jitter range [1, 2]
        v2 = _compute_transient_backoff(2)
        assert 1 <= v2 <= 2

        # attempt=3: base=4, jitter range [2, 4]
        v3 = _compute_transient_backoff(3)
        assert 2 <= v3 <= 4

        # attempt=4: base=8, jitter range [4, 8]
        v4 = _compute_transient_backoff(4)
        assert 4 <= v4 <= 8

        # attempt=5: base=16, jitter range [8, 16]
        v5 = _compute_transient_backoff(5)
        assert 8 <= v5 <= 16

        # attempt=6: base capped at 30, jitter range [15, 30]
        v6 = _compute_transient_backoff(6)
        assert 15 <= v6 <= _MAX_TRANSIENT_BACKOFF_SECONDS

        # attempt=10: still capped
        v10 = _compute_transient_backoff(10)
        assert v10 <= _MAX_TRANSIENT_BACKOFF_SECONDS


# ---------------------------------------------------------------------------
# Config validators for max_turns / max_budget_usd
# ---------------------------------------------------------------------------


class TestConfigValidators:
    """Verify ge/le bounds on max_turns and max_budget_usd."""

    def test_max_turns_rejects_zero(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_turns=0)

    def test_max_turns_rejects_negative(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_turns=-1)

    def test_max_turns_rejects_above_10000(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_turns=10001)

    def test_max_turns_accepts_boundary_values(self):
        cfg_low = _make_config(claude_agent_max_turns=1)
        assert cfg_low.claude_agent_max_turns == 1
        cfg_high = _make_config(claude_agent_max_turns=10000)
        assert cfg_high.claude_agent_max_turns == 10000

    def test_max_budget_rejects_zero(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_budget_usd=0.0)

    def test_max_budget_rejects_negative(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_budget_usd=-1.0)

    def test_max_budget_rejects_above_1000(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_budget_usd=1000.01)

    def test_max_budget_accepts_boundary_values(self):
        cfg_low = _make_config(claude_agent_max_budget_usd=0.01)
        assert cfg_low.claude_agent_max_budget_usd == 0.01
        cfg_high = _make_config(claude_agent_max_budget_usd=1000.0)
        assert cfg_high.claude_agent_max_budget_usd == 1000.0

    def test_max_transient_retries_rejects_negative(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_transient_retries=-1)

    def test_max_transient_retries_rejects_above_10(self):
        with pytest.raises(ValidationError):
            _make_config(claude_agent_max_transient_retries=11)

    def test_max_transient_retries_accepts_boundary_values(self):
        cfg_low = _make_config(claude_agent_max_transient_retries=0)
        assert cfg_low.claude_agent_max_transient_retries == 0
        cfg_high = _make_config(claude_agent_max_transient_retries=10)
        assert cfg_high.claude_agent_max_transient_retries == 10


# ---------------------------------------------------------------------------
# events_yielded counter exclusions
# ---------------------------------------------------------------------------


class TestEventsYieldedExclusions:
    """Verify that ephemeral event types don't increment events_yielded.

    The events_yielded counter in stream_chat_completion_sdk controls whether
    _next_transient_backoff() permits a retry.  StreamError and StreamStatus
    must NOT be counted so that a transient notification can be followed by a
    retry without producing duplicate content for the client.

    These tests use the production _EPHEMERAL_EVENT_TYPES constant directly so
    that any drift between the constant and these assertions is caught immediately.
    """

    def test_stream_error_is_ephemeral_type(self):
        """StreamError must be an instance of the production _EPHEMERAL_EVENT_TYPES tuple.

        Uses the production constant rather than reconstructing the tuple locally
        so that any refactor that removes StreamError from the set will fail this test.
        """
        from backend.copilot.response_model import StreamError
        from backend.copilot.sdk.service import _EPHEMERAL_EVENT_TYPES

        err = StreamError(errorText="transient", code="transient_api_error")
        assert isinstance(err, _EPHEMERAL_EVENT_TYPES), (
            "StreamError must be excluded from events_yielded — "
            "if counted, transient retries would be blocked after the first notification"
        )

    def test_stream_status_is_ephemeral_type(self):
        """StreamStatus must be excluded from the production _EPHEMERAL_EVENT_TYPES tuple."""
        from backend.copilot.response_model import StreamStatus
        from backend.copilot.sdk.service import _EPHEMERAL_EVENT_TYPES

        status = StreamStatus(message="Connection interrupted, retrying in 2s…")
        assert isinstance(status, _EPHEMERAL_EVENT_TYPES), (
            "StreamStatus must be excluded from events_yielded — "
            "retrying after emitting a status notification must still be permitted"
        )


# ---------------------------------------------------------------------------
# _next_transient_backoff — module-level pure function
# ---------------------------------------------------------------------------


class TestNextTransientBackoff:
    """Unit tests for _next_transient_backoff.

    This is the core safety mechanism that prevents:
      * duplicate content (events_yielded > 0 guard)
      * infinite transient retry loops (max_transient_retries cap)
    """

    def test_events_yielded_prevents_retry(self):
        """When events_yielded > 0, return (None, unchanged_retries) — no retry."""
        from backend.copilot.sdk.service import _next_transient_backoff

        backoff, retries = _next_transient_backoff(
            events_yielded=1, transient_retries=0, max_transient_retries=3
        )
        assert backoff is None
        # Counter NOT incremented — events already sent, not a retry budget question.
        assert retries == 0

    def test_returns_backoff_on_first_retry(self):
        """First transient retry with no prior events gets a 1 s backoff (2^0)."""
        from backend.copilot.sdk.service import _next_transient_backoff

        backoff, retries = _next_transient_backoff(
            events_yielded=0, transient_retries=0, max_transient_retries=3
        )
        assert backoff == 1  # 2^(1-1) = 1 s
        assert retries == 1

    def test_increments_counter_each_retry(self):
        """Each successive call increments the retry counter and returns non-None backoff.

        Exact backoff values vary due to jitter; this test verifies the counter
        increments correctly and that a positive backoff is returned each time.
        """
        from backend.copilot.sdk.service import _next_transient_backoff

        retries = 0
        for expected_retries in [1, 2, 3]:
            backoff, retries = _next_transient_backoff(
                events_yielded=0,
                transient_retries=retries,
                max_transient_retries=5,
            )
            assert backoff is not None
            assert backoff >= 1
            assert retries == expected_retries

    def test_returns_none_when_budget_exhausted(self):
        """When transient_retries == max_transient_retries, next call returns None."""
        from backend.copilot.sdk.service import _next_transient_backoff

        backoff, retries = _next_transient_backoff(
            events_yielded=0, transient_retries=3, max_transient_retries=3
        )
        assert backoff is None
        # Counter is still incremented to reflect the attempt was made.
        assert retries == 4

    def test_events_yielded_takes_priority_over_exhaustion(self):
        """events_yielded guard fires even when retries are also exhausted."""
        from backend.copilot.sdk.service import _next_transient_backoff

        backoff, retries = _next_transient_backoff(
            events_yielded=5, transient_retries=3, max_transient_retries=3
        )
        assert backoff is None
        # Counter stays the same — events_yielded path returns early.
        assert retries == 3


# ---------------------------------------------------------------------------
# _do_transient_backoff — module-level async generator
# ---------------------------------------------------------------------------


class TestDoTransientBackoff:
    """Unit tests for _do_transient_backoff.

    The helper encapsulates the retry ceremony shared between both exception
    handlers: emit StreamStatus, sleep, reset adapter, reset usage.
    """

    async def test_yields_stream_status_with_backoff_in_message(self):
        """The helper yields exactly one StreamStatus containing the backoff duration."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from backend.copilot.response_model import StreamStatus
        from backend.copilot.sdk.service import _do_transient_backoff

        state = MagicMock()
        state.usage = MagicMock()

        events = []
        with patch("asyncio.sleep", new=AsyncMock()):
            async for evt in _do_transient_backoff(5, state, "msg-id", "sess-id"):
                events.append(evt)

        assert len(events) == 1
        assert isinstance(events[0], StreamStatus)
        assert "5s" in events[0].message

    async def test_sleeps_for_exactly_backoff_seconds(self):
        """asyncio.sleep is called with the backoff value."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from backend.copilot.sdk.service import _do_transient_backoff

        state = MagicMock()
        state.usage = MagicMock()

        mock_sleep = AsyncMock()
        with patch("asyncio.sleep", new=mock_sleep):
            async for _ in _do_transient_backoff(7, state, "msg-id", "sess-id"):
                pass

        mock_sleep.assert_called_once_with(7)

    async def test_replaces_adapter_with_new_instance(self):
        """state.adapter is replaced with a new SDKResponseAdapter after yield."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from backend.copilot.sdk.service import _do_transient_backoff

        original_adapter = MagicMock()
        state = MagicMock()
        state.adapter = original_adapter
        state.usage = MagicMock()

        with (
            patch("asyncio.sleep", new=AsyncMock()),
            patch("backend.copilot.sdk.service.SDKResponseAdapter") as mock_cls,
        ):
            new_adapter = MagicMock()
            mock_cls.return_value = new_adapter
            async for _ in _do_transient_backoff(3, state, "msg-1", "sess-1"):
                pass

        mock_cls.assert_called_once_with(message_id="msg-1", session_id="sess-1")
        assert state.adapter is new_adapter

    async def test_resets_usage_after_yield(self):
        """state.usage.reset() is called so the next attempt starts with clean counters."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from backend.copilot.sdk.service import _do_transient_backoff

        state = MagicMock()
        state.usage = MagicMock()

        with patch("asyncio.sleep", new=AsyncMock()):
            async for _ in _do_transient_backoff(2, state, "msg-id", "sess-id"):
                pass

        state.usage.reset.assert_called_once()


# ---------------------------------------------------------------------------
# _is_fallback_stderr — module-level pure function
# ---------------------------------------------------------------------------


class TestIsFallbackStderr:
    """Unit tests for _is_fallback_stderr.

    Ensures the pure function used by _on_stderr to detect fallback-model
    activation can be tested independently of the closure.
    """

    def test_true_for_fallback_model_phrase(self):
        """Lines containing 'fallback model' must return True."""
        from backend.copilot.sdk.service import _is_fallback_stderr

        assert _is_fallback_stderr("Using fallback model: claude-sonnet-4") is True

    def test_case_insensitive(self):
        """Matching must be case-insensitive."""
        from backend.copilot.sdk.service import _is_fallback_stderr

        assert _is_fallback_stderr("FALLBACK MODEL activated") is True
        assert _is_fallback_stderr("Fallback Model switching") is True

    def test_false_for_unrelated_fallback(self):
        """'fallback' alone (no 'model') must not trigger detection."""
        from backend.copilot.sdk.service import _is_fallback_stderr

        assert _is_fallback_stderr("Using cached result fallback") is False
        assert _is_fallback_stderr("Tool retry fallback triggered") is False

    def test_false_for_empty_line(self):
        from backend.copilot.sdk.service import _is_fallback_stderr

        assert _is_fallback_stderr("") is False

    def test_false_for_unrelated_stderr(self):
        from backend.copilot.sdk.service import _is_fallback_stderr

        assert _is_fallback_stderr("Task completed successfully") is False


# ---------------------------------------------------------------------------
# _EPHEMERAL_EVENT_TYPES — module-level constant
# ---------------------------------------------------------------------------


class TestEphemeralEventTypesConstant:
    """Verify _EPHEMERAL_EVENT_TYPES is a module-level constant that stays in
    sync with the event types that must not be counted toward events_yielded.
    """

    def test_stream_error_is_in_ephemeral_types(self):
        from backend.copilot.response_model import StreamError
        from backend.copilot.sdk.service import _EPHEMERAL_EVENT_TYPES

        assert StreamError in _EPHEMERAL_EVENT_TYPES, (
            "StreamError must be in _EPHEMERAL_EVENT_TYPES — if counted, "
            "transient retries would be blocked after the first notification"
        )

    def test_stream_status_is_in_ephemeral_types(self):
        from backend.copilot.response_model import StreamStatus
        from backend.copilot.sdk.service import _EPHEMERAL_EVENT_TYPES

        assert StreamStatus in _EPHEMERAL_EVENT_TYPES, (
            "StreamStatus must be in _EPHEMERAL_EVENT_TYPES so that a retry "
            "notification can be followed by another retry"
        )

    def test_stream_heartbeat_is_in_ephemeral_types(self):
        from backend.copilot.response_model import StreamHeartbeat
        from backend.copilot.sdk.service import _EPHEMERAL_EVENT_TYPES

        assert StreamHeartbeat in _EPHEMERAL_EVENT_TYPES

    def test_is_a_tuple(self):
        """isinstance() requires a tuple (not a list) as second argument."""
        from backend.copilot.sdk.service import _EPHEMERAL_EVENT_TYPES

        assert isinstance(_EPHEMERAL_EVENT_TYPES, tuple)


# ---------------------------------------------------------------------------
# TranscriptBuilder snapshot/restore
# ---------------------------------------------------------------------------


class TestTranscriptBuilderSnapshotRestore:
    """Verify that snapshot() and restore() provide safe rollback
    without accessing private attributes directly.
    """

    def test_snapshot_returns_independent_copy(self):
        """Mutations to the builder after snapshot() must not affect the snap."""
        from backend.copilot.transcript_builder import TranscriptBuilder

        builder = TranscriptBuilder()
        builder.append_user("hello")
        snap = builder.snapshot()

        builder.append_user("world")  # mutate after snapshot
        entries_copy, _ = snap
        assert len(entries_copy) == 1, "snapshot() must return an independent list copy"
        assert len(builder._entries) == 2

    def test_restore_resets_entries_and_uuid(self):
        """restore() must bring back the exact state at snapshot time."""
        from backend.copilot.transcript_builder import TranscriptBuilder

        builder = TranscriptBuilder()
        builder.append_user("first")
        snap = builder.snapshot()
        uuid_at_snap = builder._last_uuid

        builder.append_user("second")
        builder.restore(snap)

        assert len(builder._entries) == 1
        assert builder._last_uuid == uuid_at_snap

    def test_restore_to_empty_state(self):
        """Restoring a snapshot of an empty builder must clear all entries."""
        from backend.copilot.transcript_builder import TranscriptBuilder

        builder = TranscriptBuilder()
        empty_snap = builder.snapshot()

        builder.append_user("something")
        assert builder.entry_count == 1

        builder.restore(empty_snap)
        assert builder.entry_count == 0
        assert builder._last_uuid is None


# ---------------------------------------------------------------------------
# _last_reset_attempt guard
# ---------------------------------------------------------------------------


class TestLastResetAttemptGuard:
    """Verify that transient retries within the same context-level attempt
    do NOT reset the transient_retries counter (which would create an infinite loop).

    The guard works by tracking the last attempt value that triggered a reset.
    Transient retries `continue` without incrementing `attempt`, so the reset
    only fires when `attempt` actually advances (different attempt number).
    """

    def test_transient_retry_preserves_counter(self):
        """Simulating the loop: counter must NOT reset on transient continue.

        We simulate the _last_reset_attempt logic directly by replaying
        the loop state transitions.  A transient retry stays on the same
        `attempt` value, so the guard must block the reset.
        """
        attempt = 0
        _last_reset_attempt = -1
        transient_retries = 0

        # First entry into the loop: attempt 0 is different from -1, so reset.
        if attempt != _last_reset_attempt:
            transient_retries = 0
            _last_reset_attempt = attempt

        assert transient_retries == 0
        assert _last_reset_attempt == 0

        # Simulate transient retry: transient_retries incremented, `attempt`
        # stays 0 (transient continue does NOT call attempt += 1).
        transient_retries += 1

        # Loop continues: attempt is still 0, so guard blocks the reset.
        if attempt != _last_reset_attempt:
            transient_retries = 0  # must NOT execute
            _last_reset_attempt = attempt

        assert (
            transient_retries == 1
        ), "transient_retries must not be reset when attempt has not changed"

    def test_counter_resets_on_new_attempt(self):
        """When attempt advances to 1, transient_retries must reset to 0."""
        attempt = 0
        _last_reset_attempt = -1
        transient_retries = 0

        # attempt=0: first entry, reset fires.
        if attempt != _last_reset_attempt:
            transient_retries = 0
            _last_reset_attempt = attempt

        # Simulate transient retries used up.
        transient_retries = 3

        # Context compaction: attempt advances to 1.
        attempt = 1

        # Loop top: attempt changed, reset fires.
        if attempt != _last_reset_attempt:
            transient_retries = 0
            _last_reset_attempt = attempt

        assert (
            transient_retries == 0
        ), "transient_retries must reset to 0 when attempt advances"
        assert _last_reset_attempt == 1


# ---------------------------------------------------------------------------
# Integration: _HandledStreamError transient retry path
# ---------------------------------------------------------------------------


async def _drain(agen) -> list:
    """Collect all items from an async generator into a list."""
    items = []
    async for item in agen:
        items.append(item)
    return items


class TestHandledStreamErrorTransientRetry:
    """Integration tests for the _HandledStreamError transient retry wiring.

    These tests mock _run_stream_attempt to simulate the path where:
      1. _run_stream_attempt raises _HandledStreamError(code="transient_api_error")
      2. The outer loop calls _next_transient_backoff → gets a backoff
      3. _do_transient_backoff emits StreamStatus, sleeps, resets state
      4. The loop continues and _run_stream_attempt succeeds on the next call

    Verifies: StreamStatus emitted, no StreamError, second attempt returns
    content, and the transcript is restored between attempts.
    """

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_one_transient_handled_error(self):
        """_HandledStreamError(transient_api_error) → StreamStatus → success on retry.

        Simulates the retry loop logic directly, mirroring the real loop in
        stream_chat_completion_sdk.  Validates the composition of the three
        helper functions rather than calling stream_chat_completion_sdk (which
        requires DB/redis connections unavailable in unit tests).
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from backend.copilot.response_model import (
            StreamError,
            StreamFinish,
            StreamStatus,
        )
        from backend.copilot.sdk.service import (
            _do_transient_backoff,
            _HandledStreamError,
            _next_transient_backoff,
        )

        transient_retries = 0
        max_transient_retries = 3
        attempt = 0
        _last_reset_attempt = -1
        events_yielded = 0
        emitted: list = []

        call_count = 0

        async def fake_run_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: raise transient error (not yet yielded to client)
                raise _HandledStreamError(
                    "transient", code="transient_api_error", already_yielded=False
                )
            # Second call: success — yield a content event
            yield StreamFinish()

        state = MagicMock()
        state.usage = MagicMock()
        state.transcript_builder = MagicMock()
        state.transcript_builder.snapshot.return_value = ([], None)

        # Replay the retry loop body for up to 10 iterations.
        for _iteration in range(10):
            if attempt >= 3:
                break
            if attempt != _last_reset_attempt:
                transient_retries = 0
                _last_reset_attempt = attempt

            events_yielded = 0

            try:
                async for evt in fake_run_stream():
                    if not isinstance(evt, (StreamError, StreamStatus)):
                        events_yielded += 1
                    emitted.append(evt)
                break  # success
            except _HandledStreamError as exc:
                state.transcript_builder.restore(state.transcript_builder.snapshot())
                if exc.code == "transient_api_error":
                    backoff, transient_retries = _next_transient_backoff(
                        events_yielded, transient_retries, max_transient_retries
                    )
                    if backoff is not None:
                        with patch("asyncio.sleep", new=AsyncMock()):
                            async for evt in _do_transient_backoff(
                                backoff, state, "msg-id", "sess-id"
                            ):
                                emitted.append(evt)
                        continue  # retry

        # StreamStatus emitted (retry notification)
        statuses = [e for e in emitted if isinstance(e, StreamStatus)]
        assert len(statuses) == 1
        assert (
            "retry" in statuses[0].message.lower()
            or "connection" in statuses[0].message.lower()
        )

        # No StreamError — transient error not surfaced when retry succeeded
        errors = [e for e in emitted if isinstance(e, StreamError)]
        assert len(errors) == 0

        # Content from successful second attempt is present
        content_events = [e for e in emitted if isinstance(e, StreamFinish)]
        assert len(content_events) == 1

        # Second attempt was called
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_transient_handled_error_exhaustion_yields_stream_error(self):
        """When all transient retries exhausted, StreamError must be yielded."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from backend.copilot.constants import FRIENDLY_TRANSIENT_MSG
        from backend.copilot.response_model import StreamError, StreamStatus
        from backend.copilot.sdk.service import (
            _do_transient_backoff,
            _HandledStreamError,
            _next_transient_backoff,
        )

        transient_retries = 0
        max_transient_retries = 2  # exhaust after 2 retries
        attempt = 0
        _last_reset_attempt = -1
        emitted: list = []

        async def always_fail():
            raise _HandledStreamError(
                "transient",
                error_msg="API overloaded",
                code="transient_api_error",
                already_yielded=False,
            )
            # Satisfy the type checker — unreachable
            return
            yield  # noqa: B901

        state = MagicMock()
        state.usage = MagicMock()
        state.transcript_builder = MagicMock()
        state.transcript_builder.snapshot.return_value = ([], None)

        ended_with_stream_error = False

        for _iteration in range(20):
            if attempt >= 3:
                break
            if attempt != _last_reset_attempt:
                transient_retries = 0
                _last_reset_attempt = attempt

            events_yielded = 0
            try:
                async for evt in always_fail():
                    emitted.append(evt)
                break
            except _HandledStreamError as exc:
                state.transcript_builder.restore(state.transcript_builder.snapshot())
                if exc.code == "transient_api_error":
                    backoff, transient_retries = _next_transient_backoff(
                        events_yielded, transient_retries, max_transient_retries
                    )
                    if backoff is not None:
                        with patch("asyncio.sleep", new=AsyncMock()):
                            async for evt in _do_transient_backoff(
                                backoff, state, "m", "s"
                            ):
                                emitted.append(evt)
                        continue
                # retries exhausted
                ended_with_stream_error = True
                if not exc.already_yielded:
                    emitted.append(
                        StreamError(
                            errorText=exc.error_msg or FRIENDLY_TRANSIENT_MSG,
                            code=exc.code or "transient_api_error",
                        )
                    )
                break

        # Two StreamStatus events emitted (one per retry before exhaustion)
        statuses = [e for e in emitted if isinstance(e, StreamStatus)]
        assert len(statuses) == max_transient_retries

        # One StreamError emitted after exhaustion
        errors = [e for e in emitted if isinstance(e, StreamError)]
        assert len(errors) == 1
        assert errors[0].code == "transient_api_error"

        assert ended_with_stream_error is True


# ---------------------------------------------------------------------------
# Integration: generic Exception transient retry path
# ---------------------------------------------------------------------------


class TestGenericExceptionTransientRetry:
    """Integration tests for the raw Exception transient retry wiring.

    These tests simulate the Exception handler path (e.g. ECONNRESET) that
    is a separate code path from _HandledStreamError.  The same retry
    mechanics apply — _next_transient_backoff + _do_transient_backoff +
    continue — but the entry path is different (no already_yielded flag,
    no pre-yielded StreamError from _run_stream_attempt).
    """

    @pytest.mark.asyncio
    async def test_econnreset_triggers_retry_and_succeeds(self):
        """Raw Exception('ECONNRESET') matching is_transient_api_error → retry succeeds.

        Simulates the generic Exception handler path — separate from
        _HandledStreamError.  Verifies the same retry mechanics apply:
        _next_transient_backoff + _do_transient_backoff + continue.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from backend.copilot.constants import is_transient_api_error
        from backend.copilot.response_model import (
            StreamError,
            StreamFinish,
            StreamStatus,
        )
        from backend.copilot.sdk.service import (
            _do_transient_backoff,
            _next_transient_backoff,
        )

        transient_retries = 0
        max_transient_retries = 3
        attempt = 0
        _last_reset_attempt = -1
        emitted: list = []
        call_count = 0

        async def fake_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("ECONNRESET")
            yield StreamFinish()

        state = MagicMock()
        state.usage = MagicMock()
        state.transcript_builder = MagicMock()
        state.transcript_builder.snapshot.return_value = ([], None)

        for _iteration in range(10):
            if attempt >= 3:
                break
            if attempt != _last_reset_attempt:
                transient_retries = 0
                _last_reset_attempt = attempt

            events_yielded = 0
            try:
                async for evt in fake_stream():
                    if not isinstance(evt, (StreamError, StreamStatus)):
                        events_yielded += 1
                    emitted.append(evt)
                break
            except Exception as exc:
                is_transient = is_transient_api_error(str(exc))
                state.transcript_builder.restore(state.transcript_builder.snapshot())
                if events_yielded == 0 and is_transient:
                    backoff, transient_retries = _next_transient_backoff(
                        events_yielded, transient_retries, max_transient_retries
                    )
                    if backoff is not None:
                        with patch("asyncio.sleep", new=AsyncMock()):
                            async for evt in _do_transient_backoff(
                                backoff, state, "m", "s"
                            ):
                                emitted.append(evt)
                        continue
                break

        # StreamStatus emitted during retry (notification to client)
        statuses = [e for e in emitted if isinstance(e, StreamStatus)]
        assert len(statuses) == 1

        # No StreamError — retry succeeded
        errors = [e for e in emitted if isinstance(e, StreamError)]
        assert len(errors) == 0

        # Content from successful second attempt
        finish_events = [e for e in emitted if isinstance(e, StreamFinish)]
        assert len(finish_events) == 1

        # Two total calls: first fails (ECONNRESET), second succeeds
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_generic_exception_not_retried_when_events_yielded(self):
        """When events_yielded > 0, transient Exception must NOT trigger retry."""
        from backend.copilot.constants import is_transient_api_error
        from backend.copilot.sdk.service import _next_transient_backoff

        events_yielded = 1  # content already sent
        transient_retries = 0
        max_transient_retries = 3

        # The real loop checks events_yielded before calling _next_transient_backoff.
        # We replicate that check here.
        exc = Exception("ECONNRESET")
        is_transient = is_transient_api_error(str(exc))

        assert is_transient is True

        # Mimic the loop guard: only call backoff function when events_yielded == 0
        if events_yielded > 0 or not is_transient:
            backoff = None
        else:
            backoff, _ = _next_transient_backoff(
                events_yielded, transient_retries, max_transient_retries
            )

        assert (
            backoff is None
        ), "retry must not be attempted when events have already been sent to the client"


# ---------------------------------------------------------------------------
# _session_messages_to_transcript
# ---------------------------------------------------------------------------


class TestSessionMessagesToTranscript:
    """Unit tests for _session_messages_to_transcript.

    Verifies that ChatMessage lists are converted to valid JSONL transcripts
    with proper tool_use / tool_result content blocks — giving the Claude CLI
    full structural context via --resume when no previous transcript file is
    available.
    """

    def _parse_jsonl(self, jsonl: str) -> list[dict]:
        import json as _json

        return [
            _json.loads(line) for line in jsonl.strip().splitlines() if line.strip()
        ]

    def test_empty_messages_returns_empty_string(self):
        from backend.copilot.sdk.service import _session_messages_to_transcript

        assert _session_messages_to_transcript([]) == ""

    def test_simple_user_assistant_messages(self):
        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _session_messages_to_transcript

        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
        ]
        result = _session_messages_to_transcript(messages)
        entries = self._parse_jsonl(result)

        assert len(entries) == 2
        assert entries[0]["type"] == "user"
        assert entries[0]["message"]["role"] == "user"
        assert entries[0]["message"]["content"] == "Hello"
        assert entries[1]["type"] == "assistant"
        assert entries[1]["message"]["role"] == "assistant"
        content_blocks = entries[1]["message"]["content"]
        assert any(
            b.get("type") == "text" and b.get("text") == "Hi there"
            for b in content_blocks
        )

    def test_assistant_with_tool_calls_produces_tool_use_blocks(self):
        import json as _json

        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _session_messages_to_transcript

        messages = [
            ChatMessage(role="user", content="List files"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": _json.dumps({"cmd": "ls -la"}),
                        },
                    }
                ],
            ),
        ]
        result = _session_messages_to_transcript(messages)
        entries = self._parse_jsonl(result)

        assert len(entries) == 2
        assistant_entry = entries[1]
        assert assistant_entry["type"] == "assistant"
        blocks = assistant_entry["message"]["content"]
        tool_use_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        tu = tool_use_blocks[0]
        assert tu["id"] == "call_abc123"
        assert tu["name"] == "bash"
        assert tu["input"] == {"cmd": "ls -la"}

    def test_tool_result_produces_tool_result_block(self):
        import json as _json

        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _session_messages_to_transcript

        messages = [
            ChatMessage(role="user", content="List files"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": _json.dumps({"cmd": "ls"}),
                        },
                    }
                ],
            ),
            ChatMessage(
                role="tool",
                tool_call_id="call_xyz",
                content="file1.txt\nfile2.txt\nfile3.txt",
            ),
        ]
        result = _session_messages_to_transcript(messages)
        entries = self._parse_jsonl(result)

        # user + assistant + user(tool_result)
        assert len(entries) == 3
        tool_result_entry = entries[2]
        assert tool_result_entry["type"] == "user"
        content = tool_result_entry["message"]["content"]
        assert isinstance(content, list)
        tr_blocks = [b for b in content if b.get("type") == "tool_result"]
        assert len(tr_blocks) == 1
        assert tr_blocks[0]["tool_use_id"] == "call_xyz"
        assert tr_blocks[0]["content"] == "file1.txt\nfile2.txt\nfile3.txt"

    def test_tool_result_full_content_not_truncated(self):
        """Tool result content must NOT be truncated (unlike _format_conversation_context
        which caps at 500 chars)."""
        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _session_messages_to_transcript

        big_output = "x" * 10_000
        messages = [
            ChatMessage(role="user", content="q"),
            ChatMessage(
                role="tool",
                tool_call_id="call_big",
                content=big_output,
            ),
        ]
        result = _session_messages_to_transcript(messages)
        entries = self._parse_jsonl(result)
        tool_result_entry = next(
            e
            for e in entries
            if e.get("type") == "user" and isinstance(e["message"].get("content"), list)
        )
        tr_block = next(
            b
            for b in tool_result_entry["message"]["content"]
            if b.get("type") == "tool_result"
        )
        assert tr_block["content"] == big_output, "Tool result must not be truncated"

    def test_parent_uuid_chain_is_correct(self):
        """Each entry's parentUuid must equal the previous entry's uuid."""
        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _session_messages_to_transcript

        messages = [
            ChatMessage(role="user", content="A"),
            ChatMessage(role="assistant", content="B"),
            ChatMessage(role="user", content="C"),
        ]
        result = _session_messages_to_transcript(messages)
        entries = self._parse_jsonl(result)

        assert entries[0]["parentUuid"] == ""
        for i in range(1, len(entries)):
            assert (
                entries[i]["parentUuid"] == entries[i - 1]["uuid"]
            ), f"Entry {i} parentUuid mismatch"

    def test_invalid_tool_call_arguments_use_empty_input(self):
        """Malformed JSON in tool_call arguments must not raise — use empty dict."""
        from backend.copilot.model import ChatMessage
        from backend.copilot.sdk.service import _session_messages_to_transcript

        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "broken_tool",
                            "arguments": "NOT_VALID_JSON",
                        },
                    }
                ],
            ),
        ]
        result = _session_messages_to_transcript(messages)
        entries = self._parse_jsonl(result)
        assert len(entries) == 1
        blocks = entries[0]["message"]["content"]
        tu = next(b for b in blocks if b.get("type") == "tool_use")
        assert tu["input"] == {}
