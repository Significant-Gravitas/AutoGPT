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
    "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
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
