"""Tests for P0 guardrails: _resolve_fallback_model, security env vars, TMPDIR."""

from unittest.mock import patch

from backend.copilot.config import ChatConfig


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


class TestSecurityEnvVars:
    """Verify the env-var contract in the service module.

    The production code sets CLAUDE_CODE_TMPDIR and security env vars
    inline after ``build_sdk_env()`` returns.  We grep for these string
    literals in ``service.py`` to ensure they aren't accidentally removed.
    """

    _SERVICE_PATH = "autogpt_platform/backend/backend/copilot/sdk/service.py"

    @staticmethod
    def _read_service_source() -> str:
        import pathlib

        # Walk up from this test file to the repo root
        repo = pathlib.Path(__file__).resolve().parents[5]
        return (repo / TestSecurityEnvVars._SERVICE_PATH).read_text()

    def test_tmpdir_env_var_present_in_source(self):
        """CLAUDE_CODE_TMPDIR must be set when sdk_cwd is provided."""
        src = self._read_service_source()
        assert 'sdk_env["CLAUDE_CODE_TMPDIR"]' in src

    def test_home_not_overridden_in_source(self):
        """HOME must NOT be overridden — would break git/ssh/npm."""
        src = self._read_service_source()
        assert 'sdk_env["HOME"]' not in src

    def test_security_env_vars_present_in_source(self):
        """All four security env vars must be set in the service module."""
        src = self._read_service_source()
        for var in (
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        ):
            assert var in src, f"{var} not found in service.py"


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
        assert cfg.claude_agent_max_turns == 50

    def test_max_budget_usd_default(self):
        cfg = _make_config()
        assert cfg.claude_agent_max_budget_usd == 5.0

    def test_max_transient_retries_default(self):
        cfg = _make_config()
        assert cfg.claude_agent_max_transient_retries == 3


# ---------------------------------------------------------------------------
# build_sdk_env — all 3 auth modes
# ---------------------------------------------------------------------------

_ENV = "backend.copilot.sdk.env"


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

    def test_direct_anthropic_returns_empty_dict(self):
        """Mode 2: direct Anthropic returns {} (inherits from parent env)."""
        cfg = _make_config(
            use_claude_code_subscription=False,
            use_openrouter=False,
        )
        with patch(f"{_ENV}.config", cfg):
            from backend.copilot.sdk.env import build_sdk_env

            env = build_sdk_env()

        assert env == {}

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
        """build_sdk_env must return a mutable dict (not None) so callers
        can add security env vars like CLAUDE_CODE_TMPDIR."""
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
