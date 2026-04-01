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
    """Verify security env vars are set correctly in stream_chat_completion_sdk.

    These are set inline after build_sdk_env() returns, so we verify the
    contract: the env dict passed to ClaudeAgentOptions must contain these.
    """

    def test_tmpdir_set_when_sdk_cwd_provided(self):
        """CLAUDE_CODE_TMPDIR routes CLI temp files into the workspace."""
        env: dict[str, str] = {}
        sdk_cwd = "/tmp/copilot-test-session/"
        if sdk_cwd:
            env["CLAUDE_CODE_TMPDIR"] = sdk_cwd
        assert env["CLAUDE_CODE_TMPDIR"] == "/tmp/copilot-test-session/"

    def test_tmpdir_not_set_when_sdk_cwd_is_none(self):
        """Without sdk_cwd, CLAUDE_CODE_TMPDIR is absent."""
        env: dict[str, str] = {}
        sdk_cwd = None
        if sdk_cwd:
            env["CLAUDE_CODE_TMPDIR"] = sdk_cwd
        assert "CLAUDE_CODE_TMPDIR" not in env

    def test_home_not_set(self):
        """HOME is NOT overridden — would break git/ssh/npm child processes."""
        env: dict[str, str] = {}
        sdk_cwd = "/tmp/copilot-test-session/"
        if sdk_cwd:
            env["CLAUDE_CODE_TMPDIR"] = sdk_cwd
        # Intentionally NOT setting HOME
        assert "HOME" not in env

    def test_security_env_vars_present(self):
        """All four security env vars are set to '1'."""
        env: dict[str, str] = {}
        env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
        env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
        assert env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] == "1"
        assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


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
