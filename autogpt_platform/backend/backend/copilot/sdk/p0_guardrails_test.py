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
