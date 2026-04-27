"""Unit tests for ChatConfig."""

import pytest

from .config import ChatConfig

# Env vars that the ChatConfig validators read — must be cleared so they don't
# override the explicit constructor values we pass in each test.  Includes the
# SDK/baseline model aliases so a leftover ``CHAT_MODEL=...`` in the developer
# or CI environment can't change whether
# ``_validate_sdk_model_vendor_compatibility`` raises.
_ENV_VARS_TO_CLEAR = (
    "CHAT_USE_E2B_SANDBOX",
    "CHAT_E2B_API_KEY",
    "E2B_API_KEY",
    "CHAT_USE_OPENROUTER",
    "CHAT_USE_CLAUDE_AGENT_SDK",
    "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION",
    "CHAT_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPENAI_API_KEY",
    "CHAT_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OPENAI_BASE_URL",
    "CHAT_CLAUDE_AGENT_CLI_PATH",
    "CLAUDE_AGENT_CLI_PATH",
    "CHAT_FAST_STANDARD_MODEL",
    "CHAT_FAST_MODEL",
    "CHAT_FAST_ADVANCED_MODEL",
    "CHAT_THINKING_STANDARD_MODEL",
    "CHAT_THINKING_ADVANCED_MODEL",
    "CHAT_MODEL",
    "CHAT_ADVANCED_MODEL",
    "CHAT_CLAUDE_AGENT_FALLBACK_MODEL",
    "CHAT_RENDER_REASONING_IN_UI",
    "CHAT_STREAM_REPLAY_COUNT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


def _make_direct_safe_config(**kwargs) -> ChatConfig:
    """Build a ``ChatConfig`` for tests that pass ``use_openrouter=False``
    but aren't exercising the SDK vendor-compatibility validator.

    Pins ``thinking_standard_model``/``thinking_advanced_model`` to anthropic/*
    so the construction passes ``_validate_sdk_model_vendor_compatibility``
    without each test having to repeat the override.
    """
    defaults: dict = {
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
    }
    defaults.update(kwargs)
    return ChatConfig(**defaults)


class TestOpenrouterActive:
    """Tests for the openrouter_active property."""

    def test_enabled_with_credentials_returns_true(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.openrouter_active is True

    def test_enabled_but_missing_api_key_returns_false(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.openrouter_active is False

    def test_disabled_returns_false_despite_credentials(self):
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.openrouter_active is False

    def test_strips_v1_suffix_and_still_valid(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.openrouter_active is True

    def test_invalid_base_url_returns_false(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="not-a-url",
        )
        assert cfg.openrouter_active is False


class TestE2BActive:
    """Tests for the e2b_active property — single source of truth for E2B usage."""

    def test_both_enabled_and_key_present_returns_true(self):
        """e2b_active is True when use_e2b_sandbox=True and e2b_api_key is set."""
        cfg = ChatConfig(use_e2b_sandbox=True, e2b_api_key="test-key")
        assert cfg.e2b_active is True

    def test_enabled_but_missing_key_returns_false(self):
        """e2b_active is False when use_e2b_sandbox=True but e2b_api_key is absent."""
        cfg = ChatConfig(use_e2b_sandbox=True, e2b_api_key=None)
        assert cfg.e2b_active is False

    def test_disabled_returns_false(self):
        """e2b_active is False when use_e2b_sandbox=False regardless of key."""
        cfg = ChatConfig(use_e2b_sandbox=False, e2b_api_key="test-key")
        assert cfg.e2b_active is False


class TestClaudeAgentCliPathEnvFallback:
    """``claude_agent_cli_path`` accepts both the Pydantic-prefixed
    ``CHAT_CLAUDE_AGENT_CLI_PATH`` env var and the unprefixed
    ``CLAUDE_AGENT_CLI_PATH`` form (mirrors ``api_key`` / ``base_url``).
    """

    def test_prefixed_env_var_is_picked_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake_cli = tmp_path / "fake-claude"
        fake_cli.write_text("#!/bin/sh\n")
        fake_cli.chmod(0o755)
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", str(fake_cli))
        cfg = ChatConfig()
        assert cfg.claude_agent_cli_path == str(fake_cli)

    def test_unprefixed_env_var_is_picked_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake_cli = tmp_path / "fake-claude"
        fake_cli.write_text("#!/bin/sh\n")
        fake_cli.chmod(0o755)
        monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", str(fake_cli))
        cfg = ChatConfig()
        assert cfg.claude_agent_cli_path == str(fake_cli)

    def test_prefixed_wins_over_unprefixed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        prefixed_cli = tmp_path / "fake-claude-prefixed"
        prefixed_cli.write_text("#!/bin/sh\n")
        prefixed_cli.chmod(0o755)
        unprefixed_cli = tmp_path / "fake-claude-unprefixed"
        unprefixed_cli.write_text("#!/bin/sh\n")
        unprefixed_cli.chmod(0o755)
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_CLI_PATH", str(prefixed_cli))
        monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", str(unprefixed_cli))
        cfg = ChatConfig()
        assert cfg.claude_agent_cli_path == str(prefixed_cli)

    def test_no_env_var_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = ChatConfig()
        assert cfg.claude_agent_cli_path is None

    def test_nonexistent_path_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-existent CLI path must be rejected at config time, not at
        runtime when subprocess.run fails with an opaque OS error."""
        monkeypatch.setenv(
            "CLAUDE_AGENT_CLI_PATH", "/opt/nonexistent/claude-cli-binary"
        )
        with pytest.raises(Exception, match="does not exist"):
            ChatConfig()

    def test_non_executable_path_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Path that exists but is not executable must be rejected."""
        non_exec = tmp_path / "claude-not-executable"
        non_exec.write_text("#!/bin/sh\n")
        non_exec.chmod(0o644)  # readable but not executable
        monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", str(non_exec))
        with pytest.raises(Exception, match="not executable"):
            ChatConfig()

    def test_directory_path_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Path pointing to a directory must be rejected."""
        monkeypatch.setenv("CLAUDE_AGENT_CLI_PATH", str(tmp_path))
        with pytest.raises(Exception, match="not a regular file"):
            ChatConfig()


class TestSdkModelVendorCompatibility:
    """``model_validator`` that fails fast on SDK model vs routing-mode
    mismatch — see PR #12878 iteration-2 review.  Mirrors the runtime
    guard in ``_normalize_model_name`` so misconfig surfaces at boot
    instead of as a 500 on the first SDK turn."""

    def test_direct_anthropic_with_kimi_override_raises(self):
        """A non-Anthropic SDK model must fail at config load when the
        deployment has no OpenRouter credentials."""
        with pytest.raises(Exception, match="requires an Anthropic model"):
            ChatConfig(
                use_openrouter=False,
                api_key=None,
                base_url=None,
                use_claude_code_subscription=False,
                thinking_standard_model="moonshotai/kimi-k2.6",
            )

    def test_direct_anthropic_with_anthropic_default_succeeds(self):
        """Direct-Anthropic mode is fine when both SDK slugs are anthropic/*
        — which is the default after the LD-routed model rollout."""
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
        )
        assert cfg.thinking_standard_model == "anthropic/claude-sonnet-4-6"

    def test_openrouter_with_kimi_override_succeeds(self):
        """Kimi slug round-trips cleanly when OpenRouter is on — exercised
        via the LD-flag override path in production."""
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            use_claude_code_subscription=False,
            thinking_standard_model="moonshotai/kimi-k2.6",
        )
        assert cfg.thinking_standard_model == "moonshotai/kimi-k2.6"

    def test_subscription_mode_skips_check(self):
        """Subscription path resolves the model to None and bypasses
        ``_normalize_model_name``, so the slug check is skipped."""
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=True,
        )
        assert cfg.use_claude_code_subscription is True

    def test_advanced_tier_also_validated(self):
        """Both standard and advanced SDK slugs are checked."""
        with pytest.raises(Exception, match="thinking_advanced_model"):
            ChatConfig(
                use_openrouter=False,
                api_key=None,
                base_url=None,
                use_claude_code_subscription=False,
                thinking_standard_model="anthropic/claude-sonnet-4-6",
                thinking_advanced_model="moonshotai/kimi-k2.6",
            )

    def test_fallback_model_also_validated(self):
        """``claude_agent_fallback_model`` flows through
        ``_normalize_model_name`` via ``_resolve_fallback_model`` so the
        same direct-Anthropic guard applies."""
        with pytest.raises(Exception, match="claude_agent_fallback_model"):
            ChatConfig(
                use_openrouter=False,
                api_key=None,
                base_url=None,
                use_claude_code_subscription=False,
                thinking_standard_model="anthropic/claude-sonnet-4-6",
                thinking_advanced_model="anthropic/claude-opus-4-7",
                claude_agent_fallback_model="moonshotai/kimi-k2.6",
            )

    def test_empty_fallback_skipped(self):
        """Empty ``claude_agent_fallback_model`` (no fallback configured)
        must not trip the validator — the fallback-disabled state is
        intentional and shouldn't require a placeholder anthropic/* slug."""
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            thinking_standard_model="anthropic/claude-sonnet-4-6",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            claude_agent_fallback_model="",
        )
        assert cfg.claude_agent_fallback_model == ""


class TestRenderReasoningInUi:
    """``render_reasoning_in_ui`` gates reasoning wire events globally."""

    def test_defaults_to_true(self):
        """Default must stay True — flipping it silences the reasoning
        collapse for every user, which is an opt-in operator decision."""
        cfg = ChatConfig()
        assert cfg.render_reasoning_in_ui is True

    def test_env_override_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAT_RENDER_REASONING_IN_UI", "false")
        cfg = ChatConfig()
        assert cfg.render_reasoning_in_ui is False


class TestStreamReplayCount:
    """``stream_replay_count`` caps the SSE reconnect replay batch size."""

    def test_default_is_200(self):
        """200 covers a full Kimi turn after coalescing (~150 events) while
        bounding the replay storm from 1000+ chunks."""
        cfg = ChatConfig()
        assert cfg.stream_replay_count == 200

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAT_STREAM_REPLAY_COUNT", "500")
        cfg = ChatConfig()
        assert cfg.stream_replay_count == 500

    def test_zero_rejected(self):
        """count=0 would make XREAD replay nothing — rejected via ge=1."""
        with pytest.raises(Exception):
            ChatConfig(stream_replay_count=0)
