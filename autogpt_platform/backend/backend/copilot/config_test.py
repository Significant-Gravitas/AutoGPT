"""Unit tests for ChatConfig."""

import pytest

from .config import ChatConfig

# Env vars that the ChatConfig validators read — must be cleared so they don't
# override the explicit constructor values we pass in each test.
_ENV_VARS_TO_CLEAR = (
    "CHAT_USE_E2B_SANDBOX",
    "CHAT_E2B_API_KEY",
    "E2B_API_KEY",
    "CHAT_USE_OPENROUTER",
    "CHAT_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPENAI_API_KEY",
    "CHAT_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OPENAI_BASE_URL",
    "CHAT_CLAUDE_AGENT_CLI_PATH",
    "CLAUDE_AGENT_CLI_PATH",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


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
        cfg = ChatConfig(
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
