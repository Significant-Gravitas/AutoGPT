"""Unit tests for ChatConfig."""

import pytest
from pydantic import ValidationError

from .chat_env import chat_environment_problems
from .config import ChatConfig

# Env vars ChatConfig or its validators read, cleared so they don't override the
# explicit constructor values each test passes.
_ENV_VARS_TO_CLEAR = (
    "CHAT_PLATFORM_ROUTE",
    "CHAT_OPENROUTER_API_KEY",
    "CHAT_OPENROUTER_BASE_URL",
    "CHAT_ANTHROPIC_API_KEY",
    "CHAT_ANTHROPIC_BASE_URL",
    "CHAT_LOCAL_BASE_URL",
    "CHAT_LOCAL_API_KEY",
    "CHAT_LOCAL_ADAPTER",
    "CHAT_LOCAL_CONTEXT_TOKENS",
    "CHAT_STANDARD_MODEL",
    "CHAT_ADVANCED_MODEL",
    "CHAT_USE_E2B_SANDBOX",
    "CHAT_E2B_API_KEY",
    "E2B_API_KEY",
    "ANTHROPIC_API_KEY",
    "CHAT_CLAUDE_AGENT_CLI_PATH",
    "CLAUDE_AGENT_CLI_PATH",
    "CHAT_STREAM_REPLAY_COUNT",
    "CHAT_CLAUDE_AGENT_EXTRA_BUILTIN_TOOLS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    # pydantic-settings also reads backend/.env directly (``env_file``), so
    # scrubbing the process env is not enough. Disable the file source so tests
    # see field defaults plus their explicit constructor values only.
    monkeypatch.setitem(ChatConfig.model_config, "env_file", None)


class TestPlatformRoute:
    def test_defaults_to_disabled(self):
        cfg = ChatConfig()
        assert cfg.platform_route == "disabled"
        assert cfg.route_row is None
        assert cfg.tier_model("standard") is None
        assert cfg.main_client_credentials == (None, None)

    def test_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("CHAT_PLATFORM_ROUTE", "openrouter")
        monkeypatch.setenv("CHAT_OPENROUTER_API_KEY", "sk-or")
        cfg = ChatConfig()
        assert cfg.route_row is not None and cfg.route_row.name == "openrouter"
        assert cfg.openrouter_api_key == "sk-or"

    def test_a_value_that_is_no_route_reads_as_disabled(self, monkeypatch):
        """The startup check reports it; construction never fails on it."""
        monkeypatch.setenv("CHAT_PLATFORM_ROUTE", "open_router")
        assert ChatConfig().platform_route == "disabled"

    @pytest.mark.parametrize(
        "selector",
        ["openrouter", " openrouter ", "openrouter\n", "\tdisabled", "OpenRouter"],
    )
    def test_the_startup_check_and_the_settings_read_the_route_alike(
        self, monkeypatch, selector
    ):
        """The route the check accepts is the route that runs; one it refuses
        never runs."""
        env = {"CHAT_PLATFORM_ROUTE": selector, "CHAT_OPENROUTER_API_KEY": "sk-or"}
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        problems, _ = chat_environment_problems(env)
        accepted = not any(p.startswith("CHAT_PLATFORM_ROUTE ") for p in problems)
        assert accepted == (selector.strip() in ("openrouter", "disabled"))
        expected = selector.strip() if accepted else "disabled"
        assert ChatConfig().platform_route == expected

    @pytest.mark.parametrize(
        "selector", ["openai_chat", " anthropic_passthrough\n", "Ollama"]
    )
    def test_the_startup_check_and_the_settings_read_the_adapter_alike(
        self, monkeypatch, selector
    ):
        env = {
            "CHAT_PLATFORM_ROUTE": "local",
            "CHAT_LOCAL_BASE_URL": "http://127.0.0.1:8080",
            "CHAT_LOCAL_ADAPTER": selector,
            "CHAT_LOCAL_CONTEXT_TOKENS": "65536",
            "CHAT_STANDARD_MODEL": "qwen",
            "CHAT_ADVANCED_MODEL": "qwen",
        }
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        problems, _ = chat_environment_problems(env)
        accepted = problems == []
        assert accepted == (selector != "Ollama")
        expected = selector.strip() if accepted else None
        assert ChatConfig().local_adapter == expected

    def test_an_unknown_adapter_and_unreadable_window_read_as_unset(self, monkeypatch):
        monkeypatch.setenv("CHAT_LOCAL_ADAPTER", "ollama")
        monkeypatch.setenv("CHAT_LOCAL_CONTEXT_TOKENS", "64k")
        cfg = ChatConfig()
        assert cfg.local_adapter is None
        assert cfg.local_context_tokens is None

    def test_a_keyword_that_is_no_setting_is_refused(self):
        with pytest.raises(TypeError, match="use_openrouter"):
            ChatConfig(use_openrouter=True)


class TestTiers:
    def test_route_defaults(self):
        cfg = ChatConfig(platform_route="openrouter")
        assert cfg.tier_model("standard") == "anthropic/claude-sonnet-5-5"
        assert cfg.tier_model("advanced") == "anthropic/claude-opus-5-5"

    def test_operator_tiers_win(self):
        cfg = ChatConfig(
            platform_route="openrouter",
            standard_model=" moonshotai/kimi-k2.6 ",
            advanced_model="anthropic/claude-opus-5",
        )
        assert cfg.tier_model("standard") == "moonshotai/kimi-k2.6"
        assert cfg.tier_model("advanced") == "anthropic/claude-opus-5"

    def test_local_has_no_default_tier(self):
        assert ChatConfig(platform_route="local").tier_model("standard") is None

    def test_jobs_fall_back_to_the_hosted_default(self):
        assert ChatConfig().job_tier_model("advanced") == "anthropic/claude-opus-5-5"


class TestSideCalls:
    def test_openrouter_uses_its_v1(self):
        cfg = ChatConfig(platform_route="openrouter", openrouter_api_key="sk-or")
        assert cfg.main_client_credentials == ("sk-or", "https://openrouter.ai/api/v1")
        assert cfg.aux_client_credentials == cfg.main_client_credentials
        assert cfg.aux_uses_openrouter
        assert cfg.aux_provider_label == "open_router"
        assert cfg.aux_model == "anthropic/claude-haiku-5.5"

    def test_another_server_on_the_openrouter_route_gets_plain_requests(self):
        """OpenRouter's extra request fields go to OpenRouter only: another
        OpenRouter-style server (a proxy, a gateway) rejects them."""
        cfg = ChatConfig(
            platform_route="openrouter",
            openrouter_api_key="sk-or",
            openrouter_base_url="https://gateway.example.test/api",
        )
        assert not cfg.aux_uses_openrouter

    def test_direct_uses_anthropics_openai_compatible_api(self):
        cfg = ChatConfig(platform_route="direct_anthropic", anthropic_api_key="sk-ant")
        assert cfg.main_client_credentials == (
            "sk-ant",
            "https://api.anthropic.com/v1/",
        )
        assert not cfg.aux_uses_openrouter
        assert cfg.aux_provider_label == "anthropic"

    @pytest.mark.parametrize(
        ("adapter", "url", "expected"),
        [
            ("openai_chat", "http://ollama:11434/v1", "http://ollama:11434/v1"),
            ("anthropic_passthrough", "http://llama:8080", "http://llama:8080/v1"),
        ],
    )
    def test_local_uses_the_servers_openai_api(self, adapter, url, expected):
        cfg = ChatConfig(
            platform_route="local",
            local_base_url=url,
            local_adapter=adapter,
            standard_model="qwen3:8b",
        )
        assert cfg.main_client_credentials == ("local", expected)
        assert cfg.aux_provider_label == "local"
        assert cfg.aux_model == "qwen3:8b"
        assert cfg.simulation_model == "qwen3:8b"
        assert cfg.local_request_timeout_s == 1800.0

    @pytest.mark.parametrize("route", ["subscription", "disabled"])
    def test_no_api_no_side_call_client(self, route):
        assert ChatConfig(platform_route=route).aux_client_credentials == (None, None)

    def test_the_batch_key_only_on_the_routes_that_use_the_batch_api(self, mocker):
        """OpenRouter and direct take the Batch key (the direct route's, else
        the integration key); a Claude login, a local server and a disabled
        route have none, whatever keys the deployment holds."""
        from .config import settings

        mocker.patch.object(settings.secrets, "anthropic_api_key", "sk-batch")
        for route in ("openrouter", "direct_anthropic"):
            assert (
                ChatConfig(platform_route=route).batch_anthropic_api_key == "sk-batch"
            )
        cfg = ChatConfig(
            platform_route="direct_anthropic", anthropic_api_key="sk-direct"
        )
        assert cfg.batch_anthropic_api_key == "sk-direct"
        for route in ("subscription", "local", "disabled"):
            cfg = ChatConfig(platform_route=route, anthropic_api_key="sk-direct")
            assert cfg.batch_anthropic_api_key is None

    @pytest.mark.parametrize(
        "route, expected",
        [
            ("openrouter", True),
            ("direct_anthropic", True),
            ("local", True),
            ("subscription", False),
            ("disabled", False),
        ],
    )
    def test_which_routes_have_a_platform_api(self, route, expected):
        assert ChatConfig(platform_route=route).has_platform_api is expected


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


class TestStreamReplayCount:
    """``stream_replay_count`` is the SSE reconnect replay batch size."""

    def test_default_is_200(self):
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


class TestLangfusePromptCacheTTL:
    def test_default_is_five_minutes(self):
        # Read the field default, not an instance: backend/.env can set
        # CHAT_LANGFUSE_PROMPT_CACHE_TTL and mask it.
        assert ChatConfig.model_fields["langfuse_prompt_cache_ttl"].default == 300

    def test_a_negative_ttl_is_rejected(self):
        # A negative TTL would skip our revalidation and expire the SDK entry
        # at once, which is the unbounded staleness this field exists to avoid.
        with pytest.raises(ValidationError, match="langfuse_prompt_cache_ttl"):
            ChatConfig(langfuse_prompt_cache_ttl=-1)


class TestClaudeAgentExtraBuiltinTools:
    def test_default_adds_no_builtin(self):
        # The field default, not an instance: backend/.env can set it.
        field = ChatConfig.model_fields["claude_agent_extra_builtin_tools"]
        assert field.get_default(call_default_factory=True) == []

    def test_env_var_reads_a_json_list(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CHAT_CLAUDE_AGENT_EXTRA_BUILTIN_TOOLS", '["Agent"]')
        assert ChatConfig().claude_agent_extra_builtin_tools == ["Agent"]
