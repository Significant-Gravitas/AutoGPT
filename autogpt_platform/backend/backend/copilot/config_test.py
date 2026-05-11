"""Unit tests for ChatConfig."""

import pytest

from .config import ChatConfig, _host_matches

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
    "ANTHROPIC_API_KEY",
    "CHAT_AUX_API_KEY",
    "CHAT_AUX_BASE_URL",
    "CHAT_DIRECT_ANTHROPIC_API_KEY",
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
    "CHAT_TITLE_MODEL",
    "CHAT_SIMULATION_MODEL",
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
    without each test having to repeat the override.  Also sets
    ``aux_api_key`` so the new aux-credentials validator passes by
    default — tests that target that specific check should pass
    ``aux_api_key=None`` (and clear ``api_key``/``base_url``) to opt
    in to its trip.
    """
    defaults: dict = {
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        "aux_api_key": "or-aux-key",
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
            # Aux key satisfies the aux-credential validator (this test
            # focuses on the SDK-vendor validator); without it the
            # default ``openai/gpt-4o-mini`` title model trips the new
            # aux check.
            aux_api_key="or-aux-key",
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
        # ``direct_anthropic_api_key`` + Anthropic ``title_model`` keep the
        # separate aux-client validator (``_validate_aux_client_for_direct_main``)
        # happy — without those, subscription mode + use_openrouter=False
        # would trip the aux-401 / non-Anthropic-title trap before the SDK
        # check we're targeting here runs.
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            direct_anthropic_api_key="sk-ant-test",
            use_claude_code_subscription=True,
            title_model="anthropic/claude-haiku-4-5",
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

    def test_fast_standard_model_also_validated(self):
        """Baseline (fast) tier slugs flow through the same
        ``normalize_model_for_transport``, so the validator must catch
        non-Anthropic fast-path slugs at boot — otherwise they'd fail
        per-turn at runtime when LD or env serves them."""
        with pytest.raises(Exception, match="fast_standard_model"):
            ChatConfig(
                use_openrouter=False,
                api_key=None,
                base_url=None,
                use_claude_code_subscription=False,
                thinking_standard_model="anthropic/claude-sonnet-4-6",
                thinking_advanced_model="anthropic/claude-opus-4-7",
                fast_standard_model="moonshotai/kimi-k2.6",
            )

    def test_bare_non_claude_slug_rejected(self):
        """Bare slugs without the ``vendor/`` prefix must start with
        ``claude-`` — otherwise the runtime ``normalize_model_for_transport``
        would 500 every request.  Pre-fix, a bare ``gpt-4o-mini`` slug
        slipped through the ``"/" not in value`` short-circuit."""
        with pytest.raises(Exception, match="thinking_standard_model"):
            ChatConfig(
                use_openrouter=False,
                api_key=None,
                base_url=None,
                use_claude_code_subscription=False,
                thinking_standard_model="gpt-4o-mini",
                thinking_advanced_model="anthropic/claude-opus-4-7",
                aux_api_key="or-aux-key",
            )

    def test_bare_claude_slug_accepted(self):
        """Bare ``claude-*`` slugs (the Anthropic Messages API's native
        form, e.g. ``claude-3-5-sonnet-20241022``) must pass."""
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            use_claude_code_subscription=False,
            thinking_standard_model="claude-sonnet-4-20250514",
            thinking_advanced_model="anthropic/claude-opus-4-7",
            aux_api_key="or-aux-key",
        )
        assert cfg.thinking_standard_model == "claude-sonnet-4-20250514"

    def test_fast_advanced_model_also_validated(self):
        with pytest.raises(Exception, match="fast_advanced_model"):
            ChatConfig(
                use_openrouter=False,
                api_key=None,
                base_url=None,
                use_claude_code_subscription=False,
                thinking_standard_model="anthropic/claude-sonnet-4-6",
                thinking_advanced_model="anthropic/claude-opus-4-7",
                fast_advanced_model="openai/gpt-5",
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
            aux_api_key="or-aux-key",
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


class TestMainClientCredentials:
    """``main_client_credentials`` picks the right (api_key, base_url) per transport."""

    def test_openrouter_mode_returns_main_creds(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.main_client_credentials == (
            "or-key",
            "https://openrouter.ai/api/v1",
        )

    def test_direct_mode_returns_anthropic_creds(self):
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key="or-key-leftover",
        )
        api_key, base_url = cfg.main_client_credentials
        assert api_key == "anthropic-key"
        assert base_url == "https://api.anthropic.com/v1/"


class TestAuxClientCredentials:
    """``aux_client_credentials`` keeps aux calls on OpenRouter even in direct mode."""

    def test_default_falls_back_to_main_creds(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            aux_api_key=None,
            aux_base_url=None,
        )
        # Single-key deployment: aux mirrors main.
        assert cfg.aux_client_credentials == (
            "or-key",
            "https://openrouter.ai/api/v1",
        )

    def test_leftover_or_env_does_not_force_aux_to_or_in_direct_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Migration scenario: deployment was on OR (env still has
        # ``OPEN_ROUTER_API_KEY``), now flipped to direct-Anthropic
        # without explicit aux split.  The aux key validator must NOT
        # auto-pull the leftover OR env (would 401 vs Anthropic URL);
        # aux must inherit ``main_client_credentials`` (Anthropic).
        monkeypatch.setenv("OPEN_ROUTER_API_KEY", "leftover-or-key")
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key=None,
            base_url=None,
            aux_api_key=None,
            aux_base_url=None,
            title_model="anthropic/claude-haiku-4-5",
        )
        api_key, base_url = cfg.aux_client_credentials
        assert api_key == "anthropic-key"
        assert base_url == "https://api.anthropic.com/v1/"
        assert cfg.aux_provider_label == "anthropic"

    def test_unset_aux_falls_back_to_main_client_creds_in_direct_mode(self):
        # Single-key direct-Anthropic deployment: aux env vars unset
        # → aux must inherit ``main_client_credentials`` (Anthropic),
        # NOT the raw ``api_key`` / ``base_url`` which default to OR
        # with no key.  Otherwise aux silently gets
        # ``(None, https://openrouter.ai/api/v1)`` and 401s on every
        # title call.
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key=None,
            base_url=None,
            aux_api_key=None,
            aux_base_url=None,
            title_model="anthropic/claude-haiku-4-5",
        )
        api_key, base_url = cfg.aux_client_credentials
        assert api_key == "anthropic-key"
        assert base_url == "https://api.anthropic.com/v1/"
        assert cfg.aux_uses_openrouter is False
        assert cfg.aux_provider_label == "anthropic"

    def test_explicit_aux_overrides_main(self):
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            aux_api_key="or-aux-key",
            aux_base_url="https://openrouter.ai/api/v1",
        )
        # Direct main + OR aux: aux client stays on OpenRouter.
        assert cfg.aux_client_credentials == (
            "or-aux-key",
            "https://openrouter.ai/api/v1",
        )

    def test_aux_uses_openrouter_true_for_or_url(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.aux_uses_openrouter is True

    def test_aux_uses_openrouter_false_for_other_url(self):
        # Anthropic-pointed aux + Anthropic title — valid pure-Anthropic
        # deployment.  Title must be Anthropic per the new validator.
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key="anthropic-key",
            base_url="https://api.anthropic.com/v1/",
            aux_base_url="https://api.anthropic.com/v1/",
            aux_api_key="anthropic-key",
            title_model="anthropic/claude-haiku-4-5",
        )
        assert cfg.aux_uses_openrouter is False


class TestAuxProviderLabel:
    """``aux_provider_label`` tracks the aux client's actual transport for cost-log rows."""

    def test_openrouter_url_returns_open_router(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.aux_provider_label == "open_router"

    def test_anthropic_url_returns_anthropic(self):
        # Single-key direct-Anthropic deployment: aux inherits the
        # main creds (Anthropic-pointed).  Cost row must land under
        # "anthropic", not the misleading "openai" fallback.
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key="anthropic-key",
            base_url="https://api.anthropic.com/v1/",
            aux_api_key=None,
            aux_base_url=None,
            title_model="anthropic/claude-haiku-4-5",
        )
        assert cfg.aux_provider_label == "anthropic"

    def test_other_url_returns_openai(self):
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="key",
            base_url="https://api.openai.com/v1",
            aux_api_key=None,
            aux_base_url=None,
        )
        assert cfg.aux_provider_label == "openai"


class TestAuxClientForDirectMainValidator:
    """``_validate_aux_client_for_direct_main`` fails fast at boot when
    direct-main mode + non-Anthropic aux model would land on a
    credential-less aux client."""

    def test_direct_main_with_non_anthropic_title_and_no_aux_key_raises(self):
        # OpenRouter off, no aux key, no fallback ``api_key`` → aux
        # client would 401 on every title call.  Validator must reject
        # at boot.  Pydantic wraps ValueError in ValidationError, so
        # match against ``Exception`` for parity with the SDK-vendor
        # validator tests above.
        with pytest.raises(Exception, match="title_model=.*is non-Anthropic"):
            _make_direct_safe_config(
                use_openrouter=False,
                direct_anthropic_api_key="anthropic-key",
                api_key=None,
                base_url=None,
                aux_api_key=None,
                aux_base_url=None,
                title_model="openai/gpt-4o-mini",
            )

    def test_direct_main_with_anthropic_title_passes(self):
        # Anthropic title model — aux can fall back to direct creds.
        # ``simulation_model`` is no longer validated here (it has its
        # own client acquisition path) so its default non-Anthropic
        # value doesn't trip the check.
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key=None,
            base_url=None,
            aux_api_key=None,
            aux_base_url=None,
            title_model="anthropic/claude-haiku-4-5",
        )
        assert cfg.title_model == "anthropic/claude-haiku-4-5"

    def test_simulation_model_not_validated(self):
        # Non-Anthropic ``simulation_model`` does NOT trip the
        # validator — the simulator uses its own platform-level OR key
        # (``util.clients.get_openai_client(prefer_openrouter=True)``)
        # which is independent of ChatConfig aux settings.
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            api_key=None,
            base_url=None,
            aux_api_key=None,
            aux_base_url=None,
            title_model="anthropic/claude-haiku-4-5",
            simulation_model="google/gemini-2.5-flash-lite",
        )
        assert cfg.simulation_model == "google/gemini-2.5-flash-lite"

    def test_direct_main_with_explicit_aux_key_passes(self):
        # Non-Anthropic title is fine when aux has its own creds.
        cfg = _make_direct_safe_config(
            use_openrouter=False,
            direct_anthropic_api_key="anthropic-key",
            aux_api_key="or-key",
            title_model="openai/gpt-4o-mini",
        )
        assert cfg.aux_api_key == "or-key"

    def test_openrouter_main_skips_validator(self):
        # OR main mode — aux follows main, no special check needed.
        cfg = ChatConfig(
            use_openrouter=True,
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
            aux_api_key=None,
            title_model="openai/gpt-4o-mini",
        )
        assert cfg.title_model == "openai/gpt-4o-mini"

    def test_or_main_with_explicit_anthropic_aux_and_non_anthropic_title_raises(
        self,
    ):
        # Sentry: even when ``use_openrouter=True`` for the main client,
        # an operator can still misconfigure aux to point at Anthropic
        # (e.g. setting CHAT_AUX_BASE_URL=https://api.anthropic.com).
        # In that shape, default ``title_model="openai/gpt-4o-mini"`` is
        # stripped to ``gpt-4o-mini`` and 404s on Anthropic.  Validator
        # must catch this regardless of the main flag.
        with pytest.raises(Exception, match="title_model=.*is non-Anthropic"):
            ChatConfig(
                use_openrouter=True,
                api_key="or-key",
                base_url="https://openrouter.ai/api/v1",
                aux_api_key="sk-ant-aux",
                aux_base_url="https://api.anthropic.com/v1/",
                title_model="openai/gpt-4o-mini",
            )

    def test_subscription_mode_no_aux_no_direct_key_raises(self):
        # Subscription mode (SDK uses OAuth, no api_key needed) but the
        # aux client still runs the baseline OpenAI-compat path for title
        # generation.  When CHAT_AUX_API_KEY and CHAT_DIRECT_ANTHROPIC_API_KEY
        # are both unset, aux_client_credentials returns
        # ``(None, api.anthropic.com)`` and 401s every title call.
        # Validator must reject at boot rather than silently 401 forever.
        with pytest.raises(Exception, match="Subscription mode"):
            ChatConfig(
                use_openrouter=False,
                api_key=None,
                base_url=None,
                aux_api_key=None,
                aux_base_url=None,
                direct_anthropic_api_key=None,
                use_claude_code_subscription=True,
            )

    def test_subscription_mode_with_direct_anthropic_key_passes(self):
        # Subscription + direct_anthropic_api_key set — aux can reach
        # Anthropic with valid creds.  Title must be on Anthropic for
        # the inheritance path to be valid, which is the same constraint
        # as the direct-mode-non-subscription case.
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            aux_api_key=None,
            aux_base_url=None,
            direct_anthropic_api_key="sk-ant-test",
            use_claude_code_subscription=True,
            title_model="anthropic/claude-haiku-4-5",
        )
        assert cfg.use_claude_code_subscription is True

    def test_subscription_mode_with_explicit_aux_key_passes(self):
        # Subscription + explicit aux key — aux routes to its own
        # endpoint (typically OpenRouter for cheap title model).
        cfg = ChatConfig(
            use_openrouter=False,
            api_key=None,
            base_url=None,
            aux_api_key="or-key",
            aux_base_url="https://openrouter.ai/api/v1",
            direct_anthropic_api_key=None,
            use_claude_code_subscription=True,
            title_model="openai/gpt-4o-mini",
        )
        assert cfg.aux_api_key == "or-key"

    def test_direct_main_with_aux_base_url_no_key_raises_even_for_anthropic_title(
        self,
    ):
        # Operator typo: ``CHAT_AUX_BASE_URL`` set but ``CHAT_AUX_API_KEY``
        # forgotten.  Even when ``title_model`` is Anthropic the aux client
        # would still route to the explicit ``aux_base_url`` with no creds
        # and 401 every title call — the Anthropic-title short-circuit
        # only applies when aux falls back to direct Anthropic, which
        # requires ``aux_base_url`` to be unset.
        with pytest.raises(Exception, match="CHAT_AUX_BASE_URL"):
            _make_direct_safe_config(
                use_openrouter=False,
                direct_anthropic_api_key="anthropic-key",
                api_key=None,
                base_url=None,
                aux_api_key=None,
                aux_base_url="https://openrouter.ai/api/v1",
                title_model="anthropic/claude-haiku-4-5",
            )


class TestHostMatches:
    """``_host_matches`` parses the URL and compares the actual host —
    rejects substring tricks that pass the loose ``"x" in url`` check.
    """

    def test_exact_host(self):
        assert _host_matches("https://api.anthropic.com/v1/", "anthropic.com")
        assert _host_matches("https://openrouter.ai/api/v1", "openrouter.ai")

    def test_subdomain_matches_via_dot_suffix(self):
        assert _host_matches("https://api.openrouter.ai/v1", "openrouter.ai")

    def test_substring_in_path_does_not_match(self):
        # The loose ``"anthropic.com" in base_url`` check would say yes
        # to attacker-controlled URLs that embed the suffix in the path.
        assert not _host_matches(
            "https://attacker.example/anthropic.com/v1", "anthropic.com"
        )
        assert not _host_matches("https://evil.test/openrouter.ai/api", "openrouter.ai")

    def test_substring_in_host_without_dot_does_not_match(self):
        # ``api.anthropic.com.attacker.test`` ends with ``.test`` not
        # ``.anthropic.com`` — the dot-suffix check rejects the spoof.
        assert not _host_matches(
            "https://anthropic.com.attacker.test/v1", "anthropic.com"
        )

    def test_empty_url(self):
        assert not _host_matches(None, "anthropic.com")
        assert not _host_matches("", "anthropic.com")

    def test_case_insensitive(self):
        assert _host_matches("https://API.ANTHROPIC.COM/", "anthropic.com")
