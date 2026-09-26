"""Tests for the per-route SDK context window + trigger resolver."""

import pytest

from backend.copilot.config import (
    CLAUDE_ENGINE_CONTEXT_WINDOW,
    CLI_DEFAULT_CONTEXT_WINDOW,
    CODEX_ENGINE_AUTOCOMPACT_PCT,
    CODEX_ENGINE_CONTEXT_WINDOW,
    ChatConfig,
)
from backend.copilot.sdk.context_window import (
    CodexEngineWindow,
    autocompact_pct,
    cli_autocompact_threshold,
    compaction_target_tokens,
    pinned_context_window,
    retry_target_tokens,
)


def _make_config(**overrides) -> ChatConfig:
    """ChatConfig with direct-Anthropic-shaped defaults, like env_test's."""
    defaults = {
        "use_claude_code_subscription": False,
        "use_openrouter": False,
        # Explicit: init kwargs beat both process env and the .env file,
        # so a local-flavored developer .env can't flip the transport.
        "use_local": False,
        "api_key": None,
        "base_url": None,
        # Fast tiers pinned like the thinking tiers: the direct-Anthropic
        # vendor validator rejects non-anthropic slugs, and a
        # local-flavored .env would otherwise leak llama slugs in here.
        "fast_standard_model": "anthropic/claude-sonnet-5",
        "fast_advanced_model": "anthropic/claude-opus-4-8",
        "thinking_standard_model": "anthropic/claude-sonnet-4-6",
        "thinking_advanced_model": "anthropic/claude-opus-4-7",
        "claude_agent_autocompact_pct_override": 50,
        "claude_agent_context_window": None,
        "aux_api_key": "or-aux-key",
    }
    defaults.update(overrides)
    return ChatConfig(**defaults)


def _openrouter_config(**overrides) -> ChatConfig:
    defaults = {
        "use_openrouter": True,
        "api_key": "sk-or-test",
        "base_url": "https://openrouter.ai/api/v1",
    }
    defaults.update(overrides)
    return _make_config(**defaults)


class TestPinnedContextWindow:
    def test_direct_anthropic_defaults_to_1m(self):
        cfg = _make_config()
        assert cfg.transport.name == "direct_anthropic"
        assert (
            pinned_context_window(cfg, "anthropic/claude-sonnet-5", codex_route=False)
            == CLAUDE_ENGINE_CONTEXT_WINDOW
            == 1_000_000
        )

    def test_subscription_pins_200k_for_the_plan_limit(self):
        """The engine would run 1M, but these turns draw on the subscriber's
        own plan; 200K keeps a long chat from draining a usage window."""
        cfg = _make_config(use_claude_code_subscription=True)
        assert cfg.transport.name == "subscription"
        assert (
            pinned_context_window(cfg, "anthropic/claude-sonnet-5", codex_route=False)
            == CLI_DEFAULT_CONTEXT_WINDOW
            == 200_000
        )

    def test_platform_route_defaults_to_cli_default(self):
        cfg = _openrouter_config()
        assert cfg.transport.name == "openrouter"
        assert (
            pinned_context_window(cfg, "anthropic/claude-sonnet-5", codex_route=False)
            == CLI_DEFAULT_CONTEXT_WINDOW
            == 200_000
        )

    @pytest.mark.parametrize(
        "model", ["gpt-6-astra", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-9.9-zzz", None]
    )
    def test_codex_route_takes_engine_default_for_any_slug(self, model):
        """272K for listed, unlisted, and missing slugs alike — the profile
        is bypassed, so the transport underneath is irrelevant."""
        for cfg in (
            _make_config(),
            _openrouter_config(),
            _make_config(
                use_local=True, api_key="ollama", base_url="http://h:11434/v1"
            ),
        ):
            assert (
                pinned_context_window(cfg, model, codex_route=True)
                == CODEX_ENGINE_CONTEXT_WINDOW
                == 272_000
            )

    def test_codex_route_skips_moonshot_cap(self):
        """A moonshot-shaped slug over the gateway is still a Codex turn."""
        cfg = _openrouter_config()
        assert (
            pinned_context_window(cfg, "moonshotai/kimi-k2.5", codex_route=True)
            == 272_000
        )

    def test_explicit_window_wins_on_every_route(self):
        cfgs = [
            _make_config(claude_agent_context_window=300_000),
            _make_config(
                use_claude_code_subscription=True,
                claude_agent_context_window=300_000,
            ),
            _openrouter_config(claude_agent_context_window=300_000),
        ]
        for cfg in cfgs:
            assert (
                pinned_context_window(
                    cfg, "anthropic/claude-sonnet-5", codex_route=False
                )
                == 300_000
            )
        cfg = _openrouter_config(claude_agent_context_window=300_000)
        assert pinned_context_window(cfg, "gpt-6-astra", codex_route=True) == 300_000

    def test_moonshot_capped_at_sku_window(self):
        cfg = _openrouter_config(thinking_standard_model="moonshotai/kimi-k2.5")
        assert (
            pinned_context_window(cfg, "moonshotai/kimi-k2.5", codex_route=False)
            == 200_000
        )
        cfg = _openrouter_config(
            thinking_standard_model="moonshotai/kimi-k2.5",
            claude_agent_context_window=1_000_000,
        )
        assert (
            pinned_context_window(cfg, "moonshotai/kimi-k2.5", codex_route=False)
            == 262_144
        )

    def test_moonshot_unlisted_sku_falls_back_to_cli_default(self):
        cfg = _openrouter_config(claude_agent_context_window=1_000_000)
        assert (
            pinned_context_window(cfg, "moonshotai/kimi-k3.0", codex_route=False)
            == 200_000
        )

    def test_none_model_uses_transport_default(self):
        """Subscription standard tier resolves no slug (CLI picks) — the pin
        still applies."""
        cfg = _make_config(use_claude_code_subscription=True)
        assert pinned_context_window(cfg, None, codex_route=False) == 200_000


class TestAutocompactPct:
    def test_codex_route_uses_engine_trigger(self):
        cfg = _openrouter_config()
        assert (
            autocompact_pct(cfg, "gpt-6-astra", codex_route=True)
            == CODEX_ENGINE_AUTOCOMPACT_PCT
            == 90
        )

    def test_codex_zero_pct_omits(self):
        cfg = _openrouter_config(claude_agent_autocompact_pct_override=0)
        assert autocompact_pct(cfg, "gpt-6-astra", codex_route=True) == 0

    def test_non_codex_uses_configured_pct(self):
        cfg = _make_config()
        assert (
            autocompact_pct(cfg, "anthropic/claude-opus-4-7", codex_route=False) == 50
        )

    def test_sonnet_5_scales_on_non_codex_routes(self):
        cfg = _make_config()
        assert (
            autocompact_pct(cfg, "anthropic/claude-sonnet-5", codex_route=False) == 65
        )
        cfg = _make_config(claude_agent_autocompact_pct_override=70)
        assert (
            autocompact_pct(cfg, "anthropic/claude-sonnet-5", codex_route=False) == 90
        )


class TestCodexEngineWindow:
    def test_pin_prefers_the_account_window(self):
        cfg = _openrouter_config()
        engine = CodexEngineWindow(
            context_window=400_000, auto_compact_token_limit=360_000
        )
        assert (
            pinned_context_window(
                cfg, "gpt-6-astra", codex_route=True, codex_engine=engine
            )
            == 400_000
        )
        assert (
            pinned_context_window(cfg, "gpt-6-astra", codex_route=True)
            == CODEX_ENGINE_CONTEXT_WINDOW
        )

    def test_explicit_config_still_wins(self):
        cfg = _openrouter_config(claude_agent_context_window=300_000)
        engine = CodexEngineWindow(context_window=400_000)
        assert (
            pinned_context_window(
                cfg, "gpt-6-astra", codex_route=True, codex_engine=engine
            )
            == 300_000
        )

    @pytest.mark.parametrize(
        "window, limit, expected",
        [
            (272_000, 244_800, 90),
            (400_000, 200_000, 50),
            (400_000, 399_000, 90),  # held at the CLI's usable ceiling
            (400_000, None, CODEX_ENGINE_AUTOCOMPACT_PCT),
            (400_000, 400_000, CODEX_ENGINE_AUTOCOMPACT_PCT),  # not a trigger
            (400_000, 0, CODEX_ENGINE_AUTOCOMPACT_PCT),
        ],
    )
    def test_trigger_from_the_account_limit(self, window, limit, expected):
        cfg = _openrouter_config()
        engine = CodexEngineWindow(
            context_window=window, auto_compact_token_limit=limit
        )
        assert (
            autocompact_pct(cfg, "gpt-6-astra", codex_route=True, codex_engine=engine)
            == expected
        )

    def test_configured_zero_still_disables(self):
        cfg = _openrouter_config(claude_agent_autocompact_pct_override=0)
        engine = CodexEngineWindow(
            context_window=272_000, auto_compact_token_limit=244_800
        )
        assert (
            autocompact_pct(cfg, "gpt-6-astra", codex_route=True, codex_engine=engine)
            == 0
        )

    def test_every_budget_follows_the_account_window(self):
        cfg = _openrouter_config()
        engine = CodexEngineWindow(
            context_window=400_000, auto_compact_token_limit=360_000
        )
        kw = {"codex_route": True, "codex_engine": engine}
        assert cli_autocompact_threshold(cfg, "gpt-6-astra", **kw) == 360_000
        assert compaction_target_tokens(cfg, "gpt-6-astra", **kw) == 340_000
        assert retry_target_tokens(cfg, "gpt-6-astra", **kw) == (100_000, 5_000)
