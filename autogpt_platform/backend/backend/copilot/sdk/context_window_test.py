"""Tests for the per-route SDK context window + trigger resolver."""

import pytest

from backend.copilot.config import (
    CLAUDE_ENGINE_CONTEXT_WINDOW,
    CLI_DEFAULT_CONTEXT_WINDOW,
    CODEX_ENGINE_AUTOCOMPACT_PCT,
    CODEX_ENGINE_CONTEXT_WINDOW,
    LOCAL_CONTEXT_FALLBACK,
    ChatConfig,
)
from backend.copilot.sdk.context_window import autocompact_pct, pinned_context_window


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
    @pytest.mark.parametrize("transport", ["direct_anthropic", "subscription"])
    def test_claude_engine_routes_default_to_1m(self, transport):
        if transport == "subscription":
            cfg = _make_config(use_claude_code_subscription=True)
        else:
            cfg = _make_config()
        assert cfg.transport.name == transport
        assert (
            pinned_context_window(cfg, "anthropic/claude-sonnet-5", codex_route=False)
            == CLAUDE_ENGINE_CONTEXT_WINDOW
            == 1_000_000
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
        assert pinned_context_window(cfg, None, codex_route=False) == 1_000_000


class TestPinnedContextWindowLocal:
    def _local_config(self, **overrides):
        defaults = {
            "use_local": True,
            "api_key": "ollama",
            "base_url": "http://h:11434/v1",
        }
        defaults.update(overrides)
        return _make_config(**defaults)

    def test_probed_window_pins_local_route(self):
        cfg = self._local_config()
        assert cfg.transport.name == "local"
        assert (
            pinned_context_window(
                cfg, "llama3.2:3b", codex_route=False, local_window=131_072
            )
            == 131_072
        )

    def test_unprobed_local_falls_back_to_blind_constant(self):
        cfg = self._local_config()
        assert (
            pinned_context_window(cfg, "llama3.2:3b", codex_route=False)
            == LOCAL_CONTEXT_FALLBACK
            == 32_768
        )

    def test_explicit_window_wins_over_probe(self):
        cfg = self._local_config(claude_agent_context_window=300_000)
        assert (
            pinned_context_window(
                cfg, "llama3.2:3b", codex_route=False, local_window=131_072
            )
            == 300_000
        )

    def test_moonshot_cap_still_applies_on_local(self):
        """The Moonshot min() is route-agnostic — a moonshot-shaped slug
        over a huge probed window still pins to the SKU's real window."""
        cfg = self._local_config()
        assert (
            pinned_context_window(
                cfg, "moonshotai/kimi-k2.5", codex_route=False, local_window=1_000_000
            )
            == 262_144
        )


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

    @pytest.mark.parametrize("model", ["llama3.2:3b", "anthropic/claude-sonnet-5"])
    def test_local_returns_zero_for_any_slug(self, model):
        """Operator hardware has no Anthropic cache costs to cap, so the
        local route sets no trigger override — not even the Sonnet-5
        scaling applies (the local check runs first)."""
        cfg = _make_config(
            use_local=True, api_key="ollama", base_url="http://h:11434/v1"
        )
        assert cfg.transport.name == "local"
        assert autocompact_pct(cfg, model, codex_route=False) == 0
