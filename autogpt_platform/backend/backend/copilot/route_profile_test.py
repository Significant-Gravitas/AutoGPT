"""Each route's profile: model names, window, trigger, credential, thinking."""

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.route_profile import (
    THINKING_BUDGET_TOKENS,
    codex_profile,
    normalize_model,
    platform_profile,
    route_trigger_pct,
    route_window,
    user_key_profile,
)
from backend.copilot.route_table import ROUTES
from backend.copilot.sdk.context_window import CodexEngineWindow


class TestModelNames:
    @pytest.mark.parametrize("route", ["openrouter", "local", "codex"])
    def test_kept_as_configured(self, route):
        assert normalize_model(ROUTES[route], "vendor/model-4.6") == "vendor/model-4.6"

    @pytest.mark.parametrize("route", ["direct_anthropic", "subscription"])
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("anthropic/claude-opus-4.6", "claude-opus-4-6"),
            ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ],
    )
    def test_native_claude_ids_on_the_anthropic_routes(self, route, raw, expected):
        assert normalize_model(ROUTES[route], raw) == expected

    @pytest.mark.parametrize("raw", ["moonshotai/kimi-k2.6", "gpt-4o-mini"])
    def test_other_vendors_are_refused_on_the_anthropic_routes(self, raw):
        with pytest.raises(ValueError, match="Anthropic models only"):
            normalize_model(ROUTES["direct_anthropic"], raw)


class TestWindow:
    def test_route_defaults(self):
        assert (
            route_window(ROUTES["openrouter"], "anthropic/claude-sonnet-5") == 200_000
        )
        assert route_window(ROUTES["direct_anthropic"], "claude-sonnet-5") == 1_000_000
        assert route_window(ROUTES["subscription"], "claude-sonnet-5") == 200_000

    def test_codex_takes_the_account_window_else_the_engine_default(self):
        engine = CodexEngineWindow(context_window=400_000)
        assert (
            route_window(ROUTES["codex"], "gpt-6-astra", codex_engine=engine) == 400_000
        )
        assert route_window(ROUTES["codex"], "gpt-6-astra") == 272_000

    def test_local_takes_the_operators_window(self):
        assert route_window(ROUTES["local"], "qwen3", local_window=131_072) == 131_072
        with pytest.raises(ValueError, match="needs its model's window"):
            route_window(ROUTES["local"], "qwen3")

    def test_moonshot_is_held_to_its_sku_window(self):
        from backend.copilot.moonshot import moonshot_context_window

        model = "moonshotai/kimi-k2.6"
        expected = min(200_000, moonshot_context_window(model) or 200_000)
        assert route_window(ROUTES["openrouter"], model) == expected


class TestTrigger:
    def test_platform_routes_compact_at_half(self):
        assert (
            route_trigger_pct(ROUTES["openrouter"], "anthropic/claude-opus-4-6") == 50
        )
        assert route_trigger_pct(ROUTES["direct_anthropic"], "claude-opus-4-6") == 50

    def test_sonnet_5_scaled_for_its_tokenizer(self):
        assert (
            route_trigger_pct(ROUTES["openrouter"], "anthropic/claude-sonnet-5") == 65
        )

    def test_subscription_fills_its_window(self):
        assert route_trigger_pct(ROUTES["subscription"], "claude-sonnet-5") == 90

    def test_moonshot_keeps_the_clis_own_trigger(self):
        assert route_trigger_pct(ROUTES["openrouter"], "moonshotai/kimi-k2.6") is None

    def test_codex_follows_the_account(self):
        engine = CodexEngineWindow(
            context_window=400_000, auto_compact_token_limit=200_000
        )
        assert (
            route_trigger_pct(ROUTES["codex"], "gpt-6-astra", codex_engine=engine) == 50
        )
        assert route_trigger_pct(ROUTES["codex"], "gpt-6-astra") == 90


class TestPlatformProfile:
    def test_openrouter(self):
        cfg = ChatConfig(
            platform_route="openrouter",
            openrouter_api_key="sk-or",
            openrouter_base_url="https://openrouter.ai/api/v1",
        )
        profile = platform_profile(cfg, "anthropic/claude-sonnet-5", "env")
        assert profile.name == "openrouter"
        assert profile.base_url == "https://openrouter.ai/api"
        assert profile.api_key is not None
        assert profile.api_key.get_secret_value() == "sk-or"
        assert "sk-or" not in repr(profile)
        assert profile.platform_pays and profile.prices_tokens
        assert profile.thinking_effort == "high"
        assert profile.thinking_budget == THINKING_BUDGET_TOKENS
        assert profile.compaction_target_tokens() == 130_000 - 20_000

    def test_subscription_is_the_users_own_plan(self):
        profile = platform_profile(
            ChatConfig(platform_route="subscription"), "claude-sonnet-5", "env"
        )
        assert not profile.platform_pays
        assert profile.prices_tokens
        assert profile.api_key is None

    def test_local_carries_its_adapter_and_window(self):
        cfg = ChatConfig(
            platform_route="local",
            local_base_url="http://ollama:11434/v1",
            local_adapter="openai_chat",
            local_context_tokens=131_072,
        )
        profile = platform_profile(cfg, "qwen3:8b", "env")
        assert profile.window == 131_072
        assert profile.trigger_pct == 90
        assert profile.gateway_adapter == "openai_chat"
        assert profile.pins_max_context
        assert profile.thinking_effort is None

    def test_local_records_no_price_for_the_operators_model(self):
        """No rate card knows a self-hosted model, so its turns carry no
        estimate rather than one at the fallback rates."""
        cfg = ChatConfig(
            platform_route="local",
            local_base_url="http://ollama:11434/v1",
            local_adapter="openai_chat",
            local_context_tokens=131_072,
        )
        profile = platform_profile(cfg, "qwen3:8b", "env")
        assert not profile.platform_pays
        assert not profile.prices_tokens

    def test_disabled_has_no_platform_profile(self):
        with pytest.raises(RuntimeError, match="CHAT_PLATFORM_ROUTE=disabled"):
            platform_profile(ChatConfig(), "anthropic/claude-sonnet-5", "env")


class TestCodexProfile:
    def test_no_rate_card_no_usd_ceiling_no_anthropic_thinking(self):
        profile = codex_profile(
            "gpt-6-astra", effort="xhigh", routing_source="catalog", codex_engine=None
        )
        assert profile.window_source == "engine-default"
        assert not profile.prices_tokens
        assert not profile.platform_pays
        assert not profile.row.usd_budget
        assert profile.thinking_effort is None
        assert profile.gateway_adapter == "codex_responses"


class TestUserKeyProfile:
    def test_openrouter_key_goes_to_openrouter(self):
        profile = user_key_profile("open_router", "sk-user", "claude-sonnet-4-6")
        assert profile.name == "openrouter"
        assert profile.base_url == "https://openrouter.ai/api"

    def test_any_other_key_goes_to_anthropic(self):
        profile = user_key_profile("anthropic", "sk-ant", "claude-sonnet-4-6")
        assert profile.name == "direct_anthropic"
        assert profile.base_url == ""
