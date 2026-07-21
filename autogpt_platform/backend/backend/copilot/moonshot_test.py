"""Unit tests for Moonshot pricing and cache-control helpers."""

from __future__ import annotations

import pytest

from backend.copilot.moonshot import (
    is_moonshot_model,
    moonshot_supports_cache_control,
    override_cost_usd,
    rate_card_usd,
)


class TestIsMoonshotModel:
    """Prefix detection covers every Moonshot SKU without a slug list."""

    @pytest.mark.parametrize(
        "model",
        [
            "moonshotai/kimi-k2.6",
            "moonshotai/kimi-k2-thinking",
            "moonshotai/kimi-k2.5",
            "moonshotai/kimi-k2",
            "moonshotai/kimi-k3.0",  # Future SKU must match transparently.
        ],
    )
    def test_moonshot_slugs_match(self, model: str) -> None:
        assert is_moonshot_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-opus-4.7",
            "openai/gpt-4o",
            "google/gemini-2.5-flash",
            "xai/grok-4",
            "deepseek/deepseek-v3",
            "",  # Empty string — not Moonshot.
        ],
    )
    def test_non_moonshot_slugs_do_not_match(self, model: str) -> None:
        assert is_moonshot_model(model) is False

    @pytest.mark.parametrize("model", [None, 123, ["moonshotai/kimi-k2.6"]])
    def test_non_string_returns_false(self, model) -> None:
        # Type-robust: never raise on unexpected types; callers pass None.
        assert is_moonshot_model(model) is False


class TestRateCardUsd:
    """Rate card defaults to the shared Moonshot price for every SKU."""

    def test_moonshot_default_rate(self) -> None:
        assert rate_card_usd("moonshotai/kimi-k2.6") == (0.60, 2.80)

    def test_future_moonshot_sku_inherits_default(self) -> None:
        # Verifies the prefix-based fallback — new SKUs don't need a code
        # edit to get a reasonable rate card.
        assert rate_card_usd("moonshotai/kimi-k3.0") == (0.60, 2.80)

    def test_non_moonshot_returns_none(self) -> None:
        assert rate_card_usd("anthropic/claude-sonnet-4.6") is None
        assert rate_card_usd("openai/gpt-4o") is None


class TestOverrideCostUsd:
    """Rate-card override replaces the CLI's Sonnet-rate estimate for
    Moonshot turns; Anthropic and unknown slugs pass through unchanged."""

    def test_moonshot_recomputes_from_rate_card(self) -> None:
        """A 29.5K-prompt Kimi turn should land at ~$0.018 on the
        Moonshot rate card, not the CLI's $0.09 Sonnet-rate estimate."""
        recomputed = override_cost_usd(
            model="moonshotai/kimi-k2.6",
            sdk_reported_usd=0.089862,  # What the CLI reported (Sonnet price).
            prompt_tokens=29564,
            completion_tokens=78,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
        expected = (29564 * 0.60 + 78 * 2.80) / 1_000_000
        assert recomputed == pytest.approx(expected, rel=1e-9)
        assert 0.017 < recomputed < 0.019  # Sanity against Moonshot's rate card.

    def test_anthropic_passes_through(self) -> None:
        """Anthropic slugs are priced accurately by the CLI already —
        the override returns the SDK number unchanged."""
        assert (
            override_cost_usd(
                model="anthropic/claude-sonnet-4.6",
                sdk_reported_usd=0.089862,
                prompt_tokens=29564,
                completion_tokens=78,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )
            == 0.089862
        )

    def test_unknown_non_moonshot_passes_through(self) -> None:
        """A non-Moonshot, non-Anthropic slug falls back to the SDK value
        — best-effort rather than leaking a zero or a wrong rate card."""
        assert (
            override_cost_usd(
                model="deepseek/deepseek-v3",
                sdk_reported_usd=0.05,
                prompt_tokens=10_000,
                completion_tokens=500,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )
            == 0.05
        )

    def test_none_model_passes_through(self) -> None:
        """Subscription mode sets model=None — return the SDK value."""
        assert (
            override_cost_usd(
                model=None,
                sdk_reported_usd=0.07,
                prompt_tokens=100,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )
            == 0.07
        )

    def test_cache_tokens_priced_at_input_rate(self) -> None:
        """OpenRouter's Moonshot endpoints don't expose a discounted
        cached-input price — cache_read / cache_creation tokens are
        priced at the full input rate.  The reconcile path has the
        authoritative discount for turns where Moonshot's cache engaged."""
        recomputed = override_cost_usd(
            model="moonshotai/kimi-k2.6",
            sdk_reported_usd=0.5,
            prompt_tokens=1000,
            completion_tokens=0,
            cache_read_tokens=5000,
            cache_creation_tokens=2000,
        )
        expected = (1000 + 5000 + 2000) * 0.60 / 1_000_000
        assert recomputed == pytest.approx(expected, rel=1e-9)


class TestSupportsCacheControl:
    """Gate for emitting ``cache_control: {type: ephemeral}`` on message
    blocks.  True for Moonshot (Anthropic-compat endpoint accepts it)
    and False for everything else this module knows about — Anthropic
    callers use their own ``_is_anthropic_model`` check which is
    combined with this one into a wider gate."""

    def test_moonshot_supports_cache_control(self) -> None:
        assert moonshot_supports_cache_control("moonshotai/kimi-k2.6") is True

    def test_future_moonshot_sku_supports_cache_control(self) -> None:
        assert moonshot_supports_cache_control("moonshotai/kimi-k3.0") is True

    @pytest.mark.parametrize(
        "model",
        [
            "openai/gpt-4o",
            "google/gemini-2.5-flash",
            "xai/grok-4",
            "deepseek/deepseek-v3",
            "",
            None,
        ],
    )
    def test_non_moonshot_does_not_support_cache_control(self, model) -> None:
        assert moonshot_supports_cache_control(model) is False
