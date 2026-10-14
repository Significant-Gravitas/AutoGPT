"""Unit tests for Moonshot pricing and cache-control helpers."""

from __future__ import annotations

import pytest

from backend.copilot.moonshot import (
    is_moonshot_model,
    moonshot_context_window,
    moonshot_cost_usd,
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
            "moonshotai/kimi-k3",
            "moonshotai/kimi-k4",  # Future SKU must match transparently.
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
    """Shared K2.x default plus per-slug overrides for split-priced SKUs."""

    def test_moonshot_default_rate(self) -> None:
        assert rate_card_usd("moonshotai/kimi-k2.6") == (0.60, 2.80)

    def test_kimi_k3_premium_override(self) -> None:
        # K3 is split-priced ($3.00/$15.00 per Mtok) — the override must
        # win over the shared K2.x default.  Sourced from the catalog
        # entry's provider_*_usd_per_1m, which tracks OpenRouter's
        # published price for moonshotai/kimi-k3 (/api/v1/models,
        # re-checked 2026-09-25).  A same-day snapshot had briefly shown
        # $0.8845/$10.5346, but that had already drifted back to the
        # pinned $3.00/$15.00 rate by the time of this check.
        assert rate_card_usd("moonshotai/kimi-k3") == (3.00, 15.00)

    def test_future_moonshot_sku_inherits_default(self) -> None:
        # Verifies the prefix-based fallback — new SKUs don't need a code
        # edit to get a reasonable rate card.
        assert rate_card_usd("moonshotai/kimi-k4") == (0.60, 2.80)

    def test_non_moonshot_returns_none(self) -> None:
        assert rate_card_usd("anthropic/claude-sonnet-4.6") is None
        assert rate_card_usd("openai/gpt-4o") is None


class TestMoonshotCostUsd:
    """The Moonshot rate card prices a Kimi turn; any other model is not
    Moonshot's to price, so the answer is None."""

    def test_moonshot_prices_from_the_rate_card(self) -> None:
        """A 29.5K-prompt Kimi turn lands at ~$0.018 on the Moonshot rate
        card, not the ~$0.09 Anthropic rates would give."""
        cost = moonshot_cost_usd(
            model="moonshotai/kimi-k2.6",
            prompt_tokens=29564,
            completion_tokens=78,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )
        expected = (29564 * 0.60 + 78 * 2.80) / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)
        assert cost is not None and 0.017 < cost < 0.019

    @pytest.mark.parametrize(
        "model", ["anthropic/claude-sonnet-4.6", "deepseek/deepseek-v3", None]
    )
    def test_other_models_are_not_moonshots_to_price(self, model) -> None:
        assert (
            moonshot_cost_usd(
                model=model,
                prompt_tokens=29564,
                completion_tokens=78,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )
            is None
        )

    def test_cache_tokens_priced_at_input_rate(self) -> None:
        """OpenRouter's Moonshot endpoints don't expose a discounted
        cached-input price — cache_read / cache_creation tokens are
        priced at the full input rate.  The reconcile path has the
        authoritative discount for turns where Moonshot's cache engaged."""
        cost = moonshot_cost_usd(
            model="moonshotai/kimi-k2.6",
            prompt_tokens=1000,
            completion_tokens=0,
            cache_read_tokens=5000,
            cache_creation_tokens=2000,
        )
        expected = (1000 + 5000 + 2000) * 0.60 / 1_000_000
        assert cost == pytest.approx(expected, rel=1e-9)


class TestMoonshotContextWindow:
    """The catalog is the only place the real Kimi window is written down."""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("moonshotai/kimi-k2.5", 262_144),
            ("moonshotai/kimi-k2.6", 262_144),
            ("moonshotai/kimi-k2-thinking", 262_144),
            ("moonshotai/kimi-k3", 1_048_576),
        ],
    )
    def test_reads_the_catalog_window(self, model, expected) -> None:
        assert moonshot_context_window(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "moonshotai/kimi-k9-unreleased",
            "anthropic/claude-sonnet-5",
            "",
            None,
        ],
    )
    def test_none_for_unlisted_and_non_moonshot(self, model) -> None:
        assert moonshot_context_window(model) is None
