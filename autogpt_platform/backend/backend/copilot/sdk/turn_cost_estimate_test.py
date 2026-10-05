"""Tests for the catalog-priced fallback cost of an SDK turn.

Token counts and costs come from prod ``[cost-record]`` / ``Token usage:``
log lines for 2026-09-25..10-03 (CLI 2.1.284 over OpenRouter).
"""

from __future__ import annotations

import pytest

from backend.copilot.sdk.turn_cost_estimate import (
    TokenUsage,
    estimate_cost_usd,
    fallback_turn_cost,
    token_usage_from_sdk,
)

SONNET_5_5 = "anthropic/claude-sonnet-5-5"


def _usage(uncached: int, read: int, create: int, out: int) -> TokenUsage:
    return TokenUsage(
        input_tokens=uncached,
        cache_read_tokens=read,
        cache_creation_tokens=create,
        output_tokens=out,
    )


class TestEstimateCostUsd:
    def test_prices_each_token_kind_at_the_catalog_rate(self):
        # Sonnet 5.5: $2 in, $0.20 cache read, $2.50 cache write, $10 out.
        cost = estimate_cost_usd(SONNET_5_5, _usage(1_000_000, 0, 0, 0))
        assert cost == pytest.approx(2.0)
        cost = estimate_cost_usd(SONNET_5_5, _usage(0, 1_000_000, 0, 0))
        assert cost == pytest.approx(0.2)
        cost = estimate_cost_usd(SONNET_5_5, _usage(0, 0, 1_000_000, 0))
        assert cost == pytest.approx(2.5)
        cost = estimate_cost_usd(SONNET_5_5, _usage(0, 0, 0, 1_000_000))
        assert cost == pytest.approx(10.0)

    def test_accepts_bare_and_dotted_slugs(self):
        usage = _usage(1000, 2000, 3000, 400)
        expected = estimate_cost_usd(SONNET_5_5, usage)
        assert expected is not None
        assert estimate_cost_usd("claude-sonnet-5-5", usage) == expected
        assert estimate_cost_usd("anthropic/claude-sonnet-5.5", usage) == expected

    def test_unknown_model_has_no_estimate(self):
        assert estimate_cost_usd("acme/unknown-model", _usage(1, 1, 1, 1)) is None
        assert estimate_cost_usd(None, _usage(1, 1, 1, 1)) is None

    def test_model_without_cache_rates_prices_cache_at_input_rate(self):
        # Kimi K2.6 carries no cache credit rates in the catalog.
        cached = estimate_cost_usd("moonshotai/kimi-k2.6", _usage(0, 1000, 1000, 0))
        uncached = estimate_cost_usd("moonshotai/kimi-k2.6", _usage(2000, 0, 0, 0))
        assert cached is not None
        assert cached == pytest.approx(uncached)

    # (model, uncached, cache_read, cache_create, output, OpenRouter real $)
    @pytest.mark.parametrize(
        "model,tokens,real",
        [
            (SONNET_5_5, (292, 116438, 72450, 1947), 0.224467),
            (SONNET_5_5, (51, 91990, 4278, 1231), 0.041505),
            (SONNET_5_5, (167, 189488, 48145, 2106), 0.179654),
            ("anthropic/claude-opus-5", (2581, 32394, 36586, 819), 0.278239),
            ("anthropic/claude-opus-5", (2966, 354835, 29459, 9699), 0.618841),
            ("anthropic/claude-opus-5", (282, 153955, 4734, 1647), 0.14915),
            # The catalog prices Sonnet 5 at its $3/$15 sticker; OpenRouter
            # billed the $2/$10 intro rate, so these land at 1.5x.
            ("anthropic/claude-sonnet-5", (844, 501340, 11809, 3319), 0.164668),
            ("anthropic/claude-sonnet-5", (4274, 495939, 230042, 20353), 0.886371),
            ("anthropic/claude-sonnet-5", (3235, 270239, 66143, 9755), 0.323425),
        ],
    )
    def test_lands_near_the_provider_price_on_prod_turns(self, model, tokens, real):
        estimate = estimate_cost_usd(model, _usage(*tokens))
        assert estimate is not None
        assert real / 1.05 <= estimate <= real * 1.55


class TestFallbackTurnCost:
    def test_failed_lookup_prices_the_turn_tokens_not_the_cli_total(self):
        # 10-03 02:09 8984a729 T1: every lookup failed; the CLI's running
        # session total ($9.29) was recorded for a turn worth ~$0.44.
        result = fallback_turn_cost(
            model=SONNET_5_5,
            turn_usage=_usage(109, 30869, 164649, 1799),
            resolved_costs={},
            call_usage={},
            cli_cost_usd=9.293846699999998,
        )
        assert result.source == "catalog"
        assert result.cost_usd == pytest.approx(0.436005, rel=1e-4)

    def test_failed_lookup_on_a_fresh_turn_matches_the_cli(self):
        # 10-03 03:31 b34818bb T1: no prior session cost, so the CLI figure
        # was already the turn's own price.
        result = fallback_turn_cost(
            model=SONNET_5_5,
            turn_usage=_usage(2731, 0, 35400, 817),
            resolved_costs={},
            call_usage={},
            cli_cost_usd=0.102132,
        )
        assert result.cost_usd == pytest.approx(0.102132, rel=1e-4)

    def test_partial_lookup_uses_provider_cost_for_resolved_calls(self):
        # 10-03 04:14 ab94b903 T1: partial (1/2), turn tokens
        # (109, 61738, 128529, 3230); CLI recorded $7.01.
        call_1 = _usage(54, 30869, 120000, 1600)
        call_2 = _usage(55, 30869, 8529, 1630)
        result = fallback_turn_cost(
            model=SONNET_5_5,
            turn_usage=_usage(109, 61738, 128529, 3230),
            resolved_costs={"gen-1": 0.322282},
            call_usage={"gen-1": call_1, "gen-2": call_2},
            cli_cost_usd=7.014386699999999,
        )
        assert result.source == "openrouter+catalog"
        expected_rest = estimate_cost_usd(SONNET_5_5, call_2)
        assert expected_rest is not None
        assert result.cost_usd == pytest.approx(0.322282 + expected_rest)
        assert result.cost_usd == pytest.approx(0.3662, rel=1e-3)

    def test_partial_lookup_without_per_call_usage_estimates_whole_turn(self):
        # A resolved gen with no captured usage (e.g. a subagent gen) cannot
        # be subtracted, so the estimate stays conservative.
        turn = _usage(109, 61738, 128529, 3230)
        result = fallback_turn_cost(
            model=SONNET_5_5,
            turn_usage=turn,
            resolved_costs={"gen-sub": 0.05},
            call_usage={},
            cli_cost_usd=7.0,
        )
        whole = estimate_cost_usd(SONNET_5_5, turn)
        assert whole is not None
        assert result.cost_usd == pytest.approx(0.05 + whole)

    def test_resolved_usage_larger_than_turn_never_goes_negative(self):
        result = fallback_turn_cost(
            model=SONNET_5_5,
            turn_usage=_usage(10, 10, 10, 10),
            resolved_costs={"gen-1": 0.01},
            call_usage={"gen-1": _usage(100, 100, 100, 100)},
            cli_cost_usd=1.0,
        )
        assert result.cost_usd == pytest.approx(0.01)

    def test_unknown_model_keeps_the_cli_figure(self):
        result = fallback_turn_cost(
            model="acme/unknown-model",
            turn_usage=_usage(100, 0, 0, 10),
            resolved_costs={"gen-1": 0.01},
            call_usage={},
            cli_cost_usd=0.5,
        )
        assert result.source == "cli"
        assert result.cost_usd == 0.5

    def test_no_cost_from_cli_still_gets_an_estimate(self):
        # 10-03 05:09 8984a729 T1 logged "fallback estimate=$None".
        result = fallback_turn_cost(
            model=SONNET_5_5,
            turn_usage=_usage(49, 0, 110401, 141),
            resolved_costs={},
            call_usage={},
            cli_cost_usd=None,
        )
        assert result.cost_usd == pytest.approx(0.27751, rel=1e-3)


class TestTokenUsageFromSdk:
    def test_reads_anthropic_usage_fields(self):
        usage = token_usage_from_sdk(
            {
                "input_tokens": 54,
                "cache_read_input_tokens": 30869,
                "cache_creation_input_tokens": 120000,
                "output_tokens": 1600,
            }
        )
        assert usage == _usage(54, 30869, 120000, 1600)

    def test_null_fields_count_as_zero(self):
        usage = token_usage_from_sdk(
            {"input_tokens": 5, "cache_read_input_tokens": None, "output_tokens": 2}
        )
        assert usage == _usage(5, 0, 0, 2)

    def test_missing_usage_is_none(self):
        assert token_usage_from_sdk(None) is None

    def test_merge_keeps_the_largest_count_per_field(self):
        first = _usage(54, 30869, 120000, 1)
        last = _usage(54, 30869, 120000, 1600)
        assert first.merge(last) == last
        assert last.merge(first) == last
