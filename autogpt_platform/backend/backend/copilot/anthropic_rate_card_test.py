"""Unit tests for the Anthropic rate card used in direct-mode cost computation."""

from .anthropic_rate_card import compute_anthropic_cost_usd


class TestComputeAnthropicCostUsd:
    def test_sonnet_basic(self):
        # 1M prompt × $3 + 1M completion × $15 = $18
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert cost == 18.0

    def test_opus_basic(self):
        # claude-opus-4-1 stays at the legacy Opus pricing in LiteLLM:
        # 1M prompt × $15 + 1M completion × $75 = $90.  (Newer Opus
        # models like opus-4-7 ship with revised $5/$25 rates per the
        # vendored LiteLLM data — covered by the LiteLLM source-of-truth
        # smoke below.)
        cost = compute_anthropic_cost_usd(
            model="claude-opus-4-1",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert cost == 90.0

    def test_opus_4_7_uses_litellm_revised_rates(self):
        # Regression cover for the static-rate-card era: opus-4-7 was
        # hardcoded at the legacy $15/$75 schedule, 3× over-billing every
        # turn.  LiteLLM's data has the revised $5/$25 rates.
        cost = compute_anthropic_cost_usd(
            model="claude-opus-4-7",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        # 1M × $5 + 1M × $25 = $30
        assert cost == 30.0

    def test_cache_read_at_one_tenth_input_rate(self):
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_tokens=1_000_000,
        )
        # 1M tokens × $3 × 0.1 = $0.30
        assert cost == 0.3

    def test_cache_write_at_two_times_input_rate(self):
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=0,
            completion_tokens=0,
            cache_creation_tokens=1_000_000,
        )
        # Default cache_ttl="1h": 1M tokens × $3 × 2.0 = $6
        assert cost == 6.0

    def test_cache_write_5m_ttl_uses_smaller_multiplier(self):
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=0,
            completion_tokens=0,
            cache_creation_tokens=1_000_000,
            cache_ttl="5m",
        )
        # 5m TTL: 1M tokens × $3 × 1.25 = $3.75
        assert cost == 3.75

    def test_cache_write_unknown_ttl_falls_back_to_1h(self):
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=0,
            completion_tokens=0,
            cache_creation_tokens=1_000_000,
            cache_ttl="24h",
        )
        # Unknown TTL → 1h multiplier (over-bills rather than mis-bills).
        assert cost == 6.0

    def test_unknown_model_falls_back_to_opus_4_1_rates(self, caplog):
        # Billing integrity: an unknown slug must **over-bill** at opus
        # rates rather than silently drop cost.  The error log surfaces
        # the misconfiguration to the operator without breaking the
        # already-completed API spend's accounting.
        import logging

        with caplog.at_level(
            logging.ERROR, logger="backend.copilot.anthropic_rate_card"
        ):
            cost = compute_anthropic_cost_usd(
                model="claude-future-7-5",
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
            )
        # 1M × $15 + 1M × $75 = $90 (opus-4-1 fallback rates)
        assert cost == 90.0
        assert any(
            "no entry for model=" in record.message
            and "claude-future-7-5" in record.message
            for record in caplog.records
        )

    def test_zero_tokens_zero_cost(self):
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert cost == 0.0

    def test_negative_tokens_clamped_to_zero(self):
        # Malformed upstream returning negative counts must not flip
        # the sign of the recorded cost.
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=-100,
            completion_tokens=-50,
            cache_read_tokens=-10,
            cache_creation_tokens=-20,
        )
        assert cost == 0.0

    def test_cached_tokens_subtracted_from_prompt_tokens(self):
        # Anthropic OAI-compat returns ``prompt_tokens`` as the total
        # input including cached + cache-write tokens.  Without the
        # subtract-out, fully-cached requests double-bill: full input
        # rate on prompt_tokens AND cache-read rate on cached_tokens.
        # Regression cover: 1M input all served from cache → just
        # the cache-read cost ($0.30), not $3.30.
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cache_read_tokens=1_000_000,
        )
        assert cost == 0.3

    def test_partial_cache_subtracts_correctly(self):
        # 1M input, 500K from cache → fresh = 500K, cache_read = 500K.
        # fresh = 500K × $3/M = $1.50; cache = 500K × $3 × 0.1 / M = $0.15.
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cache_read_tokens=500_000,
        )
        assert cost == 1.65

    def test_overreported_breakdown_clamps_fresh_to_zero(self):
        # Defensive: if cache_read + cache_creation > prompt_tokens
        # (upstream over-reports), fresh_input clamps to 0 instead of
        # going negative.
        cost = compute_anthropic_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=100,
            completion_tokens=0,
            cache_read_tokens=1000,
        )
        # fresh_input clamped to 0, cache_read = 1000 × $3 × 0.1 / M = $0.0003.
        assert cost == 0.0003
