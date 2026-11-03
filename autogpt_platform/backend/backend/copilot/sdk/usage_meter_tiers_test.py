import json

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from backend.copilot import anthropic_rate_card as card
from backend.copilot.sdk import usage_meter


@pytest.mark.parametrize(
    "model",
    ["claude-haiku-5-5", "anthropic/claude-haiku-5.5", "claude-haiku-5-5-20261007"],
)
def test_haiku_call_prices_full_prompt_with_caches(model):
    call = usage_meter.CallUsage(
        model=model,
        input_tokens=20_000,
        output_tokens=2_000,
        cache_read_tokens=120_000,
        cache_creation_tokens=10_000,
    )
    assert usage_meter.price_call_usd(call) == pytest.approx(0.031)


def test_small_calls_do_not_cross_a_tier_as_a_turn_total():
    meter = usage_meter.UsageMeter()
    for index in range(5):
        meter.observe(
            AssistantMessage(
                content=[TextBlock(text="x")],
                model="claude-haiku-5-5",
                message_id=f"request-{index}",
                usage={"input_tokens": 30_000, "output_tokens": 0},
            )
        )
    assert meter.main_calls == 5
    assert meter.spent_usd == pytest.approx(0.015)


def test_reference_rates_never_select_prompt_tiers(tmp_path, monkeypatch):
    data = json.loads(card._RATES_PATH.read_text())
    rates = data["rates"][usage_meter.REFERENCE_MODEL]
    rates["input_cost_per_token_above_200k_tokens"] = 6e-6
    rates["output_cost_per_token_above_200k_tokens"] = 22.5e-6
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(card, "_RATES_PATH", path)
    card.reset_cache()
    usage_meter._reference_rates.cache_clear()
    try:
        assert usage_meter._reference_rates() == pytest.approx((3.0, 15.0))
        assert usage_meter.cli_price_ratio("moonshotai/kimi-k2.6") == pytest.approx(
            15 / 2.8, rel=0.05
        )
        assert usage_meter.price_call_usd(
            usage_meter.CallUsage(
                model="someone/new-model",
                input_tokens=900_000,
                output_tokens=1_000,
            )
        ) == pytest.approx(2.715)
    finally:
        card.reset_cache()
        usage_meter._reference_rates.cache_clear()
