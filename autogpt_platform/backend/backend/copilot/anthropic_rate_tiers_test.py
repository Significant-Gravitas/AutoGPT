import json
import logging
import math
import re

import pytest

from backend.copilot import anthropic_rate_card as card

MODEL = "claude-haiku-5-5"
RATE_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
)


@pytest.fixture
def rate_file(tmp_path, monkeypatch):
    path = tmp_path / "rates.json"
    data = json.loads(card._RATES_PATH.read_text())
    monkeypatch.setattr(card, "_RATES_PATH", path)

    def write(rates):
        path.write_text(json.dumps({"rates": {MODEL: rates}}))
        card.reset_cache()

    yield data["rates"][MODEL], write
    card.reset_cache()


@pytest.mark.parametrize(
    "prompt,read,write,output,ttl,expected",
    [
        (100_001, 0, 0, 1_000, "1h", 0.0525005),
        (110_000, 20_000, 0, 0, "1h", 0.046),
        (110_000, 0, 20_000, 0, "5m", 0.0575),
        (150_000, 120_000, 10_000, 2_000, "1h", 0.031),
        (0, 100_001, 0, 0, "1h", 0.00500005),
        (10, 50_001, 50_000, 0, "1h", 0.05250005),
        (1_000_000, 0, 0, 1_000_000, "1h", 3.0),
        (2_000_000, 1_000_000, 1_000_000, 0, "5m", 0.675),
        (2_000_000, 1_000_000, 1_000_000, 0, "1h", 1.05),
        (2_000_000, 1_000_000, 1_000_000, 0, "unknown", 1.05),
    ],
)
def test_long_prompt_prices_every_bucket(prompt, read, write, output, ttl, expected):
    assert card.compute_anthropic_cost_usd(
        model=MODEL,
        prompt_tokens=prompt,
        completion_tokens=output,
        cache_read_tokens=read,
        cache_creation_tokens=write,
        cache_ttl=ttl,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "prompt,read,write,output,expected",
    [
        (100_000, 0, 0, 1_000, 0.0105),
        (100_000, 80_000, 10_000, 1_000, 0.0043),
        (0, 50_000, 50_000, 0, 0.0105),
        (1, 0, 0, 1_000_000, 0.5000001),
        (-1, -1, -1, -1, 0),
    ],
)
def test_threshold_counts_prompt_buckets_only(prompt, read, write, output, expected):
    assert card.compute_anthropic_cost_usd(
        model=MODEL,
        prompt_tokens=prompt,
        completion_tokens=output,
        cache_read_tokens=read,
        cache_creation_tokens=write,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("field", RATE_FIELDS[1:])
@pytest.mark.parametrize("bad", [None, "0.1", True, float("nan"), float("inf"), -1])
def test_missing_or_invalid_tier_bucket_uses_scaled_base(rate_file, caplog, field, bad):
    rates, write = rate_file
    key = field + "_above_100k_tokens"
    if bad is None:
        rates.pop(key)
    else:
        rates[key] = bad
    write(rates)
    with caplog.at_level(logging.ERROR):
        cost = card.compute_anthropic_cost_usd(
            model=MODEL,
            prompt_tokens=150_000,
            completion_tokens=2_000,
            cache_read_tokens=120_000,
            cache_creation_tokens=10_000,
            cache_ttl="5m" if field == RATE_FIELDS[3] else "1h",
        )
    assert cost == pytest.approx(0.02725 if field == RATE_FIELDS[3] else 0.031)
    if bad is not None:
        assert key in caplog.text


@pytest.mark.parametrize("bad", ["0.1", True, float("nan"), float("inf"), -1])
def test_invalid_input_tier_is_ignored(rate_file, caplog, bad):
    rates, write = rate_file
    key = "input_cost_per_token_above_100k_tokens"
    rates[key] = bad
    write(rates)
    with caplog.at_level(logging.ERROR):
        cost = card.compute_anthropic_cost_usd(
            model=MODEL, prompt_tokens=200_000, completion_tokens=0
        )
    assert cost == pytest.approx(0.02)
    assert key in caplog.text


def test_highest_crossed_tier_wins_without_mutating_cached_rates(rate_file):
    rates, write = rate_file
    for field in RATE_FIELDS:
        rates[field + "_above_200k_tokens"] = rates[field] * 10
    write(rates)
    for prompt, expected in [(200_001, 0.200001), (200_000, 0.1), (100_000, 0.01)]:
        assert card.compute_anthropic_cost_usd(
            model=MODEL, prompt_tokens=prompt, completion_tokens=0
        ) == pytest.approx(expected)
    assert card._load_rates()[MODEL] == rates


def test_missing_cache_fields_scale_default_multipliers(rate_file):
    rates, write = rate_file
    for key in list(rates):
        if key.startswith("cache_"):
            del rates[key]
    write(rates)
    for ttl, expected in [("5m", 0.02725), ("1h", 0.031)]:
        assert card.compute_anthropic_cost_usd(
            model=MODEL,
            prompt_tokens=150_000,
            completion_tokens=2_000,
            cache_read_tokens=120_000,
            cache_creation_tokens=10_000,
            cache_ttl=ttl,
        ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "model,expected", [("claude-sonnet-5-5", 1.81), ("claude-opus-5-5", 3.62)]
)
def test_non_tiered_models_keep_standard_long_context_rates(model, expected):
    assert card.compute_anthropic_cost_usd(
        model=model, prompt_tokens=900_000, completion_tokens=1_000
    ) == pytest.approx(expected)


def test_vendored_rates_are_complete_and_ordered():
    for model, info in card._load_rates().items():
        suffixes = [""] + [
            key.removeprefix("input_cost_per_token")
            for key in info
            if re.fullmatch(r"input_cost_per_token_above_[0-9]+k_tokens", key)
        ]
        for suffix in suffixes:
            for field in RATE_FIELDS[:2]:
                assert field + suffix in info, (model, field, suffix)
            for field in RATE_FIELDS:
                value = info.get(field + suffix)
                if value is None:
                    continue
                assert not isinstance(value, bool)
                assert isinstance(value, (int, float)) and math.isfinite(value)
                assert value >= info.get(field, 0) >= 0, (model, field, suffix)
            assert (
                info["output_cost_per_token" + suffix]
                >= info["input_cost_per_token" + suffix]
            )
            cache = [info.get(field + suffix) for field in RATE_FIELDS[2:]]
            if all(value is not None for value in cache):
                assert cache[2] >= cache[1] >= cache[0], (model, suffix)
