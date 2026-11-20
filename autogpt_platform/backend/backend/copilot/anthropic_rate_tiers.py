"""Select first-party token prices for one request's full prompt length."""

import logging
import math
import re

logger = logging.getLogger(__name__)

_INPUT_TIER = re.compile(r"input_cost_per_token_above_([0-9]+)k_tokens")
# Anthropic's documented cache prices as multiples of the input price, used
# for a rate entry that omits them: reads 0.1x, 5-minute writes 1.25x and
# 1-hour writes 2x.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0
_CACHE_MULTIPLIERS = {
    "cache_read_input_token_cost": CACHE_READ_MULTIPLIER,
    "cache_creation_input_token_cost": CACHE_WRITE_5M_MULTIPLIER,
    "cache_creation_input_token_cost_above_1hr": CACHE_WRITE_1H_MULTIPLIER,
}


def apply_prompt_tier(info: dict[str, object], prompt_tokens: int) -> dict[str, object]:
    """Overlay the highest crossed tier without changing the cached rate card.

    Missing tier buckets retain their base-rate proportions. Invalid or zero
    tier fields are logged and use the same fallback, never a free bucket or
    a NaN or negative cost.
    """
    tiers = sorted(
        (
            (int(match[1]), key)
            for key in info
            if (match := _INPUT_TIER.fullmatch(key))
            and prompt_tokens > int(match[1]) * 1_000
        ),
        reverse=True,
    )
    if not tiers:
        return info
    base_input = _read_rate(info, "input_cost_per_token")
    base_output = _read_rate(info, "output_cost_per_token")
    if base_input is None or base_input <= 0 or base_output is None:
        return info
    for threshold, key in tiers:
        tier_input = _read_tier_rate(info, key)
        if tier_input is not None:
            return {
                **info,
                **_tier_rates(info, threshold, base_input, base_output, tier_input),
            }
    return info


def _tier_rates(
    info: dict[str, object],
    threshold: int,
    base_input: float,
    base_output: float,
    tier_input: float,
) -> dict[str, float]:
    base_rates = {
        "input_cost_per_token": base_input,
        "output_cost_per_token": base_output,
    }
    for field, multiplier in _CACHE_MULTIPLIERS.items():
        rate = _read_rate(info, field)
        base_rates[field] = rate if rate is not None else base_input * multiplier
    ratio = tier_input / base_input
    return {
        field: (
            rate
            if (rate := _read_tier_rate(info, f"{field}_above_{threshold}k_tokens"))
            is not None
            else base * ratio
        )
        for field, base in base_rates.items()
    }


def _read_tier_rate(info: dict[str, object], key: str) -> float | None:
    """A tier price, where zero is as broken as missing: it would bill free."""
    rate = _read_rate(info, key)
    if rate == 0:
        logger.error(f"Invalid zero Anthropic rate {key}; using rate fallback")
        return None
    return rate


def _read_rate(info: dict[str, object], key: str) -> float | None:
    if key not in info:
        return None
    value = info[key]
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ):
        return float(value)
    logger.error(f"Invalid Anthropic rate field {key}; using rate fallback")
    return None
