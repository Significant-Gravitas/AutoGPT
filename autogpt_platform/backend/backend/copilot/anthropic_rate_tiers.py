"""Select first-party token prices for one request's full prompt length."""

import logging
import math
import re

logger = logging.getLogger(__name__)

_INPUT_TIER = re.compile(r"input_cost_per_token_above_([0-9]+)k_tokens")
_CACHE_MULTIPLIERS = {
    "cache_read_input_token_cost": 0.1,
    "cache_creation_input_token_cost": 1.25,
    "cache_creation_input_token_cost_above_1hr": 2.0,
}


def apply_prompt_tier(info: dict[str, object], prompt_tokens: int) -> dict[str, object]:
    """Overlay the highest crossed tier without changing the cached rate card.

    Missing tier buckets retain their base-rate proportions. Invalid numeric
    fields are logged and use the same fallback, never NaN or negative costs.
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
        tier_input = _read_rate(info, key)
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
            if (rate := _read_rate(info, f"{field}_above_{threshold}k_tokens"))
            is not None
            else base * ratio
        )
        for field, base in base_rates.items()
    }


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
    logger.error("Invalid Anthropic rate field %s; using rate fallback", key)
    return None
