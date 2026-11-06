"""Replicate's published prices for its official models.

Official models are billed per output (an image, a second of video, a token)
rather than by run time, so the block prices their runs from
``official_prices.json``, which ``scripts/refresh_replicate_prices.py`` rebuilds
from Replicate's official collection and each model's pricing section.
"""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

InputValue = str | int | float | bool


class Criterion(BaseModel):
    """One condition of a price tier, as Replicate names it in a run's metrics."""

    key: str | None = None
    # A range criterion ([low, high]) is never matched, so it can't pick a cheaper tier.
    value: InputValue | list[float | None] | None
    # The model input that sets it, when the run doesn't report it.
    input: str | None = None
    input_value: InputValue | None = None


class Tier(BaseModel):
    criteria: list[Criterion] = []
    usd_per_unit: dict[str, float]


class Quantity(BaseModel):
    """Where a billed quantity comes from when the run doesn't report it."""

    kind: Literal[
        "per_run", "run_time", "outputs", "input_value", "input_text", "input_files"
    ]
    inputs: list[str] = []
    max_units: float | None = None
    # No default bounds it, so a run must set one of ``inputs``.
    required: bool = False


class OfficialModelPrice(BaseModel):
    tiers: list[Tier]
    quantities: dict[str, Quantity]
    defaults: dict[str, InputValue] = {}
    # Inputs it bills in a way the block can't count, so a run using one is refused.
    unpriced_inputs: list[str] = []


class OfficialPrices(BaseModel):
    read_on: str
    collection: list[str]
    priced: dict[str, OfficialModelPrice]
    unpriced: dict[str, str]


OFFICIAL_PRICES = OfficialPrices.model_validate_json(
    (Path(__file__).parent / "official_prices.json").read_text()
)
OFFICIAL_MODEL_PRICES = OFFICIAL_PRICES.priced


def official_run_cost_usd(
    price: OfficialModelPrice,
    model_inputs: Mapping[str, object],
    metrics: Mapping[str, object],
    output: object,
) -> float:
    """What Replicate charges for one run of an official model, in USD."""
    return sum(
        _rate(price, metric, model_inputs, metrics)
        * _units(price, metric, quantity, model_inputs, metrics, output)
        for metric, quantity in price.quantities.items()
    )


def run_refusal(
    price: OfficialModelPrice, model_inputs: Mapping[str, object]
) -> str | None:
    """Why the block can't price this run before it starts, if it can't."""
    used = [key for key in price.unpriced_inputs if model_inputs.get(key)]
    if used:
        return (
            f"this model bills '{used[0]}' in a way the Replicate block can't "
            "price yet; run it without that input"
        )
    for quantity in price.quantities.values():
        values = [model_inputs.get(key) for key in quantity.inputs]
        if quantity.required and not sum(
            _input_units(quantity.kind, value) for value in values
        ):
            return f"set '{quantity.inputs[0]}' so this model's charge has a limit"
    return None


def _rate(
    price: OfficialModelPrice,
    metric: str,
    model_inputs: Mapping[str, object],
    metrics: Mapping[str, object],
) -> float:
    """The dearest tier this run can't be ruled out of, so it never pays a cheaper one."""
    tiers = [tier for tier in price.tiers if metric in tier.usd_per_unit]
    known = _known_values(tiers)
    possible = [
        tier
        for tier in tiers
        if all(
            _possible(criterion, price, model_inputs, metrics, known)
            for criterion in tier.criteria
        )
    ]
    return max(tier.usd_per_unit[metric] for tier in possible or tiers)


def _possible(
    criterion: Criterion,
    price: OfficialModelPrice,
    model_inputs: Mapping[str, object],
    metrics: Mapping[str, object],
    known: Mapping[str, set[str]],
) -> bool:
    """False only when the run shows a different value the tiers know of; a value
    no tier names rules nothing out."""
    reported = _reported(criterion.key, metrics) if criterion.key else None
    if reported is not None and _text(reported) in known.get(
        f"key:{criterion.key}", ()
    ):
        return _text(reported) == _text(criterion.value)
    if criterion.input is not None:
        value = model_inputs.get(criterion.input, price.defaults.get(criterion.input))
        if _text(value) in known.get(f"input:{criterion.input}", ()):
            return _text(value) == _text(criterion.input_value)
    return True


def _known_values(tiers: list[Tier]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    for criterion in (c for tier in tiers for c in tier.criteria):
        if criterion.key:
            known.setdefault(f"key:{criterion.key}", set()).add(_text(criterion.value))
        if criterion.input is not None:
            known.setdefault(f"input:{criterion.input}", set()).add(
                _text(criterion.input_value)
            )
    return known


def _units(
    price: OfficialModelPrice,
    metric: str,
    quantity: Quantity,
    model_inputs: Mapping[str, object],
    metrics: Mapping[str, object],
    output: object,
) -> float:
    for key in (metric, _METRIC_ALIASES.get(metric)):
        reported = _number(_reported(key, metrics)) if key else None
        if reported is not None:
            return reported
    if quantity.kind == "per_run":
        return 1.0
    if quantity.kind == "run_time":
        return _number(metrics.get("predict_time")) or 0.0
    if quantity.kind == "outputs":
        return float(len(output)) if isinstance(output, list) else 1.0
    values = [model_inputs.get(key, price.defaults.get(key)) for key in quantity.inputs]
    from_inputs = sum(_input_units(quantity.kind, value) for value in values)
    if from_inputs > 0:
        return from_inputs
    return quantity.max_units or 0.0


# Older models report these names instead of the billing metric's own.
_METRIC_ALIASES = {
    "image_output_count": "image_count",
    "token_input_count": "input_token_count",
    "token_output_count": "output_token_count",
}


def _reported(key: str, metrics: Mapping[str, object]) -> object:
    billing = metrics.get("billing")
    if isinstance(billing, Mapping) and key in billing:
        return billing[key]
    for name in (f"billing.{key}", key):
        if name in metrics:
            return metrics[name]
    return None


def _text(value: object) -> str:
    return str(value).lower()


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _input_units(kind: str, value: object) -> float:
    """A number counts as itself, text by its characters (no fewer than its
    tokens), and files one each; a link or data URI is a file, never text."""
    if value is None or isinstance(value, bool) or value == "":
        return 0.0
    if isinstance(value, (list, dict)) and kind == "input_text":
        return float(len(json.dumps(value))) if value else 0.0
    if isinstance(value, list):
        return float(sum(_input_units(kind, item) > 0 for item in value))
    if isinstance(value, (int, float)):
        return float(value) if kind == "input_value" and math.isfinite(value) else 0.0
    text = str(value)
    is_file = text.startswith(("http://", "https://", "data:"))
    if kind == "input_files":
        return 1.0
    if kind == "input_text":
        return 0.0 if is_file else float(len(text))
    try:
        number = float(text)
    except ValueError:
        return 0.0
    return number if math.isfinite(number) else 0.0
