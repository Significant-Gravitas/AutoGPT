"""What a run would cost, before any call: tokens per case from the shape
of each call, priced at the LLM catalog's provider rates.

The token figures come from the task eval's stored runs (about 62k prompt
tokens, mostly cache reads, and 950 completion tokens per chat case on
``claude-native``; a worker turn reads, writes a file and finishes, so it is
counted at twice that) and from the size of each reviewer payload. Treat
the total as an order of magnitude. ``claude``/``claude-native`` and
``cursor`` bill a subscription, not per token: their figure is what the same
tokens would cost on the API.
"""

from pydantic import BaseModel

from backend.data.llm_registry.catalog import get_catalog

from .models import FeatureCase


class Usage(BaseModel):
    model: str
    fresh: int = 0
    cached: int = 0
    out: int = 0


class Price(BaseModel):
    fresh: float
    cached: float
    out: float


# Tokens per call: (fresh prompt, cached prompt, completion).
_WORKER = (14_000, 110_000, 3_000)
_WORKER_REVIEW = (10_000, 0, 600)
_DRAFT = (2_500, 0, 700)
_REVIEW = (5_000, 0, 500)


def case_usage(
    case: FeatureCase, otto: str, drafter: str, reviewers: list[str], live: str
) -> list[Usage]:
    """The calls a case would make."""
    calls: list[Usage] = []
    if case.needs == "live":
        if case.live != "start_suggestion":
            calls.append(_usage(live, _WORKER))
        return calls
    if case.area == "controls" and case.needs == "model":
        calls.append(_usage(otto, _WORKER))
    if case.area == "suggestions" and case.needs == "model" and not case.scripted_reply:
        calls.append(_usage(drafter, _DRAFT))
    if case.review:
        shape = _WORKER_REVIEW if case.area == "controls" else _REVIEW
        calls += [_usage(r, shape) for r in reviewers]
    return calls


def _usage(model: str, shape: tuple[int, int, int]) -> Usage:
    fresh, cached, out = shape
    return Usage(model=model, fresh=fresh, cached=cached, out=out)


def price_of(model: str) -> Price | None:
    """The catalog's provider rates for a model, USD per token; None when the
    catalog has no provider price for it."""
    slug = model.split(":")[-1].split("/")[-1]
    for entry in get_catalog().models:
        if entry.slug != slug and not entry.slug.startswith(f"{slug}-"):
            continue
        cost = entry.cost
        if cost is None or cost.provider_input_usd_per_1m is None:
            return None
        fresh = cost.provider_input_usd_per_1m
        cached = cost.provider_cache_read_usd_per_1m or fresh
        out = cost.provider_output_usd_per_1m or 0.0
        return Price(fresh=fresh / 1e6, cached=cached / 1e6, out=out / 1e6)
    return None


def dollars(calls: list[Usage]) -> tuple[float, list[str]]:
    """The calls' cost, and the models the catalog could not price."""
    total = 0.0
    unpriced: list[str] = []
    for call in calls:
        price = price_of(call.model)
        if price is None:
            unpriced.append(call.model)
            continue
        total += (
            call.fresh * price.fresh + call.cached * price.cached + call.out * price.out
        )
    return total, sorted(set(unpriced))
