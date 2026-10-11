"""Catalog-priced cost for an SDK turn whose provider cost lookup fell short.

When OpenRouter's ``/generation`` lookup fails for some or all of a turn's
generations, the reconcile (:mod:`openrouter_cost`) needs a figure to record
against the rate-limit counter and the budget tree.  The CLI's
``ResultMessage.total_cost_usd`` is not usable for that: on a resumed
session it is the session's running total, not the turn's, so recording it
re-charges every earlier turn (prod 2026-09-25..10-03: a median 12.8x the
turn's own price).

This module prices the turn's own token usage with the platform's LLM
catalog instead (``backend/data/llm_registry/catalog.py``): the provider's
USD list price per token kind, derived the same way block pricing derives
it.  Calls the provider did resolve keep the provider's figure; only the
remaining tokens are estimated.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import CatalogModelCost

# Catalog credits per USD of provider list price: 100 credits per dollar at
# the catalog's uniform 1.5x margin (``block_cost_config._USD_PER_1M_DIVISOR``).
_CREDITS_PER_USD = 150.0

FallbackSource = Literal["catalog", "openrouter+catalog", "cli"]


class TokenUsage(BaseModel):
    """Token counts of one model call or one whole turn."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0

    def merge(self, other: "TokenUsage") -> "TokenUsage":
        """Field-wise max: the CLI re-emits a call's usage per content block."""
        return TokenUsage(
            input_tokens=max(self.input_tokens, other.input_tokens),
            cache_read_tokens=max(self.cache_read_tokens, other.cache_read_tokens),
            cache_creation_tokens=max(
                self.cache_creation_tokens, other.cache_creation_tokens
            ),
            output_tokens=max(self.output_tokens, other.output_tokens),
        )

    def minus(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=max(0, self.input_tokens - other.input_tokens),
            cache_read_tokens=max(0, self.cache_read_tokens - other.cache_read_tokens),
            cache_creation_tokens=max(
                0, self.cache_creation_tokens - other.cache_creation_tokens
            ),
            output_tokens=max(0, self.output_tokens - other.output_tokens),
        )


class ModelPrice(BaseModel):
    """Provider list price in USD per 1M tokens of each kind."""

    model_config = ConfigDict(frozen=True)

    input: float
    cache_read: float
    cache_creation: float
    output: float


class FallbackCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    cost_usd: float | None
    source: FallbackSource


def fallback_turn_cost(
    *,
    model: str | None,
    turn_usage: TokenUsage,
    resolved_costs: dict[str, float],
    call_usage: dict[str, TokenUsage],
    cli_cost_usd: float | None,
) -> FallbackCost:
    """Cost to record for a turn whose provider lookup failed or was partial.

    ``resolved_costs`` maps each generation the provider priced to its cost;
    ``call_usage`` maps generations to the tokens the live stream reported
    for them.  The tokens of resolved calls are taken out of the turn's
    totals and the rest is priced from the catalog.  A resolved call with no
    captured usage is left in the totals, so the estimate errs high.  With
    no catalog price for ``model`` the CLI's figure is kept.
    """
    resolved_usage = [call_usage[g] for g in resolved_costs if g in call_usage]
    remaining = turn_usage
    for usage in resolved_usage:
        remaining = remaining.minus(usage)
    estimate = estimate_cost_usd(model, remaining)
    if estimate is None:
        return FallbackCost(cost_usd=cli_cost_usd, source="cli")
    if not resolved_costs:
        return FallbackCost(cost_usd=estimate, source="catalog")
    return FallbackCost(
        cost_usd=sum(resolved_costs.values()) + estimate,
        source="openrouter+catalog",
    )


def estimate_cost_usd(model: str | None, usage: TokenUsage) -> float | None:
    """USD list price of ``usage`` on ``model``, or None if it has no price."""
    price = model_price(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_creation_tokens * price.cache_creation
        + usage.output_tokens * price.output
    ) / 1_000_000


def model_price(model: str | None) -> ModelPrice | None:
    """Catalog list price for an OpenRouter or Anthropic model id."""
    if not model:
        return None
    for slug in _catalog_slug_candidates(model):
        cost = _CATALOG_COSTS.get(slug)
        if cost is not None:
            return _price_from_cost(cost)
    return None


def token_usage_from_sdk(usage: dict[str, Any] | None) -> TokenUsage | None:
    """Read an Anthropic-shaped ``usage`` dict (null fields count as 0)."""
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("input_tokens") or 0,
        cache_read_tokens=usage.get("cache_read_input_tokens") or 0,
        cache_creation_tokens=usage.get("cache_creation_input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
    )


def record_call_usage(
    usage_by_generation: dict[str, TokenUsage],
    generation_id: str,
    sdk_usage: dict[str, Any] | None,
) -> None:
    """Fold one ``AssistantMessage.usage`` into the per-generation map."""
    usage = token_usage_from_sdk(sdk_usage)
    if usage is None:
        return
    previous = usage_by_generation.get(generation_id)
    usage_by_generation[generation_id] = (
        usage if previous is None else previous.merge(usage)
    )


def _catalog_slug_candidates(model: str) -> list[str]:
    candidates = [model]
    vendor, _, bare = model.partition("/")
    if vendor == "anthropic" and bare:
        candidates += [bare, bare.replace(".", "-")]
    return candidates


def _price_from_cost(cost: CatalogModelCost) -> ModelPrice | None:
    rates = _input_output_usd(cost)
    if rates is None:
        return None
    input_usd, output_usd = rates
    return ModelPrice(
        input=input_usd,
        cache_read=_cache_rate(cost.cache_read_credits_per_1m, input_usd),
        cache_creation=_cache_rate(cost.cache_creation_credits_per_1m, input_usd),
        output=output_usd,
    )


def _input_output_usd(cost: CatalogModelCost) -> tuple[float, float] | None:
    if (
        cost.provider_input_usd_per_1m is not None
        and cost.provider_output_usd_per_1m is not None
    ):
        return cost.provider_input_usd_per_1m, cost.provider_output_usd_per_1m
    if cost.input_credits_per_1m is None or cost.output_credits_per_1m is None:
        return None
    return (
        cost.input_credits_per_1m / _CREDITS_PER_USD,
        cost.output_credits_per_1m / _CREDITS_PER_USD,
    )


def _cache_rate(credits_per_1m: float | None, input_usd: float) -> float:
    if not credits_per_1m:
        return input_usd
    return credits_per_1m / _CREDITS_PER_USD


# The catalog is immutable within a process (it only changes at deploy).
_CATALOG_COSTS: dict[str, CatalogModelCost] = {
    m.slug: m.cost for m in get_catalog().models if m.cost is not None
}
