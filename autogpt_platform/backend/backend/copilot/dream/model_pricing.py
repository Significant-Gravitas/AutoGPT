"""Per-model USD pricing for dream-pass cost calculation.

Used as a fallback when the provider doesn't include a real cost on
``usage.cost`` (direct OpenAI / Anthropic APIs) or when computing the
batch-discounted equivalent. Rates are USD per 1M tokens.

The hard-coded table is intentionally narrow — only the models the
dream pass actually calls today. Adding a model requires updating this
file; we'd rather fail loud on an unknown model than silently bill at
zero. The OpenRouter path bypasses this entirely because OpenRouter
already returns the real spot price as ``usage.cost``.

When the Anthropic + OpenAI direct batch paths land (P0.1), the
``batch_discount`` factor in ``ExecutionPathDiscount`` is multiplied
into the computed cost so the savings flow through to the user via the
shared cost ledger.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRate:
    """Rate card for one model.

    All four token classes priced separately. ``None`` for a class
    means the model doesn't support it (e.g. no prompt cache).
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


# Rate card. Last updated 2026-05. Source: provider pricing pages.
# Keep alphabetized by provider, then by model.
_RATES: dict[str, ModelRate] = {
    # ---- Anthropic ----
    # Claude Opus 4.7
    "claude-opus-4-7": ModelRate(
        input_per_mtok=15.0,
        output_per_mtok=75.0,
        cache_read_per_mtok=1.5,
        cache_write_per_mtok=18.75,
    ),
    "anthropic/claude-opus-4.7": ModelRate(
        input_per_mtok=15.0,
        output_per_mtok=75.0,
        cache_read_per_mtok=1.5,
        cache_write_per_mtok=18.75,
    ),
    # Claude Sonnet 4.6
    "claude-sonnet-4-6": ModelRate(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_read_per_mtok=0.3,
        cache_write_per_mtok=3.75,
    ),
    "anthropic/claude-sonnet-4.6": ModelRate(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_read_per_mtok=0.3,
        cache_write_per_mtok=3.75,
    ),
    # Claude Haiku 4.5
    "claude-haiku-4-5-20251001": ModelRate(
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        cache_read_per_mtok=0.1,
        cache_write_per_mtok=1.25,
    ),
    # ---- OpenAI ----
    "gpt-5": ModelRate(input_per_mtok=10.0, output_per_mtok=30.0),
    "gpt-4o": ModelRate(
        input_per_mtok=2.5,
        output_per_mtok=10.0,
        cache_read_per_mtok=1.25,
    ),
}


ExecutionPath = Literal["sync_baseline", "anthropic_batch", "openai_batch"]


# Both Anthropic and OpenAI batch APIs offer a 50% discount on the
# normal rate for asynchronous batch processing. We pass that savings
# through to the user — recorded in the cost ledger so the discount is
# auditable.
_BATCH_DISCOUNTS: dict[ExecutionPath, float] = {
    "sync_baseline": 0.0,
    "anthropic_batch": 0.5,
    "openai_batch": 0.5,
}


def execution_path_discount(path: ExecutionPath) -> float:
    """Discount factor in [0.0, 1.0] applied to the base rate.

    Returns 0.5 for batch (50% off), 0.0 for sync_baseline.
    """
    return _BATCH_DISCOUNTS.get(path, 0.0)


def compute_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    execution_path: ExecutionPath = "sync_baseline",
) -> float | None:
    """Compute USD cost for a single LLM call from the rate card.

    Returns None when the model isn't in the rate card — caller should
    log a warning and treat the cost as unknown rather than silently
    bill at zero. When the provider already supplied a real cost on
    ``usage.cost`` the caller should prefer that value over this
    fallback; this function is the synthetic estimate when no spot
    price is available.

    ``execution_path`` discount is applied at the very end so the
    return value is the *final* user-facing cost, not the pre-discount
    base.
    """
    rate = _RATES.get(model) or _RATES.get(model.lower())
    if rate is None:
        logger.warning(
            "dream model_pricing: no rate card for model %r — cost unknown", model
        )
        return None

    input_cost = (input_tokens / 1_000_000.0) * rate.input_per_mtok
    output_cost = (output_tokens / 1_000_000.0) * rate.output_per_mtok

    cache_read_cost = 0.0
    if cache_read_tokens and rate.cache_read_per_mtok is not None:
        cache_read_cost = (cache_read_tokens / 1_000_000.0) * rate.cache_read_per_mtok

    cache_write_cost = 0.0
    if cache_creation_tokens and rate.cache_write_per_mtok is not None:
        cache_write_cost = (
            cache_creation_tokens / 1_000_000.0
        ) * rate.cache_write_per_mtok

    base = input_cost + output_cost + cache_read_cost + cache_write_cost
    discount = execution_path_discount(execution_path)
    return base * (1.0 - discount)
