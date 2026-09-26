"""One price card for background LLM inference, read from the LLM catalog.

The catalog (``backend/data/llm_registry/catalog.py``) authors what each
provider charges per million tokens in its ``provider_*_usd_per_1m`` cost
fields. :func:`price_for` finds a model's card from whichever spelling a
transport uses, and :func:`compute_cost_usd` prices one call from its
token counts.

This reads the catalog file itself (``get_catalog()``), not the in-process
registry: the registry is filled only where ``load_catalog()`` runs (the
REST API and the copilot executor), while background inference also runs
in the scheduler and the batch executor.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import CatalogModel, CatalogModelCost
from backend.data.llm_registry.llm_models import transport_slug_candidates

logger = logging.getLogger(__name__)

_TOKENS_PER_MTOK = 1_000_000

# Snapshot suffixes a transport spelling drops: Anthropic's -YYYYMMDD
# (claude-haiku-4-5-20251001) and OpenAI's -YYYY-MM-DD
# (gpt-4.1-mini-2025-04-14). The registry's date-stripped index knows only
# the first, which is why ``model_router.catalog_lookup`` misses gpt-4.1-mini.
_SNAPSHOT_SUFFIX_RE = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$")

# Slugs already reported at error level in this process (see _report_unpriced).
_reported_unpriced: set[str] = set()


class PriceCard(BaseModel):
    """What a model costs, in USD per million tokens, bucket by bucket.

    The buckets add up: ``input`` is the uncached input alone, and cache
    reads and cache writes are priced on top of it (Anthropic's usage
    convention, which the dream token counts follow).
    """

    model_config = ConfigDict(frozen=True)

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_usd_per_mtok: float
    cache_creation_usd_per_mtok: float
    source: Literal["catalog"] = "catalog"


def price_for(model: str) -> PriceCard | None:
    """The catalog price card for *model*, in any transport spelling.

    Takes the forms copilot config and the transports use, ignoring case:
    ``anthropic/claude-sonnet-5``, ``claude-opus-5.5``,
    ``anthropic/claude-haiku-4-5`` (a dated snapshot in the catalog) and
    bare ``gpt-4.1-mini``.

    Returns ``None``, logged at error level, when the model is not in the
    catalog or its entry lacks any of the four USD prices; a missing bucket
    price would otherwise have to be invented or read as free. Callers then
    treat the cost as unknown, never as zero.
    """
    entry = _catalog_model(model)
    if entry is None:
        _report_unpriced(model, "is not in the LLM catalog")
        return None
    card = _card_from_cost(entry.cost)
    if card is None:
        _report_unpriced(
            model,
            f"resolves to catalog model {entry.slug!r}, which lacks a provider "
            "USD price for input, output, cache read or cache write",
        )
    return card


def compute_cost_usd(
    *,
    price: PriceCard,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    discount: float = 0.0,
) -> float:
    """USD cost of one call: each bucket at its own rate, then the discount.

    *discount* is the share of the list price taken off, e.g. ``0.5`` on
    Anthropic's Message Batches API.
    """
    if not 0.0 <= discount <= 1.0:
        raise ValueError(f"discount must be within [0, 1], got {discount}")
    list_cost = (
        input_tokens * price.input_usd_per_mtok
        + output_tokens * price.output_usd_per_mtok
        + cache_read_tokens * price.cache_read_usd_per_mtok
        + cache_creation_tokens * price.cache_creation_usd_per_mtok
    ) / _TOKENS_PER_MTOK
    return list_cost * (1.0 - discount)


def _catalog_model(model: str) -> CatalogModel | None:
    """Every exact spelling first, then the undated ones: the order
    ``model_router.catalog_lookup`` tries them in."""
    exact, undated = _catalog_index()
    candidates = transport_slug_candidates(model.lower())
    for index in (exact, undated):
        for candidate in candidates:
            entry = index.get(candidate)
            if entry is not None:
                return entry
    return None


@functools.cache
def _catalog_index() -> tuple[dict[str, CatalogModel], dict[str, CatalogModel]]:
    """Lower-cased slug to entry, and snapshot-free slug to entry. Built once:
    the catalog is code, so it cannot change under a running process."""
    models = get_catalog().models
    exact = {m.slug.lower(): m for m in models}
    undated = {
        stripped: m
        for m in models
        if (stripped := _SNAPSHOT_SUFFIX_RE.sub("", m.slug.lower())) != m.slug.lower()
    }
    return exact, undated


def _card_from_cost(cost: CatalogModelCost | None) -> PriceCard | None:
    if (
        cost is None
        or cost.provider_input_usd_per_1m is None
        or cost.provider_output_usd_per_1m is None
        or cost.provider_cache_read_usd_per_1m is None
        or cost.provider_cache_creation_usd_per_1m is None
    ):
        return None
    return PriceCard(
        input_usd_per_mtok=cost.provider_input_usd_per_1m,
        output_usd_per_mtok=cost.provider_output_usd_per_1m,
        cache_read_usd_per_mtok=cost.provider_cache_read_usd_per_1m,
        cache_creation_usd_per_mtok=cost.provider_cache_creation_usd_per_1m,
    )


def _report_unpriced(model: str, reason: str) -> None:
    """Error level the first time per slug in a process (Sentry's logging
    integration turns error records into events), a warning after that: a
    misconfigured model is priced on every background call, and one Sentry
    event per call would be noise."""
    message = f"[price_card] {model!r} {reason}; its cost is unknown"
    if model in _reported_unpriced:
        logger.warning(message)
        return
    _reported_unpriced.add(model)
    logger.error(message)
