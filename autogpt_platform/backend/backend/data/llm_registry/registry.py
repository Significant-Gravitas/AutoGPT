"""In-process view of the LLM catalog.

``load_catalog()`` builds the L1 lookup structures from the canonical
``catalog.py`` file at startup; copilot routing reads through the
functions here, while the block layer projects the same file directly
(``backend.blocks.llm`` / ``block_cost_config``).
The file only changes at deploy, so a startup load is the whole cache
story — no Redis layer, no cross-pod invalidation.

The read interface is deliberately stable: it survived the DB-registry →
catalog-as-code redesign unchanged so consumers never care where model
facts live.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import (
    CatalogModelCost,
    CatalogPayload,
    ModelVisibility,
)
from backend.data.llm_registry.llm_models import MODEL_DATE_SUFFIX_RE

logger = logging.getLogger(__name__)


class RegistryModelMetadata(BaseModel):
    """Model facts in the shape block schemas expect.

    Field-compatible with ``llm_models.ModelMetadata`` by design — the two
    shapes must not drift (one is the block-facing NamedTuple projection,
    this is the router-facing pydantic view of the same catalog facts).

    Standalone rather than reusing the ``ModelMetadata`` NamedTuple from
    ``backend.blocks.llm`` — the registry must not couple to the hardcoded
    enum module it will eventually replace, and NamedTuples serialize as
    JSON arrays, which is wrong for API payloads.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    context_window: int
    max_output_tokens: int | None
    display_name: str
    provider_name: str
    creator_name: str
    price_tier: Literal[1, 2, 3]


class RegistryModel(BaseModel):
    """A catalog model joined with its provider/creator display data."""

    model_config = ConfigDict(frozen=True)

    slug: str
    display_name: str
    description: str | None = None
    metadata: RegistryModelMetadata
    provider_display_name: str
    # is_enabled is the kill switch for SERVING NEW WORK: copilot refuses the
    # model and it leaves picker metadata. Existing agent graphs keep
    # executing (and billing) — hard-stopping users' running agents is a
    # deliberate separate act (the retire CLI), never a flag side-effect.
    is_enabled: bool
    is_recommended: bool = False

    # visibility only controls who SEES the model — HIDDEN still serves
    # when explicitly routed.
    visibility: ModelVisibility = "GA"
    fallback_model_slug: str | None = None

    supports_tools: bool = False
    supports_json_output: bool = False
    supports_reasoning: bool = False
    supports_parallel_tool_calls: bool = False

    cost: CatalogModelCost | None = None


# L1 lookup structures — built once per process by load_catalog(). Reference
# swaps are atomic, and the file can't change under a running process, so no
# locking is needed.
_dynamic_models: dict[str, RegistryModel] = {}
# Date-stripped slug → model (claude-haiku-4-5 → the -20251001 entry), so
# the router's snapshot-suffix fallback is an O(1) lookup, not a scan.
_date_stripped_models: dict[str, RegistryModel] = {}
_routes: dict[tuple[str, str, str], str] = {}
_loaded = False


def _build_models(payload: CatalogPayload) -> dict[str, RegistryModel]:
    providers = {p.name: p for p in payload.providers}
    creators = {c.name: c for c in payload.creators}
    models: dict[str, RegistryModel] = {}
    for m in payload.models:
        provider = providers.get(m.provider)
        provider_display = provider.display_name if provider else m.provider
        creator = creators.get(m.creator) if m.creator else None
        models[m.slug] = RegistryModel(
            slug=m.slug,
            display_name=m.display_name,
            description=m.description,
            metadata=RegistryModelMetadata(
                provider=m.provider,
                context_window=m.context_window,
                # None means "unknown/no published cap" — substituting
                # context_window would overstate the output limit (a 1M-token
                # context model does not emit 1M output tokens).
                max_output_tokens=m.max_output_tokens,
                display_name=m.display_name,
                provider_name=provider_display,
                creator_name=creator.display_name if creator else "Unknown",
                price_tier=m.price_tier,
            ),
            provider_display_name=provider_display,
            is_enabled=m.is_enabled,
            is_recommended=m.is_recommended,
            visibility=m.visibility,
            fallback_model_slug=m.fallback_model_slug,
            supports_tools=m.supports_tools,
            supports_json_output=m.supports_json_output,
            supports_reasoning=m.supports_reasoning,
            supports_parallel_tool_calls=m.supports_parallel_tool_calls,
            cost=m.cost,
        )
    return models


def load_catalog(payload: CatalogPayload | None = None) -> None:
    """Build the L1 structures from the catalog file. Called at startup.

    Raises on an inconsistent payload — the catalog is load-bearing, so
    callers let the failure stop the boot rather than serving with an
    empty registry (which would silently disable routing cells and
    serve-time gating).
    """
    global _dynamic_models, _date_stripped_models, _routes, _loaded
    if payload is None:
        payload = get_catalog()
    models = _build_models(payload)
    routes = {
        (surface, mode, tier): slug
        for surface, modes in payload.routing.items()
        for mode, tiers in modes.items()
        for tier, slug in tiers.items()
    }
    _dynamic_models = models
    _date_stripped_models = {
        stripped: model
        for slug, model in models.items()
        if (stripped := MODEL_DATE_SUFFIX_RE.sub("", slug)) != slug
    }
    _routes = routes
    _loaded = True
    logger.info(
        f"LLM catalog loaded: {len(models)} models, {len(routes)} routing cells"
    )


def has_models() -> bool:
    """O(1) emptiness check — the resolver asks this on every turn."""
    return bool(_dynamic_models)


def is_loaded() -> bool:
    """Whether load_catalog() has run in this process — distinguishes a
    deliberately-dormant registry from an entrypoint that forgot to load
    (the latter silently disables copilot gating and routing cells)."""
    return _loaded


def get_model(slug: str) -> RegistryModel | None:
    """Get a model by slug from the catalog."""
    return _dynamic_models.get(slug)


def get_model_by_date_stripped_slug(slug: str) -> RegistryModel | None:
    """Resolve a date-suffix-less spelling to its snapshot-suffixed catalog
    entry (``claude-haiku-4-5`` → the ``-20251001`` model)."""
    return _date_stripped_models.get(slug)


def get_all_models() -> list[RegistryModel]:
    """All catalog models, including disabled ones."""
    return list(_dynamic_models.values())


def get_route(surface: str, mode: str, tier: str) -> str | None:
    """Return the catalog's routing-cell slug, if the cell is set.

    Pure L1 lookup — callers own validation of the returned slug (existence
    and is_enabled are checked by the resolver so a stale cell degrades to
    the next routing layer instead of serving a dead model).
    """
    return _routes.get((surface, mode, tier))
