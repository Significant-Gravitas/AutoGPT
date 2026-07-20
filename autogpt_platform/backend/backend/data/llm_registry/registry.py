"""In-process view of the LLM catalog.

``load_catalog()`` builds the L1 lookup structures from the canonical
``catalog.py`` file at startup; every consumer (copilot routing, the public
catalog endpoint, Phase B block options/costs) reads through the functions
here. The file only changes at deploy, so a startup load is the whole cache
story — no Redis layer, no cross-pod invalidation.

The read interface is deliberately stable: it survived the DB-registry →
catalog-as-code redesign unchanged so consumers never care where model
facts live.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from backend.data.llm_registry.catalog import get_catalog
from backend.data.llm_registry.catalog_model import CatalogModelCost, CatalogPayload

logger = logging.getLogger(__name__)


class RegistryModelMetadata(BaseModel):
    """Model facts in the shape block schemas expect.

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


class RegistryModelCreator(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None


class RegistryModel(BaseModel):
    """A catalog model joined with its provider/creator display data."""

    model_config = ConfigDict(frozen=True)

    slug: str
    display_name: str
    description: str | None = None
    metadata: RegistryModelMetadata
    capabilities: dict[str, Any] = {}
    extra_metadata: dict[str, Any] = {}
    provider_display_name: str
    is_enabled: bool
    is_recommended: bool = False
    creator: RegistryModelCreator | None = None

    # is_enabled is the kill switch (never serves when False); visibility only
    # controls who SEES the model — HIDDEN still serves when explicitly routed.
    kind: str = "CHAT"
    visibility: str = "GA"
    min_subscription_tier: str | None = None
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
_schema_options: list[dict[str, str]] = []
_routes: dict[tuple[str, str, str], str] = {}


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
                max_output_tokens=(
                    m.max_output_tokens
                    if m.max_output_tokens is not None
                    else m.context_window
                ),
                display_name=m.display_name,
                provider_name=provider_display,
                creator_name=creator.display_name if creator else "Unknown",
                price_tier=m.price_tier,
            ),
            capabilities=dict(m.capabilities),
            extra_metadata=dict(m.metadata),
            provider_display_name=provider_display,
            is_enabled=m.is_enabled,
            is_recommended=m.is_recommended,
            creator=(
                RegistryModelCreator(
                    name=creator.name,
                    display_name=creator.display_name,
                    description=creator.description,
                    website_url=creator.website_url,
                    logo_url=creator.logo_url,
                )
                if creator
                else None
            ),
            kind=m.kind,
            visibility=m.visibility,
            min_subscription_tier=m.min_subscription_tier,
            fallback_model_slug=m.fallback_model_slug,
            supports_tools=m.supports_tools,
            supports_json_output=m.supports_json_output,
            supports_reasoning=m.supports_reasoning,
            supports_parallel_tool_calls=m.supports_parallel_tool_calls,
            cost=m.cost,
        )
    return models


def _build_schema_options(models: dict[str, RegistryModel]) -> list[dict[str, str]]:
    """Model-selection dropdown options. Enabled models only."""
    return [
        {
            "label": model.display_name,
            "value": model.slug,
            "group": model.metadata.provider,
            "description": model.description or "",
        }
        for model in sorted(models.values(), key=lambda m: m.display_name.lower())
        if model.is_enabled
    ]


def load_catalog(payload: CatalogPayload | None = None) -> None:
    """Build the L1 structures from the catalog file. Called at startup.

    Raises on an inconsistent payload — the caller owns fail-soft (an empty
    registry degrades every consumer to pre-catalog behavior).
    """
    global _dynamic_models, _schema_options, _routes
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
    _schema_options = _build_schema_options(models)
    _routes = routes
    logger.info(
        "LLM catalog loaded: %d models, %d schema options, %d routing cells",
        len(models),
        len(_schema_options),
        len(routes),
    )


def get_model(slug: str) -> RegistryModel | None:
    """Get a model by slug from the catalog."""
    return _dynamic_models.get(slug)


def get_all_models() -> list[RegistryModel]:
    """All catalog models, including disabled ones."""
    return list(_dynamic_models.values())


def get_enabled_models() -> list[RegistryModel]:
    """Only enabled catalog models."""
    return [model for model in _dynamic_models.values() if model.is_enabled]


def get_schema_options() -> list[dict[str, str]]:
    """Model-selection dropdown options (enabled models only)."""
    return list(_schema_options)


def get_default_model_slug() -> str | None:
    """First recommended enabled model, else first enabled model."""
    models = sorted(_dynamic_models.values(), key=lambda m: m.display_name)
    recommended = next(
        (m.slug for m in models if m.is_recommended and m.is_enabled), None
    )
    return recommended or next((m.slug for m in models if m.is_enabled), None)


def get_all_model_slugs_for_validation() -> list[str]:
    """Enabled model slugs, for validating user-supplied model ids."""
    return [model.slug for model in _dynamic_models.values() if model.is_enabled]


def get_route(surface: str, mode: str, tier: str) -> str | None:
    """Return the catalog's routing-cell slug, if the cell is set.

    Pure L1 lookup — callers own validation of the returned slug (existence
    and is_enabled are checked by the resolver so a stale cell degrades to
    the next routing layer instead of serving a dead model).
    """
    return _routes.get((surface, mode, tier))
