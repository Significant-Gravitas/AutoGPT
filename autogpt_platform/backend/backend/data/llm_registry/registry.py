"""Core LLM registry implementation for managing models dynamically."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Literal

import prisma.models
from pydantic import BaseModel, ConfigDict

from backend.util.cache import cached

logger = logging.getLogger(__name__)


class RegistryModelMetadata(BaseModel):
    """Model facts in the shape block schemas expect.

    Deliberately a standalone pydantic model rather than reusing the
    ``ModelMetadata`` NamedTuple from ``backend.blocks.llm`` — the registry
    must not couple to the hardcoded enum module it will eventually replace,
    and NamedTuples serialize as JSON arrays, which is wrong for API payloads.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    context_window: int
    max_output_tokens: int | None
    display_name: str
    provider_name: str
    creator_name: str
    price_tier: Literal[1, 2, 3]


class RegistryModelCost(BaseModel):
    """Cost configuration for an LLM model."""

    model_config = ConfigDict(frozen=True)

    unit: str  # "RUN" or "TOKENS"
    credit_cost: int
    credential_provider: str
    credential_id: str | None = None
    credential_type: str | None = None
    currency: str | None = None
    metadata: dict[str, Any] = {}


class RegistryModelCreator(BaseModel):
    """Creator information for an LLM model."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    display_name: str
    description: str | None = None
    website_url: str | None = None
    logo_url: str | None = None


class RegistryModel(BaseModel):
    """Represents a model in the LLM registry."""

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
    costs: tuple[RegistryModelCost, ...] = ()
    creator: RegistryModelCreator | None = None

    # isEnabled is the kill switch (never serves when false); visibility only
    # controls who SEES the model — HIDDEN still serves when explicitly routed.
    kind: str = "CHAT"
    visibility: str = "GA"
    min_subscription_tier: str | None = None
    source: str = "SEED"
    catalog_removed_at: datetime | None = None
    fallback_model_slug: str | None = None

    # Typed capability fields from DB schema
    supports_tools: bool = False
    supports_json_output: bool = False
    supports_reasoning: bool = False
    supports_parallel_tool_calls: bool = False


# L1 in-process cache — Redis is the shared L2 via @cached(shared_cache=True)
_dynamic_models: dict[str, RegistryModel] = {}
_schema_options: list[dict[str, str]] = []
_routes: dict[tuple[str, str, str], str] = {}
_lock = asyncio.Lock()


def _record_to_registry_model(record: prisma.models.LlmModel) -> RegistryModel:  # type: ignore[name-defined]
    """Transform a raw Prisma LlmModel record into a RegistryModel instance."""
    costs = tuple(
        RegistryModelCost(
            unit=str(cost.unit),
            credit_cost=cost.creditCost,
            credential_provider=cost.credentialProvider,
            credential_id=cost.credentialId,
            credential_type=cost.credentialType,
            currency=cost.currency,
            metadata=dict(cost.metadata or {}),
        )
        for cost in (record.Costs or [])
    )

    creator = None
    if record.Creator:
        creator = RegistryModelCreator(
            id=record.Creator.id,
            name=record.Creator.name,
            display_name=record.Creator.displayName,
            description=record.Creator.description,
            website_url=record.Creator.websiteUrl,
            logo_url=record.Creator.logoUrl,
        )

    capabilities = dict(record.capabilities or {})

    if not record.Provider:
        logger.warning(
            "LlmModel %s has no Provider despite NOT NULL FK - "
            "falling back to providerId %s",
            record.slug,
            record.providerId,
        )
    provider_name = record.Provider.name if record.Provider else record.providerId
    provider_display = (
        record.Provider.displayName if record.Provider else record.providerId
    )
    creator_name = record.Creator.displayName if record.Creator else "Unknown"

    if record.priceTier not in (1, 2, 3):
        logger.warning(
            "LlmModel %s has out-of-range priceTier=%s, defaulting to 1",
            record.slug,
            record.priceTier,
        )
    price_tier = record.priceTier if record.priceTier in (1, 2, 3) else 1

    metadata = RegistryModelMetadata(
        provider=provider_name,
        context_window=record.contextWindow,
        max_output_tokens=(
            record.maxOutputTokens
            if record.maxOutputTokens is not None
            else record.contextWindow
        ),
        display_name=record.displayName,
        provider_name=provider_display,
        creator_name=creator_name,
        price_tier=price_tier,
    )

    return RegistryModel(
        slug=record.slug,
        display_name=record.displayName,
        description=record.description,
        metadata=metadata,
        capabilities=capabilities,
        extra_metadata=dict(record.metadata or {}),
        provider_display_name=provider_display,
        is_enabled=record.isEnabled,
        is_recommended=record.isRecommended,
        costs=costs,
        creator=creator,
        kind=str(record.kind),
        visibility=str(record.visibility),
        min_subscription_tier=(
            str(record.minSubscriptionTier)
            if record.minSubscriptionTier is not None
            else None
        ),
        source=str(record.source),
        catalog_removed_at=record.catalogRemovedAt,
        fallback_model_slug=record.fallbackModelSlug,
        supports_tools=record.supportsTools,
        supports_json_output=record.supportsJsonOutput,
        supports_reasoning=record.supportsReasoning,
        supports_parallel_tool_calls=record.supportsParallelToolCalls,
    )


@cached(maxsize=1, ttl_seconds=300, shared_cache=True, refresh_ttl_on_get=True)
async def _fetch_registry_from_db() -> list[RegistryModel]:
    """Fetch all LLM models from the database.

    Results are cached in Redis (shared_cache=True) so subsequent calls within
    the TTL window skip the DB entirely — both within this process and across
    all other workers that share the same Redis instance.
    """
    records = await prisma.models.LlmModel.prisma().find_many(  # type: ignore[attr-defined]
        include={"Provider": True, "Costs": True, "Creator": True}
    )
    logger.info("Fetched %d LLM models from database", len(records))
    return [_record_to_registry_model(r) for r in records]


@cached(maxsize=1, ttl_seconds=300, shared_cache=True, refresh_ttl_on_get=True)
async def _fetch_routes_from_db() -> list[tuple[str, str, str, str]]:
    """Fetch all routing cells as (surface, mode, tier, modelSlug) tuples."""
    records = await prisma.models.LlmModelRoute.prisma().find_many()  # type: ignore[attr-defined]
    return [(r.surface, r.mode, r.tier, r.modelSlug) for r in records]


def clear_registry_cache() -> None:
    """Invalidate the shared Redis cache for the registry DB fetches.

    Call this before refresh_llm_registry() after any admin DB mutation so the
    next fetch hits the database rather than serving the now-stale cached data.
    """
    _fetch_registry_from_db.cache_clear()
    _fetch_routes_from_db.cache_clear()


async def refresh_llm_registry() -> None:
    """Refresh the in-process L1 cache from Redis/DB.

    On the first call (or after clear_registry_cache()), fetches fresh data
    from the database and stores it in Redis.  Subsequent calls by other
    workers hit the Redis cache instead of the DB.
    """
    async with _lock:
        try:
            models = await _fetch_registry_from_db()
            routes = await _fetch_routes_from_db()
            new_models = {m.slug: m for m in models}

            global _dynamic_models, _schema_options, _routes
            _dynamic_models = new_models
            _schema_options = _build_schema_options()
            _routes = {(s, m, t): slug for s, m, t, slug in routes}

            logger.info(
                "LLM registry refreshed: %d models, %d schema options, %d routes",
                len(_dynamic_models),
                len(_schema_options),
                len(_routes),
            )
        except Exception as e:
            logger.error("Failed to refresh LLM registry: %s", e, exc_info=True)
            raise


def _build_schema_options() -> list[dict[str, str]]:
    """Build schema options for model selection dropdown. Only includes enabled models."""
    return [
        {
            "label": model.display_name,
            "value": model.slug,
            "group": model.metadata.provider,
            "description": model.description or "",
        }
        for model in sorted(
            _dynamic_models.values(), key=lambda m: m.display_name.lower()
        )
        if model.is_enabled
    ]


def get_model(slug: str) -> RegistryModel | None:
    """Get a model by slug from the registry."""
    return _dynamic_models.get(slug)


def get_all_models() -> list[RegistryModel]:
    """Get all models from the registry (including disabled)."""
    return list(_dynamic_models.values())


def get_enabled_models() -> list[RegistryModel]:
    """Get only enabled models from the registry."""
    return [model for model in _dynamic_models.values() if model.is_enabled]


def get_schema_options() -> list[dict[str, str]]:
    """Get schema options for model selection dropdown (enabled models only)."""
    return list(_schema_options)


def get_default_model_slug() -> str | None:
    """Get the default model slug (first recommended, or first enabled)."""
    models = sorted(_dynamic_models.values(), key=lambda m: m.display_name)
    recommended = next(
        (m.slug for m in models if m.is_recommended and m.is_enabled), None
    )
    return recommended or next((m.slug for m in models if m.is_enabled), None)


def get_all_model_slugs_for_validation() -> list[str]:
    """Get all model slugs for validation (enabled models only)."""
    return [model.slug for model in _dynamic_models.values() if model.is_enabled]


def get_route(surface: str, mode: str, tier: str) -> str | None:
    """Return the admin-configured model slug for a routing cell, if set.

    Pure L1 lookup — callers own validation of the returned slug (existence
    and isEnabled are checked by the resolver so a stale cell degrades to the
    next routing layer instead of serving a dead model).
    """
    return _routes.get((surface, mode, tier))
