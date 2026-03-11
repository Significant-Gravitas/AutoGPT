"""Core LLM registry implementation for managing models dynamically."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import prisma.models

from backend.data.llm_registry.model import ModelMetadata
from backend.util.cache import cached

logger = logging.getLogger(__name__)


def _json_to_dict(value: Any) -> dict[str, Any]:
    """Convert Prisma Json type to dict, with fallback to empty dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    # Prisma Json type should always be a dict at runtime
    return dict(value) if value else {}


@dataclass(frozen=True)
class RegistryModelCost:
    """Cost configuration for an LLM model."""

    credit_cost: int
    credential_provider: str
    credential_id: str | None
    credential_type: str | None
    currency: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegistryModelCreator:
    """Creator information for an LLM model."""

    id: str
    name: str
    display_name: str
    description: str | None
    website_url: str | None
    logo_url: str | None


@dataclass(frozen=True)
class RegistryModel:
    """Represents a model in the LLM registry."""

    slug: str
    display_name: str
    description: str | None
    metadata: ModelMetadata
    capabilities: dict[str, Any]
    extra_metadata: dict[str, Any]
    provider_display_name: str
    is_enabled: bool
    is_recommended: bool = False
    costs: tuple[RegistryModelCost, ...] = field(default_factory=tuple)
    creator: RegistryModelCreator | None = None


_static_metadata: dict[str, ModelMetadata] = {}
_static_costs: dict[str, int] = {}
_dynamic_models: dict[str, RegistryModel] = {}
_schema_options: list[dict[str, str]] = []
_discriminator_mapping: dict[str, str] = {}
_lock = asyncio.Lock()


def register_static_metadata(metadata: dict[Any, ModelMetadata]) -> None:
    """Register static metadata for legacy models (deprecated)."""
    _static_metadata.update({str(key): value for key, value in metadata.items()})
    _refresh_cached_schema()


def register_static_costs(costs: dict[Any, int]) -> None:
    """Register static costs for legacy models (deprecated)."""
    _static_costs.update({str(key): value for key, value in costs.items()})


def _build_schema_options() -> list[dict[str, str]]:
    """Build schema options for model selection dropdown. Only includes enabled models."""
    options: list[dict[str, str]] = []
    # Only include enabled models in the dropdown options
    for model in sorted(_dynamic_models.values(), key=lambda m: m.display_name.lower()):
        if model.is_enabled:
            options.append(
                {
                    "label": model.display_name,
                    "value": model.slug,
                    "group": model.metadata.provider,
                    "description": model.description or "",
                }
            )

    for slug, metadata in _static_metadata.items():
        if slug in _dynamic_models:
            continue
        options.append(
            {
                "label": slug,
                "value": slug,
                "group": metadata.provider,
                "description": "",
            }
        )
    return options


@cached(maxsize=1, ttl_seconds=300, shared_cache=True, refresh_ttl_on_get=True)
async def _fetch_registry_from_db() -> list[dict[str, Any]]:
    """
    Fetch all LLM models from database with related data.

    Cached in Redis with 300s TTL. Thundering herd protection ensures
    only one executor queries DB even if 1000 receive notification simultaneously.
    """
    records = await prisma.models.LlmModel.prisma().find_many(
        include={
            "Provider": True,
            "Costs": True,
            "Creator": True,
        }
    )
    logger.debug("Fetched %d LLM model records from database", len(records))

    # Serialize to plain dicts for caching
    return [
        {
            "id": record.id,
            "slug": record.slug,
            "displayName": record.displayName,
            "description": record.description,
            "providerId": record.providerId,
            "creatorId": record.creatorId,
            "contextWindow": record.contextWindow,
            "maxOutputTokens": record.maxOutputTokens,
            "priceTier": getattr(record, "priceTier", 1) or 1,
            "isEnabled": record.isEnabled,
            "isRecommended": record.isRecommended,
            "capabilities": record.capabilities,
            "metadata": record.metadata,
            "Provider": (
                {
                    "name": (
                        record.Provider.name if record.Provider else record.providerId
                    ),
                    "displayName": (
                        record.Provider.displayName
                        if record.Provider
                        else record.providerId
                    ),
                }
                if record.Provider
                else None
            ),
            "Costs": [
                {
                    "creditCost": cost.creditCost,
                    "credentialProvider": cost.credentialProvider,
                    "credentialId": cost.credentialId,
                    "credentialType": cost.credentialType,
                    "currency": cost.currency,
                    "metadata": cost.metadata,
                }
                for cost in (record.Costs or [])
            ],
            "Creator": (
                {
                    "id": record.Creator.id,
                    "name": record.Creator.name,
                    "displayName": record.Creator.displayName,
                    "description": record.Creator.description,
                    "websiteUrl": record.Creator.websiteUrl,
                    "logoUrl": record.Creator.logoUrl,
                }
                if record.Creator
                else None
            ),
        }
        for record in records
    ]


async def refresh_llm_registry(models_data: list[dict[str, Any]] | None = None) -> None:
    """
    Refresh the LLM registry from the database or provided data.

    Args:
        models_data: Optional pre-fetched model data from notification payload
    """
    async with _lock:
        try:
            if models_data is None:
                # Fetch from cache (thundering herd protected)
                models_data = await _fetch_registry_from_db()

            logger.debug("Processing %d LLM model records", len(models_data))
        except Exception as exc:
            logger.error(
                "Failed to refresh LLM registry from DB: %s", exc, exc_info=True
            )
            return

        dynamic: dict[str, RegistryModel] = {}
        for record_dict in models_data:
            provider = record_dict.get("Provider")
            creator_data = record_dict.get("Creator")

            provider_name = provider["name"] if provider else record_dict["providerId"]
            provider_display_name = (
                provider["displayName"] if provider else record_dict["providerId"]
            )
            # Creator name: prefer Creator.name, fallback to provider display name
            creator_name = (
                creator_data["name"] if creator_data else provider_display_name
            )
            # Price tier: default to 1 (cheapest) if not set
            price_tier = record_dict.get("priceTier", 1) or 1
            # Clamp to valid range 1-3
            price_tier = max(1, min(3, price_tier))

            metadata = ModelMetadata(
                provider=provider_name,
                context_window=record_dict["contextWindow"],
                max_output_tokens=record_dict["maxOutputTokens"],
                display_name=record_dict["displayName"],
                provider_name=provider_display_name,
                creator_name=creator_name,
                price_tier=price_tier,  # type: ignore[arg-type]
            )
            costs = tuple(
                RegistryModelCost(
                    credit_cost=cost["creditCost"],
                    credential_provider=cost["credentialProvider"],
                    credential_id=cost.get("credentialId"),
                    credential_type=cost.get("credentialType"),
                    currency=cost.get("currency"),
                    metadata=_json_to_dict(cost.get("metadata")),
                )
                for cost in record_dict.get("Costs", [])
            )

            # Map creator if present
            creator = None
            if creator_data:
                creator = RegistryModelCreator(
                    id=creator_data["id"],
                    name=creator_data["name"],
                    display_name=creator_data["displayName"],
                    description=creator_data.get("description"),
                    website_url=creator_data.get("websiteUrl"),
                    logo_url=creator_data.get("logoUrl"),
                )

            dynamic[record_dict["slug"]] = RegistryModel(
                slug=record_dict["slug"],
                display_name=record_dict["displayName"],
                description=record_dict.get("description"),
                metadata=metadata,
                capabilities=_json_to_dict(record_dict.get("capabilities")),
                extra_metadata=_json_to_dict(record_dict.get("metadata")),
                provider_display_name=provider_display_name,
                is_enabled=record_dict["isEnabled"],
                is_recommended=record_dict["isRecommended"],
                costs=costs,
                creator=creator,
            )

        # Atomic swap - build new structures then replace references
        # This ensures readers never see partially updated state
        global _dynamic_models
        _dynamic_models = dynamic
        _refresh_cached_schema()
        logger.info(
            "LLM registry refreshed with %s dynamic models (enabled: %s, disabled: %s)",
            len(dynamic),
            sum(1 for m in dynamic.values() if m.is_enabled),
            sum(1 for m in dynamic.values() if not m.is_enabled),
        )


def _refresh_cached_schema() -> None:
    """Refresh cached schema options and discriminator mapping."""
    global _schema_options, _discriminator_mapping

    # Build new structures
    new_options = _build_schema_options()
    new_mapping = {
        slug: entry.metadata.provider for slug, entry in _dynamic_models.items()
    }
    for slug, metadata in _static_metadata.items():
        new_mapping.setdefault(slug, metadata.provider)

    # Atomic swap - replace references to ensure readers see consistent state
    _schema_options = new_options
    _discriminator_mapping = new_mapping


def get_llm_model_metadata(slug: str) -> ModelMetadata | None:
    """Get model metadata by slug. Checks dynamic models first, then static metadata."""
    if slug in _dynamic_models:
        return _dynamic_models[slug].metadata
    return _static_metadata.get(slug)


def get_llm_model_cost(slug: str) -> tuple[RegistryModelCost, ...]:
    """Get model cost configuration by slug."""
    if slug in _dynamic_models:
        return _dynamic_models[slug].costs
    cost_value = _static_costs.get(slug)
    if cost_value is None:
        return tuple()
    return (
        RegistryModelCost(
            credit_cost=cost_value,
            credential_provider="static",
            credential_id=None,
            credential_type=None,
            currency=None,
            metadata={},
        ),
    )


def get_llm_model_schema_options() -> list[dict[str, str]]:
    """
    Get schema options for LLM model selection dropdown.

    Returns a copy of cached schema options that are refreshed when the registry is
    updated via refresh_llm_registry() (called on startup and via Redis pub/sub).
    """
    # Return a copy to prevent external mutation
    return list(_schema_options)


def get_llm_discriminator_mapping() -> dict[str, str]:
    """
    Get discriminator mapping for LLM models.

    Returns a copy of cached discriminator mapping that is refreshed when the registry
    is updated via refresh_llm_registry() (called on startup and via Redis pub/sub).
    """
    # Return a copy to prevent external mutation
    return dict(_discriminator_mapping)


def get_dynamic_model_slugs() -> set[str]:
    """Get all dynamic model slugs from the registry."""
    return set(_dynamic_models.keys())


def get_all_model_slugs_for_validation() -> set[str]:
    """
    Get ALL model slugs (both enabled and disabled) for validation purposes.

    This is used for JSON schema enum validation - we need to accept any known
    model value (even disabled ones) so that existing graphs don't fail validation.
    The actual fallback/enforcement happens at runtime in llm_call().
    """
    all_slugs = set(_dynamic_models.keys())
    all_slugs.update(_static_metadata.keys())
    return all_slugs


def iter_dynamic_models() -> Iterable[RegistryModel]:
    """Iterate over all dynamic models in the registry."""
    return tuple(_dynamic_models.values())


def get_fallback_model_for_disabled(disabled_model_slug: str) -> RegistryModel | None:
    """
    Find a fallback model when the requested model is disabled.

    Looks for an enabled model from the same provider. Prefers models with
    similar names or capabilities if possible.

    Args:
        disabled_model_slug: The slug of the disabled model

    Returns:
        An enabled RegistryModel from the same provider, or None if no fallback found
    """
    disabled_model = _dynamic_models.get(disabled_model_slug)
    if not disabled_model:
        return None

    provider = disabled_model.metadata.provider

    # Find all enabled models from the same provider
    candidates = [
        model
        for model in _dynamic_models.values()
        if model.is_enabled and model.metadata.provider == provider
    ]

    if not candidates:
        return None

    # Sort by: prefer models with similar context window, then by name
    candidates.sort(
        key=lambda m: (
            abs(m.metadata.context_window - disabled_model.metadata.context_window),
            m.display_name.lower(),
        )
    )

    return candidates[0]


def is_model_enabled(model_slug: str) -> bool:
    """Check if a model is enabled in the registry."""
    model = _dynamic_models.get(model_slug)
    if not model:
        # Model not in registry - assume it's a static/legacy model and allow it
        return True
    return model.is_enabled


def get_model_info(model_slug: str) -> RegistryModel | None:
    """Get model info from the registry."""
    return _dynamic_models.get(model_slug)


def get_default_model_slug() -> str | None:
    """
    Get the default model slug to use for block defaults.

    Returns the recommended model if set (configured via admin UI),
    otherwise returns the first enabled model alphabetically.
    Returns None if no models are available or enabled.
    """
    # Return the recommended model if one is set and enabled
    for model in _dynamic_models.values():
        if model.is_recommended and model.is_enabled:
            return model.slug

    # No recommended model set - find first enabled model alphabetically
    for model in sorted(_dynamic_models.values(), key=lambda m: m.display_name.lower()):
        if model.is_enabled:
            logger.warning(
                "No recommended model set, using '%s' as default",
                model.slug,
            )
            return model.slug

    # No enabled models available
    if _dynamic_models:
        logger.error(
            "No enabled models found in registry (%d models registered but all disabled)",
            len(_dynamic_models),
        )
    else:
        logger.error("No models registered in LLM registry")

    return None
