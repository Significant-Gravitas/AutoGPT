"""Core LLM registry implementation for managing models dynamically."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import prisma.models

from backend.data.llm_registry.model_types import ModelMetadata

logger = logging.getLogger(__name__)


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


async def refresh_llm_registry() -> None:
    """Refresh the LLM registry from the database. Loads all models (enabled and disabled)."""
    async with _lock:
        try:
            records = await prisma.models.LlmModel.prisma().find_many(
                include={
                    "Provider": True,
                    "Costs": True,
                    "Creator": True,
                }
            )
            logger.debug("Found %d LLM model records in database", len(records))
        except Exception as exc:
            logger.error(
                "Failed to refresh LLM registry from DB: %s", exc, exc_info=True
            )
            return

        dynamic: dict[str, RegistryModel] = {}
        for record in records:
            provider_name = (
                record.Provider.name if record.Provider else record.providerId
            )
            metadata = ModelMetadata(
                provider=provider_name,
                context_window=record.contextWindow,
                max_output_tokens=record.maxOutputTokens,
            )
            costs = tuple(
                RegistryModelCost(
                    credit_cost=cost.creditCost,
                    credential_provider=cost.credentialProvider,
                    credential_id=cost.credentialId,
                    credential_type=cost.credentialType,
                    currency=cost.currency,
                    metadata=cost.metadata or {},
                )
                for cost in (record.Costs or [])
            )

            # Map creator if present
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

            dynamic[record.slug] = RegistryModel(
                slug=record.slug,
                display_name=record.displayName,
                description=record.description,
                metadata=metadata,
                capabilities=record.capabilities or {},
                extra_metadata=record.metadata or {},
                provider_display_name=(
                    record.Provider.displayName
                    if record.Provider
                    else record.providerId
                ),
                is_enabled=record.isEnabled,
                costs=costs,
                creator=creator,
            )

        _dynamic_models.clear()
        _dynamic_models.update(dynamic)
        _refresh_cached_schema()
        logger.info(
            "LLM registry refreshed with %s dynamic models (enabled: %s, disabled: %s)",
            len(dynamic),
            sum(1 for m in dynamic.values() if m.is_enabled),
            sum(1 for m in dynamic.values() if not m.is_enabled),
        )


def _refresh_cached_schema() -> None:
    """Refresh cached schema options and discriminator mapping."""
    new_options = _build_schema_options()
    _schema_options.clear()
    _schema_options.extend(new_options)
    _discriminator_mapping.clear()
    _discriminator_mapping.update(
        {slug: entry.metadata.provider for slug, entry in _dynamic_models.items()}
    )
    for slug, metadata in _static_metadata.items():
        _discriminator_mapping.setdefault(slug, metadata.provider)


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

    Returns cached schema options that are refreshed when the registry is updated
    via refresh_llm_registry() (called on startup and via Redis pub/sub notifications).
    """
    return _schema_options


def get_llm_discriminator_mapping() -> dict[str, str]:
    """
    Get discriminator mapping for LLM models.

    Returns cached discriminator mapping that is refreshed when the registry is updated
    via refresh_llm_registry() (called on startup and via Redis pub/sub notifications).
    """
    return _discriminator_mapping


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


def get_default_model_slug() -> str:
    """
    Get the default model slug to use for block defaults.

    Prefers "gpt-4o" if it exists and is enabled, otherwise returns
    the first enabled model from the registry, or "gpt-4o" as fallback.
    """
    # Prefer gpt-4o if available and enabled
    preferred_slug = "gpt-4o"
    preferred_model = _dynamic_models.get(preferred_slug)
    if preferred_model and preferred_model.is_enabled:
        return preferred_slug

    # Find first enabled model
    for model in sorted(_dynamic_models.values(), key=lambda m: m.display_name.lower()):
        if model.is_enabled:
            return model.slug

    # Fallback to preferred slug even if not in registry (for backwards compatibility)
    return preferred_slug

