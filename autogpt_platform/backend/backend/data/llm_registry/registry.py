"""Core LLM registry implementation for managing models dynamically."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import prisma.models

from backend.data.llm_registry.model import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryModelCost:
    """Cost configuration for an LLM model."""

    unit: str  # "RUN" or "TOKENS"
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


# In-memory cache (will be replaced with Redis in PR #6)
_dynamic_models: dict[str, RegistryModel] = {}
_schema_options: list[dict[str, str]] = []
_lock = asyncio.Lock()


async def refresh_llm_registry() -> None:
    """
    Refresh the LLM registry from the database.

    Fetches all models with their costs, providers, and creators,
    then updates the in-memory cache.
    """
    async with _lock:
        try:
            records = await prisma.models.LlmModel.prisma().find_many(
                include={
                    "Provider": True,
                    "Costs": True,
                    "Creator": True,
                }
            )
            logger.info(f"Fetched {len(records)} LLM models from database")

            # Build model instances
            new_models: dict[str, RegistryModel] = {}
            for record in records:
                # Parse costs
                costs = tuple(
                    RegistryModelCost(
                        unit=str(cost.unit),  # Convert enum to string
                        credit_cost=cost.creditCost,
                        credential_provider=cost.credentialProvider,
                        credential_id=cost.credentialId,
                        credential_type=cost.credentialType,
                        currency=cost.currency,
                        metadata=dict(cost.metadata or {}),
                    )
                    for cost in (record.Costs or [])
                )

                # Parse creator
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

                # Parse capabilities
                capabilities = dict(record.capabilities or {})

                # Build metadata from record
                # Warn if Provider relation is missing (indicates data corruption)
                if not record.Provider:
                    logger.warning(
                        f"LlmModel {record.slug} has no Provider despite NOT NULL FK - "
                        f"falling back to providerId {record.providerId}"
                    )
                provider_name = (
                    record.Provider.name if record.Provider else record.providerId
                )
                provider_display = (
                    record.Provider.displayName
                    if record.Provider
                    else record.providerId
                )

                # Extract creator name (fallback to "Unknown" if no creator)
                creator_name = (
                    record.Creator.displayName if record.Creator else "Unknown"
                )

                # Price tier defaults to 1 if not set
                price_tier = record.priceTier if record.priceTier in (1, 2, 3) else 1

                metadata = ModelMetadata(
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

                # Create model instance
                model = RegistryModel(
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
                )
                new_models[record.slug] = model

            # Atomic swap
            global _dynamic_models, _schema_options
            _dynamic_models = new_models
            _schema_options = _build_schema_options()

            logger.info(
                f"LLM registry refreshed: {len(_dynamic_models)} models, "
                f"{len(_schema_options)} schema options"
            )
        except Exception as e:
            logger.error(f"Failed to refresh LLM registry: {e}", exc_info=True)
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
    return _schema_options


def get_default_model_slug() -> str | None:
    """Get the default model slug (first recommended, or first enabled)."""
    # Sort once and use next() to short-circuit on first match
    models = sorted(_dynamic_models.values(), key=lambda m: m.display_name)

    # Prefer recommended models
    recommended = next(
        (m.slug for m in models if m.is_recommended and m.is_enabled), None
    )
    if recommended:
        return recommended

    # Fallback to first enabled model
    return next((m.slug for m in models if m.is_enabled), None)


def get_all_model_slugs_for_validation() -> list[str]:
    """
    Get all model slugs for validation (enables migrate_llm_models to work).
    Returns slugs for enabled models only.
    """
    return [model.slug for model in _dynamic_models.values() if model.is_enabled]
