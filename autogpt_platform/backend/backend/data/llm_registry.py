from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import prisma.models

from backend.data.llm_model_types import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryModelCost:
    credit_cost: int
    credential_provider: str
    credential_id: str | None
    credential_type: str | None
    currency: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RegistryModel:
    slug: str
    display_name: str
    description: str | None
    metadata: ModelMetadata
    capabilities: dict[str, Any]
    extra_metadata: dict[str, Any]
    provider_display_name: str
    is_enabled: bool
    costs: tuple[RegistryModelCost, ...] = field(default_factory=tuple)


_static_metadata: dict[str, ModelMetadata] = {}
_static_costs: dict[str, int] = {}
_dynamic_models: dict[str, RegistryModel] = {}
_schema_options: list[dict[str, str]] = []
_discriminator_mapping: dict[str, str] = {}
_lock = asyncio.Lock()


def register_static_metadata(metadata: dict[Any, ModelMetadata]) -> None:
    _static_metadata.update({str(key): value for key, value in metadata.items()})
    _refresh_cached_schema()


def register_static_costs(costs: dict[Any, int]) -> None:
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


# Removed get_llm_model_metadata_async - direct database queries don't work in executor context
# The registry should be refreshed on startup via initialize_blocks() or rest_api lifespan


def get_llm_model_cost(slug: str) -> tuple[RegistryModelCost, ...]:
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
    return set(_dynamic_models.keys())


def iter_dynamic_models() -> Iterable[RegistryModel]:
    return tuple(_dynamic_models.values())
