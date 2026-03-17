"""Database write operations for LLM registry admin API."""

from typing import Any

import prisma
import prisma.models

from backend.data import llm_registry


def _build_provider_data(
    name: str,
    display_name: str,
    description: str | None = None,
    default_credential_provider: str | None = None,
    default_credential_id: str | None = None,
    default_credential_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provider data dict for Prisma operations."""
    return {
        "name": name,
        "displayName": display_name,
        "description": description,
        "defaultCredentialProvider": default_credential_provider,
        "defaultCredentialId": default_credential_id,
        "defaultCredentialType": default_credential_type,
        "metadata": prisma.Json(metadata or {}),
    }


def _build_model_data(
    slug: str,
    display_name: str,
    provider_id: str,
    context_window: int,
    price_tier: int,
    description: str | None = None,
    creator_id: str | None = None,
    max_output_tokens: int | None = None,
    is_enabled: bool = True,
    is_recommended: bool = False,
    supports_tools: bool = False,
    supports_json_output: bool = False,
    supports_reasoning: bool = False,
    supports_parallel_tool_calls: bool = False,
    capabilities: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build model data dict for Prisma operations."""
    data: dict[str, Any] = {
        "slug": slug,
        "displayName": display_name,
        "description": description,
        "Provider": {"connect": {"id": provider_id}},
        "contextWindow": context_window,
        "maxOutputTokens": max_output_tokens,
        "priceTier": price_tier,
        "isEnabled": is_enabled,
        "isRecommended": is_recommended,
        "supportsTools": supports_tools,
        "supportsJsonOutput": supports_json_output,
        "supportsReasoning": supports_reasoning,
        "supportsParallelToolCalls": supports_parallel_tool_calls,
        "capabilities": prisma.Json(capabilities or {}),
        "metadata": prisma.Json(metadata or {}),
    }
    if creator_id:
        data["Creator"] = {"connect": {"id": creator_id}}
    return data


async def create_provider(
    name: str,
    display_name: str,
    description: str | None = None,
    default_credential_provider: str | None = None,
    default_credential_id: str | None = None,
    default_credential_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmProvider:
    """Create a new LLM provider."""
    data = _build_provider_data(
        name=name,
        display_name=display_name,
        description=description,
        default_credential_provider=default_credential_provider,
        default_credential_id=default_credential_id,
        default_credential_type=default_credential_type,
        metadata=metadata,
    )
    provider = await prisma.models.LlmProvider.prisma().create(
        data=data,
        include={"Models": True},
    )
    if not provider:
        raise ValueError("Failed to create provider")
    return provider


async def update_provider(
    provider_id: str,
    display_name: str | None = None,
    description: str | None = None,
    default_credential_provider: str | None = None,
    default_credential_id: str | None = None,
    default_credential_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmProvider:
    """Update an existing LLM provider."""
    # Fetch existing provider to get current name
    provider = await prisma.models.LlmProvider.prisma().find_unique(
        where={"id": provider_id}
    )
    if not provider:
        raise ValueError(f"Provider with id '{provider_id}' not found")

    # Build update data (only include fields that are provided)
    data: dict[str, Any] = {}
    if display_name is not None:
        data["displayName"] = display_name
    if description is not None:
        data["description"] = description
    if default_credential_provider is not None:
        data["defaultCredentialProvider"] = default_credential_provider
    if default_credential_id is not None:
        data["defaultCredentialId"] = default_credential_id
    if default_credential_type is not None:
        data["defaultCredentialType"] = default_credential_type
    if metadata is not None:
        data["metadata"] = prisma.Json(metadata)

    updated = await prisma.models.LlmProvider.prisma().update(
        where={"id": provider_id},
        data=data,
        include={"Models": True},
    )
    if not updated:
        raise ValueError("Failed to update provider")
    return updated


async def delete_provider(provider_id: str) -> bool:
    """Delete an LLM provider.

    A provider can only be deleted if it has no associated models.
    """
    # Check if provider exists
    provider = await prisma.models.LlmProvider.prisma().find_unique(
        where={"id": provider_id},
        include={"Models": True},
    )
    if not provider:
        raise ValueError(f"Provider with id '{provider_id}' not found")

    # Check if provider has any models
    model_count = len(provider.Models) if provider.Models else 0
    if model_count > 0:
        raise ValueError(
            f"Cannot delete provider '{provider.displayName}' because it has "
            f"{model_count} model(s). Delete all models first."
        )

    await prisma.models.LlmProvider.prisma().delete(where={"id": provider_id})
    return True


async def create_model(
    slug: str,
    display_name: str,
    provider_id: str,
    context_window: int,
    price_tier: int,
    description: str | None = None,
    creator_id: str | None = None,
    max_output_tokens: int | None = None,
    is_enabled: bool = True,
    is_recommended: bool = False,
    supports_tools: bool = False,
    supports_json_output: bool = False,
    supports_reasoning: bool = False,
    supports_parallel_tool_calls: bool = False,
    capabilities: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmModel:
    """Create a new LLM model."""
    data = _build_model_data(
        slug=slug,
        display_name=display_name,
        provider_id=provider_id,
        context_window=context_window,
        price_tier=price_tier,
        description=description,
        creator_id=creator_id,
        max_output_tokens=max_output_tokens,
        is_enabled=is_enabled,
        is_recommended=is_recommended,
        supports_tools=supports_tools,
        supports_json_output=supports_json_output,
        supports_reasoning=supports_reasoning,
        supports_parallel_tool_calls=supports_parallel_tool_calls,
        capabilities=capabilities,
        metadata=metadata,
    )
    model = await prisma.models.LlmModel.prisma().create(
        data=data,
        include={"Costs": True, "Creator": True, "Provider": True},
    )
    if not model:
        raise ValueError("Failed to create model")
    return model


async def update_model(
    model_id: str,
    display_name: str | None = None,
    description: str | None = None,
    creator_id: str | None = None,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    price_tier: int | None = None,
    is_enabled: bool | None = None,
    is_recommended: bool | None = None,
    supports_tools: bool | None = None,
    supports_json_output: bool | None = None,
    supports_reasoning: bool | None = None,
    supports_parallel_tool_calls: bool | None = None,
    capabilities: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> prisma.models.LlmModel:
    """Update an existing LLM model."""
    # Build update data (only include fields that are provided)
    data: dict[str, Any] = {}
    if display_name is not None:
        data["displayName"] = display_name
    if description is not None:
        data["description"] = description
    if context_window is not None:
        data["contextWindow"] = context_window
    if max_output_tokens is not None:
        data["maxOutputTokens"] = max_output_tokens
    if price_tier is not None:
        data["priceTier"] = price_tier
    if is_enabled is not None:
        data["isEnabled"] = is_enabled
    if is_recommended is not None:
        data["isRecommended"] = is_recommended
    if supports_tools is not None:
        data["supportsTools"] = supports_tools
    if supports_json_output is not None:
        data["supportsJsonOutput"] = supports_json_output
    if supports_reasoning is not None:
        data["supportsReasoning"] = supports_reasoning
    if supports_parallel_tool_calls is not None:
        data["supportsParallelToolCalls"] = supports_parallel_tool_calls
    if capabilities is not None:
        data["capabilities"] = prisma.Json(capabilities)
    if metadata is not None:
        data["metadata"] = prisma.Json(metadata)
    if creator_id is not None:
        data["creatorId"] = creator_id if creator_id else None

    model = await prisma.models.LlmModel.prisma().update(
        where={"id": model_id},
        data=data,
        include={"Costs": True, "Creator": True, "Provider": True},
    )
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")
    return model


async def delete_model(model_id: str) -> bool:
    """Delete an LLM model.

    Note: This should check if any workflows are using this model first.
    For now, we'll allow deletion and rely on FK constraints.
    """
    # Check if model exists
    model = await prisma.models.LlmModel.prisma().find_unique(where={"id": model_id})
    if not model:
        raise ValueError(f"Model with id '{model_id}' not found")

    await prisma.models.LlmModel.prisma().delete(where={"id": model_id})
    return True


async def refresh_runtime_caches() -> None:
    """Refresh the LLM registry and clear all related caches."""
    # Refresh the in-memory registry
    await llm_registry.refresh_llm_registry()

    # TODO: Clear block schema caches when block integration is implemented
    # TODO: Publish registry refresh notification to executors
