"""Admin write API for LLM registry management.

Provides endpoints for creating, updating, and deleting:
- Models
- Providers

All endpoints require admin authentication and refresh the registry cache after mutations.
"""

import logging
from typing import Any

import autogpt_libs.auth
from fastapi import APIRouter, HTTPException, Security, status

from backend.server.v2.llm import db_write
from backend.server.v2.llm.admin_model import (
    CreateLlmModelRequest,
    CreateLlmProviderRequest,
    UpdateLlmModelRequest,
    UpdateLlmProviderRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _map_provider_response(provider: Any) -> dict[str, Any]:
    """Map Prisma provider model to response dict."""
    return {
        "id": provider.id,
        "name": provider.name,
        "display_name": provider.displayName,
        "description": provider.description,
        "default_credential_provider": provider.defaultCredentialProvider,
        "default_credential_id": provider.defaultCredentialId,
        "default_credential_type": provider.defaultCredentialType,
        "metadata": dict(provider.metadata or {}),
        "created_at": provider.createdAt.isoformat() if provider.createdAt else None,
        "updated_at": provider.updatedAt.isoformat() if provider.updatedAt else None,
    }


def _map_model_response(model: Any) -> dict[str, Any]:
    """Map Prisma model to response dict."""
    return {
        "id": model.id,
        "slug": model.slug,
        "display_name": model.displayName,
        "description": model.description,
        "provider_id": model.providerId,
        "creator_id": model.creatorId,
        "context_window": model.contextWindow,
        "max_output_tokens": model.maxOutputTokens,
        "price_tier": model.priceTier,
        "is_enabled": model.isEnabled,
        "is_recommended": model.isRecommended,
        "supports_tools": model.supportsTools,
        "supports_json_output": model.supportsJsonOutput,
        "supports_reasoning": model.supportsReasoning,
        "supports_parallel_tool_calls": model.supportsParallelToolCalls,
        "capabilities": dict(model.capabilities or {}),
        "metadata": dict(model.metadata or {}),
        "created_at": model.createdAt.isoformat() if model.createdAt else None,
        "updated_at": model.updatedAt.isoformat() if model.updatedAt else None,
    }


@router.post(
    "/llm/models",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def create_model(
    request: CreateLlmModelRequest,
) -> dict[str, Any]:
    """Create a new LLM model.

    Requires admin authentication.
    """
    try:
        model = await db_write.create_model(
            slug=request.slug,
            display_name=request.display_name,
            provider_id=request.provider_id,
            context_window=request.context_window,
            price_tier=request.price_tier,
            description=request.description,
            creator_id=request.creator_id,
            max_output_tokens=request.max_output_tokens,
            is_enabled=request.is_enabled,
            is_recommended=request.is_recommended,
            supports_tools=request.supports_tools,
            supports_json_output=request.supports_json_output,
            supports_reasoning=request.supports_reasoning,
            supports_parallel_tool_calls=request.supports_parallel_tool_calls,
            capabilities=request.capabilities,
            metadata=request.metadata,
        )
        await db_write.refresh_runtime_caches()
        logger.info(f"Created model '{request.slug}' (id: {model.id})")
        return _map_model_response(model)
    except ValueError as e:
        logger.warning(f"Model creation validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail="Failed to create model")


@router.patch(
    "/llm/models/{slug}",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def update_model(
    slug: str,
    request: UpdateLlmModelRequest,
) -> dict[str, Any]:
    """Update an existing LLM model.

    Requires admin authentication.
    """
    try:
        # Find model by slug first to get ID
        import prisma.models

        existing = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": slug}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Model with slug '{slug}' not found"
            )

        model = await db_write.update_model(
            model_id=existing.id,
            display_name=request.display_name,
            description=request.description,
            creator_id=request.creator_id,
            context_window=request.context_window,
            max_output_tokens=request.max_output_tokens,
            price_tier=request.price_tier,
            is_enabled=request.is_enabled,
            is_recommended=request.is_recommended,
            supports_tools=request.supports_tools,
            supports_json_output=request.supports_json_output,
            supports_reasoning=request.supports_reasoning,
            supports_parallel_tool_calls=request.supports_parallel_tool_calls,
            capabilities=request.capabilities,
            metadata=request.metadata,
        )
        await db_write.refresh_runtime_caches()
        logger.info(f"Updated model '{slug}' (id: {model.id})")
        return _map_model_response(model)
    except ValueError as e:
        logger.warning(f"Model update validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to update model: {e}")
        raise HTTPException(status_code=500, detail="Failed to update model")


@router.delete(
    "/llm/models/{slug}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def delete_model(
    slug: str,
) -> None:
    """Delete an LLM model.

    Requires admin authentication.
    """
    try:
        # Find model by slug first to get ID
        import prisma.models

        existing = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": slug}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Model with slug '{slug}' not found"
            )

        await db_write.delete_model(model_id=existing.id)
        await db_write.refresh_runtime_caches()
        logger.info(f"Deleted model '{slug}' (id: {existing.id})")
    except ValueError as e:
        logger.warning(f"Model deletion validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")


@router.post(
    "/llm/providers",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def create_provider(
    request: CreateLlmProviderRequest,
) -> dict[str, Any]:
    """Create a new LLM provider.

    Requires admin authentication.
    """
    try:
        provider = await db_write.create_provider(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            default_credential_provider=request.default_credential_provider,
            default_credential_id=request.default_credential_id,
            default_credential_type=request.default_credential_type,
            metadata=request.metadata,
        )
        await db_write.refresh_runtime_caches()
        logger.info(f"Created provider '{request.name}' (id: {provider.id})")
        return _map_provider_response(provider)
    except ValueError as e:
        logger.warning(f"Provider creation validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to create provider")


@router.patch(
    "/llm/providers/{name}",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def update_provider(
    name: str,
    request: UpdateLlmProviderRequest,
) -> dict[str, Any]:
    """Update an existing LLM provider.

    Requires admin authentication.
    """
    try:
        # Find provider by name first to get ID
        import prisma.models

        existing = await prisma.models.LlmProvider.prisma().find_unique(
            where={"name": name}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Provider with name '{name}' not found"
            )

        provider = await db_write.update_provider(
            provider_id=existing.id,
            display_name=request.display_name,
            description=request.description,
            default_credential_provider=request.default_credential_provider,
            default_credential_id=request.default_credential_id,
            default_credential_type=request.default_credential_type,
            metadata=request.metadata,
        )
        await db_write.refresh_runtime_caches()
        logger.info(f"Updated provider '{name}' (id: {provider.id})")
        return _map_provider_response(provider)
    except ValueError as e:
        logger.warning(f"Provider update validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to update provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to update provider")


@router.delete(
    "/llm/providers/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def delete_provider(
    name: str,
) -> None:
    """Delete an LLM provider.

    Requires admin authentication.
    A provider can only be deleted if it has no associated models.
    """
    try:
        # Find provider by name first to get ID
        import prisma.models

        existing = await prisma.models.LlmProvider.prisma().find_unique(
            where={"name": name}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Provider with name '{name}' not found"
            )

        await db_write.delete_provider(provider_id=existing.id)
        await db_write.refresh_runtime_caches()
        logger.info(f"Deleted provider '{name}' (id: {existing.id})")
    except ValueError as e:
        logger.warning(f"Provider deletion validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to delete provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete provider")
