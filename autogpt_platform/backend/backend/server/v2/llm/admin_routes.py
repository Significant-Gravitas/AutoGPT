"""Admin API for LLM registry management.

Provides endpoints for:
- Reading creators (GET)
- Creating, updating, and deleting models
- Creating, updating, and deleting providers

All endpoints require admin authentication. Mutations refresh the registry cache.
"""

import logging
from typing import Any

import prisma
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


def _map_creator_response(creator: Any) -> dict[str, Any]:
    """Map Prisma creator model to response dict."""
    return {
        "id": creator.id,
        "name": creator.name,
        "display_name": creator.displayName,
        "description": creator.description,
        "website_url": creator.websiteUrl,
        "logo_url": creator.logoUrl,
        "metadata": dict(creator.metadata or {}),
        "created_at": creator.createdAt.isoformat() if creator.createdAt else None,
        "updated_at": creator.updatedAt.isoformat() if creator.updatedAt else None,
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
        import prisma.models as pm

        # Resolve provider name to ID
        provider = await pm.LlmProvider.prisma().find_unique(
            where={"name": request.provider_id}
        )
        if not provider:
            # Try as UUID fallback
            provider = await pm.LlmProvider.prisma().find_unique(
                where={"id": request.provider_id}
            )
        if not provider:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{request.provider_id}' not found",
            )

        model = await db_write.create_model(
            slug=request.slug,
            display_name=request.display_name,
            provider_id=provider.id,
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
        # Create costs if provided in the raw request body
        if hasattr(request, 'costs') and request.costs:
            for cost_input in request.costs:
                await pm.LlmModelCost.prisma().create(
                    data={
                        "unit": cost_input.get("unit", "RUN"),
                        "creditCost": int(cost_input.get("credit_cost", 1)),
                        "credentialProvider": provider.name,
                        "metadata": prisma.Json(cost_input.get("metadata", {})),
                        "Model": {"connect": {"id": model.id}},
                    }
                )

        await db_write.refresh_runtime_caches()
        logger.info(f"Created model '{request.slug}' (id: {model.id})")

        # Re-fetch with costs included
        model = await pm.LlmModel.prisma().find_unique(
            where={"id": model.id},
            include={"Costs": True, "Creator": True},
        )
        return _map_model_response(model)
    except ValueError as e:
        logger.warning(f"Model creation validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail="Failed to create model")


@router.patch(
    "/llm/models/{slug:path}",
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
    "/llm/models/{slug:path}",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def delete_model(
    slug: str,
    replacement_model_slug: str | None = None,
) -> dict[str, Any]:
    """Delete an LLM model with optional migration.

    If workflows are using this model and no replacement_model_slug is given,
    returns 400 with the node count. Provide replacement_model_slug to migrate
    affected nodes before deletion.
    """
    try:
        import prisma.models

        existing = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": slug}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Model with slug '{slug}' not found"
            )

        result = await db_write.delete_model(
            model_id=existing.id,
            replacement_model_slug=replacement_model_slug,
        )
        await db_write.refresh_runtime_caches()
        logger.info(
            f"Deleted model '{slug}' (migrated {result['nodes_migrated']} nodes)"
        )
        return result
    except ValueError as e:
        logger.warning(f"Model deletion validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")


@router.get(
    "/llm/models/{slug:path}/usage",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def get_model_usage(slug: str) -> dict[str, Any]:
    """Get usage count for a model — how many workflow nodes reference it."""
    try:
        return await db_write.get_model_usage(slug)
    except Exception as e:
        logger.exception(f"Failed to get model usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model usage")


@router.post(
    "/llm/models/{slug:path}/toggle",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def toggle_model(
    slug: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Toggle a model's enabled status with optional migration when disabling.

    Body params:
        is_enabled: bool
        migrate_to_slug: optional str
        migration_reason: optional str
        custom_credit_cost: optional int
    """
    try:
        import prisma.models

        existing = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": slug}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Model with slug '{slug}' not found"
            )

        result = await db_write.toggle_model_with_migration(
            model_id=existing.id,
            is_enabled=request.get("is_enabled", True),
            migrate_to_slug=request.get("migrate_to_slug"),
            migration_reason=request.get("migration_reason"),
            custom_credit_cost=request.get("custom_credit_cost"),
        )
        await db_write.refresh_runtime_caches()
        logger.info(
            f"Toggled model '{slug}' enabled={request.get('is_enabled')} "
            f"(migrated {result['nodes_migrated']} nodes)"
        )
        return result
    except ValueError as e:
        logger.warning(f"Model toggle failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to toggle model: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle model")


@router.get(
    "/llm/migrations",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def list_migrations(
    include_reverted: bool = False,
) -> dict[str, Any]:
    """List model migrations."""
    try:
        migrations = await db_write.list_migrations(
            include_reverted=include_reverted
        )
        return {"migrations": migrations}
    except Exception as e:
        logger.exception(f"Failed to list migrations: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to list migrations"
        )


@router.post(
    "/llm/migrations/{migration_id}/revert",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def revert_migration(
    migration_id: str,
    re_enable_source_model: bool = True,
) -> dict[str, Any]:
    """Revert a model migration, restoring affected nodes."""
    try:
        result = await db_write.revert_migration(
            migration_id=migration_id,
            re_enable_source_model=re_enable_source_model,
        )
        await db_write.refresh_runtime_caches()
        logger.info(
            f"Reverted migration {migration_id}: "
            f"{result['nodes_reverted']} nodes restored"
        )
        return result
    except ValueError as e:
        logger.warning(f"Migration revert failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to revert migration: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to revert migration"
        )


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


@router.get(
    "/llm/admin/providers",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def admin_list_providers() -> dict[str, Any]:
    """List all LLM providers from the database.

    Unlike the public endpoint, this returns ALL providers including
    those with no models. Requires admin authentication.
    """
    try:
        import prisma.models

        providers = await prisma.models.LlmProvider.prisma().find_many(
            order={"name": "asc"},
            include={"Models": True},
        )
        return {
            "providers": [
                {**_map_provider_response(p), "model_count": len(p.Models) if p.Models else 0}
                for p in providers
            ]
        }
    except Exception as e:
        logger.exception(f"Failed to list providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to list providers")


@router.get(
    "/llm/admin/models",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def admin_list_models(
    page: int = 1,
    page_size: int = 100,
    enabled_only: bool = False,
) -> dict[str, Any]:
    """List all LLM models from the database.

    Unlike the public endpoint, this returns full model data including
    costs and creator info. Requires admin authentication.
    """
    try:
        import prisma.models

        where = {"isEnabled": True} if enabled_only else {}
        models = await prisma.models.LlmModel.prisma().find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order={"displayName": "asc"},
            include={"Costs": True, "Creator": True},
        )
        return {
            "models": [
                {
                    **_map_model_response(m),
                    "creator": _map_creator_response(m.Creator) if m.Creator else None,
                    "costs": [
                        {
                            "unit": c.unit,
                            "credit_cost": float(c.creditCost),
                            "credential_provider": c.credentialProvider,
                            "credential_type": c.credentialType,
                            "metadata": dict(c.metadata or {}),
                        }
                        for c in (m.Costs or [])
                    ],
                }
                for m in models
            ]
        }
    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get(
    "/llm/creators",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def list_creators() -> dict[str, Any]:
    """List all LLM model creators.

    Requires admin authentication.
    """
    try:
        import prisma.models

        creators = await prisma.models.LlmModelCreator.prisma().find_many(
            order={"name": "asc"}
        )
        logger.info(f"Retrieved {len(creators)} creators")
        return {"creators": [_map_creator_response(c) for c in creators]}
    except Exception as e:
        logger.exception(f"Failed to list creators: {e}")
        raise HTTPException(status_code=500, detail="Failed to list creators")


@router.post(
    "/llm/creators",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def create_creator(
    request: dict[str, Any],
) -> dict[str, Any]:
    """Create a new LLM model creator."""
    try:
        import prisma.models

        creator = await prisma.models.LlmModelCreator.prisma().create(
            data={
                "name": request["name"],
                "displayName": request["display_name"],
                "description": request.get("description"),
                "websiteUrl": request.get("website_url"),
                "logoUrl": request.get("logo_url"),
                "metadata": prisma.Json(request.get("metadata", {})),
            }
        )
        logger.info(f"Created creator '{creator.name}' (id: {creator.id})")
        return _map_creator_response(creator)
    except Exception as e:
        logger.exception(f"Failed to create creator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/llm/creators/{name}",
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def update_creator(
    name: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Update an existing LLM model creator."""
    try:
        import prisma.models

        existing = await prisma.models.LlmModelCreator.prisma().find_unique(
            where={"name": name}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Creator '{name}' not found"
            )

        data: dict[str, Any] = {}
        if "display_name" in request:
            data["displayName"] = request["display_name"]
        if "description" in request:
            data["description"] = request["description"]
        if "website_url" in request:
            data["websiteUrl"] = request["website_url"]
        if "logo_url" in request:
            data["logoUrl"] = request["logo_url"]

        creator = await prisma.models.LlmModelCreator.prisma().update(
            where={"id": existing.id},
            data=data,
        )
        logger.info(f"Updated creator '{name}' (id: {creator.id})")
        return _map_creator_response(creator)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update creator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/llm/creators/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Security(autogpt_libs.auth.requires_admin_user)],
)
async def delete_creator(
    name: str,
) -> None:
    """Delete an LLM model creator."""
    try:
        import prisma.models

        existing = await prisma.models.LlmModelCreator.prisma().find_unique(
            where={"name": name},
            include={"Models": True},
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Creator '{name}' not found"
            )

        if existing.Models and len(existing.Models) > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete creator '{name}' — it has {len(existing.Models)} associated models",
            )

        await prisma.models.LlmModelCreator.prisma().delete(
            where={"id": existing.id}
        )
        logger.info(f"Deleted creator '{name}' (id: {existing.id})")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete creator: {e}")
        raise HTTPException(status_code=500, detail=str(e))
