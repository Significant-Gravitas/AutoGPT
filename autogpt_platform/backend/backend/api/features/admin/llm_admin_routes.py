"""Admin API for LLM registry management.

Mounted at /api/admin/llm. Every mutation writes an AuditLog row and claims
the touched row for the admin (source=LOCAL — the catalog importer keeps
hands off from then on), then refreshes the runtime caches across pods.
"""

import logging
from typing import Any

import prisma
import prisma.models
from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Security, status

import backend.data.llm_registry as llm_registry
from backend.api.features.admin.llm_admin_model import (
    CreateLlmCreatorRequest,
    CreateLlmModelRequest,
    LlmCreatorAdminResponse,
    LlmCreatorsAdminListResponse,
    LlmMigrationAdminResponse,
    LlmMigrationsAdminListResponse,
    LlmModelAdminResponse,
    LlmModelCostAdminResponse,
    LlmModelsAdminListResponse,
    LlmProviderAdminResponse,
    LlmProvidersAdminListResponse,
    LlmRouteResponse,
    LlmRoutesListResponse,
    SetLlmRouteRequest,
    SetLlmRouteResponse,
    ToggleLlmModelRequest,
    UpdateLlmCreatorRequest,
    UpdateLlmModelRequest,
)
from backend.copilot.route_warnings import RouteWarning, get_route_warnings
from backend.data.llm_registry import db_routes, db_write
from backend.data.llm_registry.catalog_model import CatalogPayload
from backend.data.llm_registry.db_routes import UnknownRouteModelError

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Security(requires_admin_user)])


async def _audit(
    actor_user_id: str,
    entity_type: str,
    entity_id: str,
    action: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> None:
    """Persist an AuditLog row; failures log but never fail the admin op."""
    try:
        data: dict[str, Any] = {
            "actorUserId": actor_user_id,
            "entityType": entity_type,
            "entityId": entity_id,
            "action": action,
        }
        if before is not None:
            data["beforeJson"] = prisma.Json(before)
        if after is not None:
            data["afterJson"] = prisma.Json(after)
        await prisma.models.AuditLog.prisma().create(data=data)
    except Exception:
        logger.exception(
            "Failed to write AuditLog for %s %s on %s", action, entity_type, entity_id
        )


def _model_snapshot(model: prisma.models.LlmModel) -> dict[str, Any]:
    return {
        "slug": model.slug,
        "displayName": model.displayName,
        "isEnabled": model.isEnabled,
        "isRecommended": model.isRecommended,
        "visibility": str(model.visibility),
        "priceTier": model.priceTier,
        "contextWindow": model.contextWindow,
        "fallbackModelSlug": model.fallbackModelSlug,
        "source": str(model.source),
    }


def _map_creator(creator: prisma.models.LlmModelCreator) -> LlmCreatorAdminResponse:
    return LlmCreatorAdminResponse(
        id=creator.id,
        name=creator.name,
        display_name=creator.displayName,
        description=creator.description,
        website_url=creator.websiteUrl,
        logo_url=creator.logoUrl,
        source=str(creator.source),
        metadata=dict(creator.metadata or {}),
        created_at=creator.createdAt.isoformat() if creator.createdAt else None,
        updated_at=creator.updatedAt.isoformat() if creator.updatedAt else None,
    )


def _map_provider(
    provider: prisma.models.LlmProvider, model_count: int | None = None
) -> LlmProviderAdminResponse:
    return LlmProviderAdminResponse(
        id=provider.id,
        name=provider.name,
        display_name=provider.displayName,
        description=provider.description,
        source=str(provider.source),
        metadata=dict(provider.metadata or {}),
        created_at=provider.createdAt.isoformat() if provider.createdAt else None,
        updated_at=provider.updatedAt.isoformat() if provider.updatedAt else None,
        model_count=model_count,
    )


def _map_model(model: prisma.models.LlmModel) -> LlmModelAdminResponse:
    return LlmModelAdminResponse(
        id=model.id,
        slug=model.slug,
        display_name=model.displayName,
        description=model.description,
        provider_id=model.providerId,
        creator_id=model.creatorId,
        context_window=model.contextWindow,
        max_output_tokens=model.maxOutputTokens,
        price_tier=model.priceTier,
        is_enabled=model.isEnabled,
        is_recommended=model.isRecommended,
        kind=str(model.kind),
        visibility=str(model.visibility),
        min_subscription_tier=(
            str(model.minSubscriptionTier)
            if model.minSubscriptionTier is not None
            else None
        ),
        fallback_model_slug=model.fallbackModelSlug,
        source=str(model.source),
        catalog_removed_at=(
            model.catalogRemovedAt.isoformat() if model.catalogRemovedAt else None
        ),
        supports_tools=model.supportsTools,
        supports_json_output=model.supportsJsonOutput,
        supports_reasoning=model.supportsReasoning,
        supports_parallel_tool_calls=model.supportsParallelToolCalls,
        capabilities=dict(model.capabilities or {}),
        metadata=dict(model.metadata or {}),
        created_at=model.createdAt.isoformat() if model.createdAt else None,
        updated_at=model.updatedAt.isoformat() if model.updatedAt else None,
        creator=_map_creator(model.Creator) if model.Creator else None,
        costs=[
            LlmModelCostAdminResponse(
                unit=str(c.unit),
                credit_cost=float(c.creditCost),
                credential_provider=c.credentialProvider,
                credential_type=c.credentialType,
                metadata=dict(c.metadata or {}),
            )
            for c in (model.Costs or [])
        ],
    )


def _map_route(route: prisma.models.LlmModelRoute) -> LlmRouteResponse:
    return LlmRouteResponse(
        surface=route.surface,
        mode=route.mode,
        tier=route.tier,
        model_slug=route.modelSlug,
        updated_at=route.updatedAt.isoformat() if route.updatedAt else None,
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@router.post("/models", status_code=status.HTTP_201_CREATED)
async def create_model(
    request: CreateLlmModelRequest,
    admin_user_id: str = Security(get_user_id),
) -> LlmModelAdminResponse:
    try:
        provider = await prisma.models.LlmProvider.prisma().find_unique(
            where={"name": request.provider_name}
        )
        if not provider:
            provider = await prisma.models.LlmProvider.prisma().find_unique(
                where={"id": request.provider_name}
            )
        if not provider:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{request.provider_name}' not found",
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
            kind=request.kind,
            visibility=request.visibility,
            min_subscription_tier=request.min_subscription_tier,
            fallback_model_slug=request.fallback_model_slug,
            supports_tools=request.supports_tools,
            supports_json_output=request.supports_json_output,
            supports_reasoning=request.supports_reasoning,
            supports_parallel_tool_calls=request.supports_parallel_tool_calls,
            capabilities=request.capabilities,
            metadata=request.metadata,
        )
        for cost_input in request.costs:
            await prisma.models.LlmModelCost.prisma().create(
                data={
                    "unit": cost_input.get("unit", "RUN"),
                    "creditCost": int(cost_input.get("credit_cost", 1)),
                    "credentialProvider": provider.name,
                    "metadata": prisma.Json(cost_input.get("metadata", {})),
                    "Model": {"connect": {"id": model.id}},
                }
            )

        await _audit(
            admin_user_id,
            "LlmModel",
            model.id,
            "LLM_MODEL_CREATED",
            None,
            _model_snapshot(model),
        )
        await llm_registry.refresh_runtime_caches()
        logger.info(f"Created model '{request.slug}' (id: {model.id})")

        created = await prisma.models.LlmModel.prisma().find_unique(
            where={"id": model.id},
            include={"Costs": True, "Creator": True},
        )
        if not created:
            raise HTTPException(status_code=500, detail="Model vanished after create")
        return _map_model(created)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail="Failed to create model")


@router.patch("/models/{slug:path}")
async def update_model(
    slug: str,
    request: UpdateLlmModelRequest,
    admin_user_id: str = Security(get_user_id),
) -> LlmModelAdminResponse:
    try:
        existing = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": slug}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Model with slug '{slug}' not found"
            )

        model = await db_write.update_model(
            model_id=existing.id,
            **request.model_dump(exclude_unset=True),
        )
        await _audit(
            admin_user_id,
            "LlmModel",
            model.id,
            "LLM_MODEL_UPDATED",
            _model_snapshot(existing),
            _model_snapshot(model),
        )
        await llm_registry.refresh_runtime_caches()
        logger.info(f"Updated model '{slug}' (id: {model.id})")
        return _map_model(model)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to update model: {e}")
        raise HTTPException(status_code=500, detail="Failed to update model")


@router.get("/models/{slug:path}/usage")
async def get_model_usage(slug: str) -> dict:
    try:
        return await db_write.get_model_usage(slug)
    except Exception as e:
        logger.exception(f"Failed to get model usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model usage")


@router.post("/models/{slug:path}/toggle")
async def toggle_model(
    slug: str,
    request: ToggleLlmModelRequest,
    admin_user_id: str = Security(get_user_id),
) -> dict:
    try:
        existing = await prisma.models.LlmModel.prisma().find_unique(
            where={"slug": slug}
        )
        if not existing:
            raise HTTPException(
                status_code=404, detail=f"Model with slug '{slug}' not found"
            )
        if not request.is_enabled and existing.isRecommended:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot disable the recommended model. "
                    "Change the recommended model before disabling this one."
                ),
            )

        result = await db_write.toggle_model_with_migration(
            model_id=existing.id,
            is_enabled=request.is_enabled,
            migrate_to_slug=request.migrate_to_slug,
            migration_reason=request.migration_reason,
            custom_credit_cost=request.custom_credit_cost,
        )
        await _audit(
            admin_user_id,
            "LlmModel",
            existing.id,
            "LLM_MODEL_ENABLED" if request.is_enabled else "LLM_MODEL_DISABLED",
            _model_snapshot(existing),
            {"isEnabled": request.is_enabled, **result},
        )
        await llm_registry.refresh_runtime_caches()
        logger.info(
            f"Toggled model '{slug}' enabled={request.is_enabled} "
            f"(migrated {result['nodes_migrated']} nodes)"
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to toggle model: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle model")


@router.delete("/models/{slug:path}")
async def delete_model(
    slug: str,
    replacement_model_slug: str | None = None,
    admin_user_id: str = Security(get_user_id),
) -> dict:
    try:
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
        await _audit(
            admin_user_id,
            "LlmModel",
            existing.id,
            "LLM_MODEL_DELETED",
            _model_snapshot(existing),
            result,
        )
        await llm_registry.refresh_runtime_caches()
        logger.info(
            f"Deleted model '{slug}' (migrated {result['nodes_migrated']} nodes)"
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete model")


@router.get("/models")
async def admin_list_models(
    page: int = 1,
    page_size: int = 100,
    enabled_only: bool = False,
) -> LlmModelsAdminListResponse:
    try:
        where = {"isEnabled": True} if enabled_only else {}
        models = await prisma.models.LlmModel.prisma().find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order={"displayName": "asc"},
            include={"Costs": True, "Creator": True},
        )
        return LlmModelsAdminListResponse(models=[_map_model(m) for m in models])
    except Exception as e:
        logger.exception(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


# --------------------------------------------------------------------------
# Migrations
# --------------------------------------------------------------------------


@router.get("/migrations")
async def list_migrations(
    include_reverted: bool = False,
) -> LlmMigrationsAdminListResponse:
    try:
        migrations = await db_write.list_migrations(include_reverted=include_reverted)
        return LlmMigrationsAdminListResponse(
            migrations=[LlmMigrationAdminResponse(**m) for m in migrations]
        )
    except Exception as e:
        logger.exception(f"Failed to list migrations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list migrations")


@router.post("/migrations/{migration_id}/revert")
async def revert_migration(
    migration_id: str,
    re_enable_source_model: bool = True,
    admin_user_id: str = Security(get_user_id),
) -> dict:
    try:
        result = await db_write.revert_migration(
            migration_id=migration_id,
            re_enable_source_model=re_enable_source_model,
        )
        await _audit(
            admin_user_id,
            "LlmModelMigration",
            migration_id,
            "LLM_MIGRATION_REVERTED",
            None,
            result,
        )
        await llm_registry.refresh_runtime_caches()
        logger.info(
            f"Reverted migration {migration_id}: "
            f"{result['nodes_reverted']} nodes restored"
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to revert migration: {e}")
        raise HTTPException(status_code=500, detail="Failed to revert migration")


# --------------------------------------------------------------------------
# Providers (read-only — the provider set is fixed; new providers are code)
# --------------------------------------------------------------------------


@router.get("/providers")
async def admin_list_providers() -> LlmProvidersAdminListResponse:
    try:
        providers = await prisma.models.LlmProvider.prisma().find_many(
            order={"name": "asc"},
            include={"Models": True},
        )
        return LlmProvidersAdminListResponse(
            providers=[
                _map_provider(p, model_count=len(p.Models) if p.Models else 0)
                for p in providers
            ]
        )
    except Exception as e:
        logger.exception(f"Failed to list providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to list providers")


# --------------------------------------------------------------------------
# Creators
# --------------------------------------------------------------------------


@router.get("/creators")
async def list_creators() -> LlmCreatorsAdminListResponse:
    try:
        creators = await prisma.models.LlmModelCreator.prisma().find_many(
            order={"name": "asc"}
        )
        return LlmCreatorsAdminListResponse(
            creators=[_map_creator(c) for c in creators]
        )
    except Exception as e:
        logger.exception(f"Failed to list creators: {e}")
        raise HTTPException(status_code=500, detail="Failed to list creators")


@router.post("/creators", status_code=status.HTTP_201_CREATED)
async def create_creator(
    request: CreateLlmCreatorRequest,
    admin_user_id: str = Security(get_user_id),
) -> LlmCreatorAdminResponse:
    try:
        creator = await prisma.models.LlmModelCreator.prisma().create(
            data={
                "name": request.name,
                "displayName": request.display_name,
                "description": request.description,
                "websiteUrl": request.website_url,
                "logoUrl": request.logo_url,
                "metadata": prisma.Json(request.metadata),
                "source": "LOCAL",
            }
        )
        await _audit(
            admin_user_id,
            "LlmModelCreator",
            creator.id,
            "LLM_CREATOR_CREATED",
            None,
            {"name": creator.name, "displayName": creator.displayName},
        )
        logger.info(f"Created creator '{creator.name}' (id: {creator.id})")
        return _map_creator(creator)
    except Exception as e:
        logger.exception(f"Failed to create creator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/creators/{name}")
async def update_creator(
    name: str,
    request: UpdateLlmCreatorRequest,
    admin_user_id: str = Security(get_user_id),
) -> LlmCreatorAdminResponse:
    try:
        existing = await prisma.models.LlmModelCreator.prisma().find_unique(
            where={"name": name}
        )
        if not existing:
            raise HTTPException(status_code=404, detail=f"Creator '{name}' not found")

        data: dict = {"source": "LOCAL"}
        if request.display_name is not None:
            data["displayName"] = request.display_name
        if request.description is not None:
            data["description"] = request.description
        if request.website_url is not None:
            data["websiteUrl"] = request.website_url
        if request.logo_url is not None:
            data["logoUrl"] = request.logo_url
        if request.metadata is not None:
            data["metadata"] = prisma.Json(request.metadata)

        creator = await prisma.models.LlmModelCreator.prisma().update(
            where={"id": existing.id},
            data=data,
        )
        if not creator:
            raise HTTPException(status_code=500, detail="Creator vanished mid-update")
        await _audit(
            admin_user_id,
            "LlmModelCreator",
            creator.id,
            "LLM_CREATOR_UPDATED",
            {"displayName": existing.displayName},
            {"displayName": creator.displayName},
        )
        logger.info(f"Updated creator '{name}' (id: {creator.id})")
        return _map_creator(creator)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update creator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/creators/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_creator(
    name: str,
    admin_user_id: str = Security(get_user_id),
) -> None:
    try:
        existing = await prisma.models.LlmModelCreator.prisma().find_unique(
            where={"name": name},
            include={"Models": True},
        )
        if not existing:
            raise HTTPException(status_code=404, detail=f"Creator '{name}' not found")
        if existing.Models and len(existing.Models) > 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot delete creator '{name}' — it has "
                    f"{len(existing.Models)} associated models"
                ),
            )

        await prisma.models.LlmModelCreator.prisma().delete(where={"id": existing.id})
        await _audit(
            admin_user_id,
            "LlmModelCreator",
            existing.id,
            "LLM_CREATOR_DELETED",
            {"name": existing.name},
            None,
        )
        logger.info(f"Deleted creator '{name}' (id: {existing.id})")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete creator: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Routing cells
# --------------------------------------------------------------------------


@router.get("/routes")
async def list_routes() -> LlmRoutesListResponse:
    try:
        routes = await db_routes.list_routes()
        return LlmRoutesListResponse(routes=[_map_route(r) for r in routes])
    except Exception as e:
        logger.exception(f"Failed to list routes: {e}")
        raise HTTPException(status_code=500, detail="Failed to list routes")


@router.put("/routes")
async def set_route(
    request: SetLlmRouteRequest,
    admin_user_id: str = Security(get_user_id),
) -> SetLlmRouteResponse:
    try:
        row, warnings = await db_routes.set_route(
            request.surface, request.mode, request.tier, request.model_slug
        )
        await _audit(
            admin_user_id,
            "LlmModelRoute",
            row.id if row else f"{request.surface}/{request.mode}/{request.tier}",
            "LLM_ROUTE_SET" if row else "LLM_ROUTE_CLEARED",
            None,
            {
                "surface": request.surface,
                "mode": request.mode,
                "tier": request.tier,
                "modelSlug": request.model_slug,
            },
        )
        await llm_registry.refresh_runtime_caches()
        logger.info(
            f"Routing cell ({request.surface}, {request.mode}, {request.tier}) "
            f"set to {request.model_slug!r}"
        )
        return SetLlmRouteResponse(
            route=_map_route(row) if row else None, warnings=warnings
        )
    except UnknownRouteModelError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to set route: {e}")
        raise HTTPException(status_code=500, detail="Failed to set route")


@router.get("/routes/warnings")
async def list_route_warnings() -> list[RouteWarning]:
    return await get_route_warnings()


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


@router.get("/catalog/export")
async def export_catalog() -> CatalogPayload:
    """Current registry as a catalog payload — pipe to catalog.json to
    refresh the bundled snapshot without direct DB access."""
    try:
        return await llm_registry.export_catalog()
    except Exception as e:
        logger.exception(f"Failed to export catalog: {e}")
        raise HTTPException(status_code=500, detail="Failed to export catalog")
