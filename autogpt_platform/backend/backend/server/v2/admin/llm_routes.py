import logging

import autogpt_libs.auth
import fastapi

from backend.data import llm_registry
from backend.data.block_cost_config import refresh_llm_costs
from backend.server.v2.llm import db as llm_db
from backend.server.v2.llm import model as llm_model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    prefix="/admin/llm",
    tags=["llm", "admin"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_admin_user)],
)


async def _refresh_runtime_state() -> None:
    """Refresh the LLM registry and clear all related caches to ensure real-time updates."""
    logger.info("Refreshing LLM registry runtime state...")

    # Refresh registry from database
    await llm_registry.refresh_llm_registry()
    refresh_llm_costs()

    # Clear block schema caches so they're regenerated with updated model options
    from backend.data.block import BlockSchema

    BlockSchema.clear_all_schema_caches()
    logger.info("Cleared all block schema caches")

    # Clear the /blocks endpoint cache so frontend gets updated schemas
    try:
        from backend.server.routers.v1 import _get_cached_blocks

        _get_cached_blocks.cache_clear()
        logger.info("Cleared /blocks endpoint cache")
    except Exception as e:
        logger.warning("Failed to clear /blocks cache: %s", e)

    # Clear the v2 builder providers cache (if it exists)
    try:
        from backend.server.v2.builder import db as builder_db

        if hasattr(builder_db, "_get_all_providers"):
            builder_db._get_all_providers.cache_clear()
            logger.info("Cleared v2 builder providers cache")
    except Exception as e:
        logger.debug("Could not clear v2 builder cache: %s", e)

    # Notify all executor services to refresh their registry cache
    from backend.data.llm_registry import (
        publish_registry_refresh_notification,
    )

    publish_registry_refresh_notification()
    logger.info("Published registry refresh notification")


@router.get(
    "/providers",
    summary="List LLM providers",
    response_model=llm_model.LlmProvidersResponse,
)
async def list_llm_providers(include_models: bool = True):
    providers = await llm_db.list_providers(include_models=include_models)
    return llm_model.LlmProvidersResponse(providers=providers)


@router.post(
    "/providers",
    summary="Create LLM provider",
    response_model=llm_model.LlmProvider,
)
async def create_llm_provider(request: llm_model.UpsertLlmProviderRequest):
    provider = await llm_db.upsert_provider(request=request)
    await _refresh_runtime_state()
    return provider


@router.patch(
    "/providers/{provider_id}",
    summary="Update LLM provider",
    response_model=llm_model.LlmProvider,
)
async def update_llm_provider(
    provider_id: str,
    request: llm_model.UpsertLlmProviderRequest,
):
    provider = await llm_db.upsert_provider(request=request, provider_id=provider_id)
    await _refresh_runtime_state()
    return provider


@router.get(
    "/models",
    summary="List LLM models",
    response_model=llm_model.LlmModelsResponse,
)
async def list_llm_models(provider_id: str | None = fastapi.Query(default=None)):
    models = await llm_db.list_models(provider_id=provider_id)
    return llm_model.LlmModelsResponse(models=models)


@router.post(
    "/models",
    summary="Create LLM model",
    response_model=llm_model.LlmModel,
)
async def create_llm_model(request: llm_model.CreateLlmModelRequest):
    model = await llm_db.create_model(request=request)
    await _refresh_runtime_state()
    return model


@router.patch(
    "/models/{model_id}",
    summary="Update LLM model",
    response_model=llm_model.LlmModel,
)
async def update_llm_model(
    model_id: str,
    request: llm_model.UpdateLlmModelRequest,
):
    model = await llm_db.update_model(model_id=model_id, request=request)
    await _refresh_runtime_state()
    return model


@router.patch(
    "/models/{model_id}/toggle",
    summary="Toggle LLM model availability",
    response_model=llm_model.ToggleLlmModelResponse,
)
async def toggle_llm_model(
    model_id: str,
    request: llm_model.ToggleLlmModelRequest,
):
    """
    Toggle a model's enabled status, optionally migrating workflows when disabling.

    If disabling a model and `migrate_to_slug` is provided, all workflows using
    this model will be migrated to the specified replacement model before disabling.
    """
    try:
        result = await llm_db.toggle_model(
            model_id=model_id,
            is_enabled=request.is_enabled,
            migrate_to_slug=request.migrate_to_slug,
        )
        await _refresh_runtime_state()
        if result.nodes_migrated > 0:
            logger.info(
                "Toggled model '%s' to %s and migrated %d nodes to '%s'",
                result.model.slug,
                "enabled" if request.is_enabled else "disabled",
                result.nodes_migrated,
                result.migrated_to_slug,
            )
        return result
    except ValueError as exc:
        logger.warning("Model toggle validation failed: %s", exc)
        raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to toggle LLM model %s: %s", model_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to toggle model availability",
        ) from exc


@router.get(
    "/models/{model_id}/usage",
    summary="Get model usage count",
    response_model=llm_model.LlmModelUsageResponse,
)
async def get_llm_model_usage(model_id: str):
    """Get the number of workflow nodes using this model."""
    try:
        return await llm_db.get_model_usage(model_id=model_id)
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to get model usage %s: %s", model_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to get model usage",
        ) from exc


@router.delete(
    "/models/{model_id}",
    summary="Delete LLM model and migrate workflows",
    response_model=llm_model.DeleteLlmModelResponse,
)
async def delete_llm_model(
    model_id: str,
    replacement_model_slug: str = fastapi.Query(
        ..., description="Slug of the model to migrate existing workflows to"
    ),
):
    """
    Delete a model and automatically migrate all workflows using it to a replacement model.

    This endpoint:
    1. Validates the replacement model exists and is enabled
    2. Counts how many workflow nodes use the model being deleted
    3. Updates all AgentNode.constantInput->model fields to the replacement
    4. Deletes the model record
    5. Refreshes all caches and notifies executors

    Example: DELETE /admin/llm/models/{id}?replacement_model_slug=gpt-4o
    """
    try:
        result = await llm_db.delete_model(
            model_id=model_id, replacement_model_slug=replacement_model_slug
        )
        await _refresh_runtime_state()
        logger.info(
            "Deleted model '%s' and migrated %d nodes to '%s'",
            result.deleted_model_slug,
            result.nodes_migrated,
            result.replacement_model_slug,
        )
        return result
    except ValueError as exc:
        # Validation errors (model not found, replacement invalid, etc.)
        logger.warning("Model deletion validation failed: %s", exc)
        raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete LLM model %s: %s", model_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to delete model and migrate workflows",
        ) from exc
