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
        if hasattr(builder_db, '_get_all_providers'):
            builder_db._get_all_providers.cache_clear()
            logger.info("Cleared v2 builder providers cache")
    except Exception as e:
        logger.debug("Could not clear v2 builder cache: %s", e)
    
    # Notify all executor services to refresh their registry cache
    from backend.data.llm_registry_notifications import publish_registry_refresh_notification
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
    response_model=llm_model.LlmModel,
)
async def toggle_llm_model(
    model_id: str,
    request: llm_model.ToggleLlmModelRequest,
):
    try:
        model = await llm_db.toggle_model(model_id=model_id, is_enabled=request.is_enabled)
        await _refresh_runtime_state()
        return model
    except Exception as exc:
        logger.exception("Failed to toggle LLM model %s: %s", model_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to toggle model availability",
        ) from exc

