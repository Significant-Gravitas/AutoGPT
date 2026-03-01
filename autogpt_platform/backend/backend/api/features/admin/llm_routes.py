import logging

import autogpt_libs.auth
import fastapi

from backend.data import llm_registry
from backend.data.block_cost_config import refresh_llm_costs
from backend.server.v2.llm import db as llm_db
from backend.server.v2.llm import model as llm_model

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    tags=["llm", "admin"],
    dependencies=[fastapi.Security(autogpt_libs.auth.requires_admin_user)],
)


async def _refresh_runtime_state() -> None:
    """Refresh the LLM registry and clear all related caches to ensure real-time updates."""
    logger.info("Refreshing LLM registry runtime state...")
    try:
        # Refresh registry from database
        await llm_registry.refresh_llm_registry()
        await refresh_llm_costs()

        # Clear block schema caches so they're regenerated with updated model options
        from backend.blocks._base import BlockSchema

        BlockSchema.clear_all_schema_caches()
        logger.info("Cleared all block schema caches")

        # Clear the /blocks endpoint cache so frontend gets updated schemas
        try:
            from backend.api.features.v1 import _get_cached_blocks

            _get_cached_blocks.cache_clear()
            logger.info("Cleared /blocks endpoint cache")
        except Exception as e:
            logger.warning("Failed to clear /blocks cache: %s", e)

        # Clear the v2 builder caches
        try:
            from backend.api.features.builder import db as builder_db

            builder_db._get_all_providers.cache_clear()
            logger.info("Cleared v2 builder providers cache")
            builder_db._build_cached_search_results.cache_clear()
            logger.info("Cleared v2 builder search results cache")
            builder_db._get_llm_models.cache_clear()
            logger.info("Cleared v2 builder LLM models cache")
        except Exception as e:
            logger.debug("Could not clear v2 builder cache: %s", e)

        # Notify all executor services to refresh their registry cache
        from backend.data.llm_registry import publish_registry_refresh_notification

        await publish_registry_refresh_notification()
        logger.info("Published registry refresh notification")
    except Exception as exc:
        logger.exception(
            "LLM runtime state refresh failed; caches may be stale: %s", exc
        )


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


@router.delete(
    "/providers/{provider_id}",
    summary="Delete LLM provider",
    response_model=dict,
)
async def delete_llm_provider(provider_id: str):
    """
    Delete an LLM provider.

    A provider can only be deleted if it has no associated models.
    Delete all models from the provider first before deleting the provider.
    """
    try:
        await llm_db.delete_provider(provider_id)
        await _refresh_runtime_state()
        logger.info("Deleted LLM provider '%s'", provider_id)
        return {"success": True, "message": "Provider deleted successfully"}
    except ValueError as e:
        logger.warning("Failed to delete provider '%s': %s", provider_id, e)
        raise fastapi.HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to delete provider '%s': %s", provider_id, e)
        raise fastapi.HTTPException(status_code=500, detail=str(e))


@router.get(
    "/models",
    summary="List LLM models",
    response_model=llm_model.LlmModelsResponse,
)
async def list_llm_models(
    provider_id: str | None = fastapi.Query(default=None),
    page: int = fastapi.Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = fastapi.Query(
        default=50, ge=1, le=100, description="Number of models per page"
    ),
):
    return await llm_db.list_models(
        provider_id=provider_id, page=page, page_size=page_size
    )


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
    A migration record is created which can be reverted later using the revert endpoint.

    Optional fields:
    - `migration_reason`: Reason for the migration (e.g., "Provider outage")
    - `custom_credit_cost`: Custom pricing override for billing during migration
    """
    try:
        result = await llm_db.toggle_model(
            model_id=model_id,
            is_enabled=request.is_enabled,
            migrate_to_slug=request.migrate_to_slug,
            migration_reason=request.migration_reason,
            custom_credit_cost=request.custom_credit_cost,
        )
        await _refresh_runtime_state()
        if result.nodes_migrated > 0:
            logger.info(
                "Toggled model '%s' to %s and migrated %d nodes to '%s' (migration_id=%s)",
                result.model.slug,
                "enabled" if request.is_enabled else "disabled",
                result.nodes_migrated,
                result.migrated_to_slug,
                result.migration_id,
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
    replacement_model_slug: str | None = fastapi.Query(
        default=None,
        description="Slug of the model to migrate existing workflows to (required only if workflows use this model)",
    ),
):
    """
    Delete a model and optionally migrate workflows using it to a replacement model.

    If no workflows are using this model, it can be deleted without providing a
    replacement. If workflows exist, replacement_model_slug is required.

    This endpoint:
    1. Counts how many workflow nodes use the model being deleted
    2. If nodes exist, validates the replacement model and migrates them
    3. Deletes the model record
    4. Refreshes all caches and notifies executors

    Example: DELETE /api/llm/admin/models/{id}?replacement_model_slug=gpt-4o
    Example (no usage): DELETE /api/llm/admin/models/{id}
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


# ============================================================================
# Migration Management Endpoints
# ============================================================================


@router.get(
    "/migrations",
    summary="List model migrations",
    response_model=llm_model.LlmMigrationsResponse,
)
async def list_llm_migrations(
    include_reverted: bool = fastapi.Query(
        default=False, description="Include reverted migrations in the list"
    ),
):
    """
    List all model migrations.

    Migrations are created when disabling a model with the migrate_to_slug option.
    They can be reverted to restore the original model configuration.
    """
    try:
        migrations = await llm_db.list_migrations(include_reverted=include_reverted)
        return llm_model.LlmMigrationsResponse(migrations=migrations)
    except Exception as exc:
        logger.exception("Failed to list migrations: %s", exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to list migrations",
        ) from exc


@router.get(
    "/migrations/{migration_id}",
    summary="Get migration details",
    response_model=llm_model.LlmModelMigration,
)
async def get_llm_migration(migration_id: str):
    """Get details of a specific migration."""
    try:
        migration = await llm_db.get_migration(migration_id)
        if not migration:
            raise fastapi.HTTPException(
                status_code=404, detail=f"Migration '{migration_id}' not found"
            )
        return migration
    except fastapi.HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get migration %s: %s", migration_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to get migration",
        ) from exc


@router.post(
    "/migrations/{migration_id}/revert",
    summary="Revert a model migration",
    response_model=llm_model.RevertMigrationResponse,
)
async def revert_llm_migration(
    migration_id: str,
    request: llm_model.RevertMigrationRequest | None = None,
):
    """
    Revert a model migration, restoring affected workflows to their original model.

    This only reverts the specific nodes that were part of the migration.
    The source model must exist for the revert to succeed.

    Options:
    - `re_enable_source_model`: Whether to re-enable the source model if disabled (default: True)

    Response includes:
    - `nodes_reverted`: Number of nodes successfully reverted
    - `nodes_already_changed`: Number of nodes that were modified since migration (not reverted)
    - `source_model_re_enabled`: Whether the source model was re-enabled

    Requirements:
    - Migration must not already be reverted
    - Source model must exist
    """
    try:
        re_enable = request.re_enable_source_model if request else True
        result = await llm_db.revert_migration(
            migration_id,
            re_enable_source_model=re_enable,
        )
        await _refresh_runtime_state()
        logger.info(
            "Reverted migration '%s': %d nodes restored from '%s' to '%s' "
            "(%d already changed, source re-enabled=%s)",
            migration_id,
            result.nodes_reverted,
            result.target_model_slug,
            result.source_model_slug,
            result.nodes_already_changed,
            result.source_model_re_enabled,
        )
        return result
    except ValueError as exc:
        logger.warning("Migration revert validation failed: %s", exc)
        raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to revert migration %s: %s", migration_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to revert migration",
        ) from exc


# ============================================================================
# Creator Management Endpoints
# ============================================================================


@router.get(
    "/creators",
    summary="List model creators",
    response_model=llm_model.LlmCreatorsResponse,
)
async def list_llm_creators():
    """
    List all model creators.

    Creators are organizations that create/train models (e.g., OpenAI, Meta, Anthropic).
    This is distinct from providers who host/serve the models (e.g., OpenRouter).
    """
    try:
        creators = await llm_db.list_creators()
        return llm_model.LlmCreatorsResponse(creators=creators)
    except Exception as exc:
        logger.exception("Failed to list creators: %s", exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to list creators",
        ) from exc


@router.get(
    "/creators/{creator_id}",
    summary="Get creator details",
    response_model=llm_model.LlmModelCreator,
)
async def get_llm_creator(creator_id: str):
    """Get details of a specific model creator."""
    try:
        creator = await llm_db.get_creator(creator_id)
        if not creator:
            raise fastapi.HTTPException(
                status_code=404, detail=f"Creator '{creator_id}' not found"
            )
        return creator
    except fastapi.HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get creator %s: %s", creator_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to get creator",
        ) from exc


@router.post(
    "/creators",
    summary="Create model creator",
    response_model=llm_model.LlmModelCreator,
)
async def create_llm_creator(request: llm_model.UpsertLlmCreatorRequest):
    """
    Create a new model creator.

    A creator represents an organization that creates/trains AI models,
    such as OpenAI, Anthropic, Meta, or Google.
    """
    try:
        creator = await llm_db.upsert_creator(request=request)
        await _refresh_runtime_state()
        logger.info("Created model creator '%s' (%s)", creator.display_name, creator.id)
        return creator
    except Exception as exc:
        logger.exception("Failed to create creator: %s", exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to create creator",
        ) from exc


@router.patch(
    "/creators/{creator_id}",
    summary="Update model creator",
    response_model=llm_model.LlmModelCreator,
)
async def update_llm_creator(
    creator_id: str,
    request: llm_model.UpsertLlmCreatorRequest,
):
    """Update an existing model creator."""
    try:
        creator = await llm_db.upsert_creator(request=request, creator_id=creator_id)
        await _refresh_runtime_state()
        logger.info("Updated model creator '%s' (%s)", creator.display_name, creator_id)
        return creator
    except Exception as exc:
        logger.exception("Failed to update creator %s: %s", creator_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to update creator",
        ) from exc


@router.delete(
    "/creators/{creator_id}",
    summary="Delete model creator",
    response_model=dict,
)
async def delete_llm_creator(creator_id: str):
    """
    Delete a model creator.

    This will remove the creator association from all models that reference it
    (sets creatorId to NULL), but will not delete the models themselves.
    """
    try:
        await llm_db.delete_creator(creator_id)
        await _refresh_runtime_state()
        logger.info("Deleted model creator '%s'", creator_id)
        return {"success": True, "message": f"Creator '{creator_id}' deleted"}
    except ValueError as exc:
        logger.warning("Creator deletion validation failed: %s", exc)
        raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete creator %s: %s", creator_id, exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to delete creator",
        ) from exc


# ============================================================================
# Recommended Model Endpoints
# ============================================================================


@router.get(
    "/recommended-model",
    summary="Get recommended model",
    response_model=llm_model.RecommendedModelResponse,
)
async def get_recommended_model():
    """
    Get the currently recommended LLM model.

    The recommended model is shown to users as the default/suggested option
    in model selection dropdowns.
    """
    try:
        model = await llm_db.get_recommended_model()
        return llm_model.RecommendedModelResponse(
            model=model,
            slug=model.slug if model else None,
        )
    except Exception as exc:
        logger.exception("Failed to get recommended model: %s", exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to get recommended model",
        ) from exc


@router.post(
    "/recommended-model",
    summary="Set recommended model",
    response_model=llm_model.SetRecommendedModelResponse,
)
async def set_recommended_model(request: llm_model.SetRecommendedModelRequest):
    """
    Set a model as the recommended model.

    This clears the recommended flag from any other model and sets it on
    the specified model. The model must be enabled to be set as recommended.

    The recommended model is displayed to users as the default/suggested
    option in model selection dropdowns throughout the platform.
    """
    try:
        model, previous_slug = await llm_db.set_recommended_model(request.model_id)
        await _refresh_runtime_state()
        logger.info(
            "Set recommended model to '%s' (previous: %s)",
            model.slug,
            previous_slug or "none",
        )
        return llm_model.SetRecommendedModelResponse(
            model=model,
            previous_recommended_slug=previous_slug,
            message=f"Model '{model.display_name}' is now the recommended model",
        )
    except ValueError as exc:
        logger.warning("Set recommended model validation failed: %s", exc)
        raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to set recommended model: %s", exc)
        raise fastapi.HTTPException(
            status_code=500,
            detail="Failed to set recommended model",
        ) from exc
