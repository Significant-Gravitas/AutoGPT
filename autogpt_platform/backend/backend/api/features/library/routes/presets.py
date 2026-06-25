import logging
from typing import Any, Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Security, status

from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.execution import GraphExecutionMeta
from backend.data.model import CredentialsMetaInput
from backend.executor.utils import add_graph_execution
from backend.util.exceptions import NotFoundError, WebhookRegistrationError

from .. import db
from .. import model as models
from ..triggers import (
    delete_preset_with_webhook_cleanup,
    setup_triggered_preset,
    update_triggered_preset,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["presets"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "/presets",
    summary="List presets",
    description="Retrieve a paginated list of presets for the current user.",
)
async def list_presets(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1),
    graph_id: Optional[str] = Query(
        description="Allows to filter presets by a specific agent graph"
    ),
) -> models.LibraryAgentPresetResponse:
    """
    Retrieve a paginated list of presets for the current user.

    Args:
        user_id (str): ID of the authenticated user.
        page (int): Page number for pagination.
        page_size (int): Number of items per page.
        graph_id: Allows to filter presets by a specific agent graph.

    Returns:
        models.LibraryAgentPresetResponse: A response containing the list of presets.
    """
    try:
        return await db.list_presets(
            user_id=user_id,
            graph_id=graph_id,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        logger.exception("Failed to list presets for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get(
    "/presets/{preset_id}",
    summary="Get a specific preset",
    description="Retrieve details for a specific preset by its ID.",
)
async def get_preset(
    preset_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> models.LibraryAgentPreset:
    """
    Retrieve details for a specific preset by its ID.

    Args:
        preset_id (str): ID of the preset to retrieve.
        user_id (str): ID of the authenticated user.

    Returns:
        models.LibraryAgentPreset: The preset details.

    Raises:
        HTTPException: If the preset is not found or an error occurs.
    """
    try:
        preset = await db.get_preset(user_id, preset_id)
    except Exception as e:
        logger.exception(
            "Error retrieving preset %s for user %s: %s", preset_id, user_id, e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset #{preset_id} not found",
        )
    return preset


@router.post(
    "/presets",
    summary="Create a new preset",
    description="Create a new preset for the current user.",
)
async def create_preset(
    preset: (
        models.LibraryAgentPresetCreatable
        | models.LibraryAgentPresetCreatableFromGraphExecution
    ),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> models.LibraryAgentPreset:
    """
    Create a new library agent preset. Automatically corrects node_input format if needed.

    Args:
        preset (models.LibraryAgentPresetCreatable): The preset data to create.
        user_id (str): ID of the authenticated user.

    Returns:
        models.LibraryAgentPreset: The created preset.

    Raises:
        HTTPException: If an error occurs while creating the preset.
    """
    try:
        if isinstance(preset, models.LibraryAgentPresetCreatable):
            return await db.create_preset(user_id, preset)
        else:
            return await db.create_preset_from_graph_execution(user_id, preset)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("Preset creation failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/presets/setup-trigger")
async def setup_trigger(
    params: models.TriggeredPresetSetupRequest = Body(),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> models.LibraryAgentPreset:
    """
    Sets up a webhook-triggered `LibraryAgentPreset` for a `LibraryAgent`.
    Returns the correspondingly created `LibraryAgentPreset` with `webhook_id` set.
    """
    try:
        return await setup_triggered_preset(
            user_id=user_id,
            graph_id=params.graph_id,
            graph_version=params.graph_version,
            name=params.name,
            description=params.description,
            trigger_config=params.trigger_config,
            agent_credentials=params.agent_credentials,
        )
    except WebhookRegistrationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not set up the trigger: {e}",
        )


@router.patch(
    "/presets/{preset_id}",
    summary="Update an existing preset",
    description="Update an existing preset by its ID.",
)
async def update_preset(
    preset_id: str,
    preset: models.LibraryAgentPresetUpdatable,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> models.LibraryAgentPreset:
    """
    Update an existing library agent preset.

    Args:
        preset_id (str): ID of the preset to update.
        preset (models.LibraryAgentPresetUpdatable): The preset data to update.
        user_id (str): ID of the authenticated user.

    Returns:
        models.LibraryAgentPreset: The updated preset.

    Raises:
        HTTPException: If an error occurs while updating the preset.
    """
    # Webhook re-registration + old-webhook pruning lives in the shared
    # update_triggered_preset (see triggers.py) so the copilot update_preset tool
    # reuses the exact same logic. NotFoundError/InvalidInputError are mapped to
    # 404/400 by the global exception handlers.
    return await update_triggered_preset(
        user_id=user_id,
        preset_id=preset_id,
        inputs=preset.inputs,
        credentials=preset.credentials,
        name=preset.name,
        description=preset.description,
        is_active=preset.is_active,
    )


@router.delete(
    "/presets/{preset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a preset",
    description="Delete an existing preset by its ID.",
)
async def delete_preset(
    preset_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> None:
    """
    Delete a preset by its ID. Returns 204 No Content on success.

    Args:
        preset_id (str): ID of the preset to delete.
        user_id (str): ID of the authenticated user.

    Raises:
        HTTPException: If an error occurs while deleting the preset.
    """
    # Webhook detach + prune-if-dangling lives in the shared
    # delete_preset_with_webhook_cleanup (see triggers.py) so the copilot
    # delete_preset tool reuses it. NotFoundError → 404 via the global handler.
    await delete_preset_with_webhook_cleanup(user_id=user_id, preset_id=preset_id)


@router.post(
    "/presets/{preset_id}/execute",
    tags=["presets"],
    summary="Execute a preset",
    description="Execute a preset with the given graph and node input for the current user.",
    dependencies=[Depends(enforce_payment_paywall)],
    responses={
        402: {"description": "Subscription required (NO_TIER user, paywall on)"},
        503: {"description": "Subscription state temporarily unavailable"},
    },
)
async def execute_preset(
    preset_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    inputs: dict[str, Any] = Body(..., embed=True, default_factory=dict),
    credential_inputs: dict[str, CredentialsMetaInput] = Body(
        ..., embed=True, default_factory=dict
    ),
) -> GraphExecutionMeta:
    """
    Execute a preset given graph parameters, returning the execution ID on success.

    Args:
        preset_id: ID of the preset to execute.
        user_id: ID of the authenticated user.
        inputs: Optionally, inputs to override the preset for execution.
        credential_inputs: Optionally, credentials to override the preset for execution.

    Returns:
        GraphExecutionMeta: Object representing the created execution.

    Raises:
        HTTPException: If the preset is not found or an error occurs while executing the preset.
    """
    preset = await db.get_preset(user_id, preset_id)
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset #{preset_id} not found",
        )

    # Merge input overrides with preset inputs
    merged_node_input = preset.inputs | inputs
    merged_credential_inputs = preset.credentials | credential_inputs

    return await add_graph_execution(
        user_id=user_id,
        graph_id=preset.graph_id,
        graph_version=preset.graph_version,
        preset_id=preset_id,
        inputs=merged_node_input,
        graph_credentials_inputs=merged_credential_inputs,
    )
