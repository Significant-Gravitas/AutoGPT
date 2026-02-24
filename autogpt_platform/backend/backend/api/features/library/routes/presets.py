import logging
from typing import Any, Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Body, HTTPException, Query, Security, status

from backend.data.execution import GraphExecutionMeta
from backend.data.graph import get_graph
from backend.data.integrations import get_webhook
from backend.data.model import CredentialsMetaInput
from backend.executor.utils import add_graph_execution, make_node_credentials_input_map
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.webhooks import get_webhook_manager
from backend.integrations.webhooks.utils import setup_webhook_for_block
from backend.util.exceptions import NotFoundError

from .. import db
from .. import model as models

logger = logging.getLogger(__name__)

credentials_manager = IntegrationCredentialsManager()
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
    graph = await get_graph(
        params.graph_id, version=params.graph_version, user_id=user_id
    )
    if not graph:
        raise HTTPException(
            status.HTTP_410_GONE,
            f"Graph #{params.graph_id} not accessible (anymore)",
        )
    if not (trigger_node := graph.webhook_input_node):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Graph #{params.graph_id} does not have a webhook node",
        )

    trigger_config_with_credentials = {
        **params.trigger_config,
        **(
            make_node_credentials_input_map(graph, params.agent_credentials).get(
                trigger_node.id
            )
            or {}
        ),
    }

    new_webhook, feedback = await setup_webhook_for_block(
        user_id=user_id,
        trigger_block=trigger_node.block,
        trigger_config=trigger_config_with_credentials,
    )
    if not new_webhook:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not set up webhook: {feedback}",
        )

    new_preset = await db.create_preset(
        user_id=user_id,
        preset=models.LibraryAgentPresetCreatable(
            graph_id=params.graph_id,
            graph_version=params.graph_version,
            name=params.name,
            description=params.description,
            inputs=trigger_config_with_credentials,
            credentials=params.agent_credentials,
            webhook_id=new_webhook.id,
            is_active=True,
        ),
    )
    return new_preset


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
    current = await get_preset(preset_id, user_id=user_id)
    if not current:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Preset #{preset_id} not found")

    graph = await get_graph(
        current.graph_id,
        current.graph_version,
        user_id=user_id,
    )
    if not graph:
        raise HTTPException(
            status.HTTP_410_GONE,
            f"Graph #{current.graph_id} not accessible (anymore)",
        )

    trigger_inputs_updated, new_webhook, feedback = False, None, None
    if (trigger_node := graph.webhook_input_node) and (
        preset.inputs is not None and preset.credentials is not None
    ):
        trigger_config_with_credentials = {
            **preset.inputs,
            **(
                make_node_credentials_input_map(graph, preset.credentials).get(
                    trigger_node.id
                )
                or {}
            ),
        }
        new_webhook, feedback = await setup_webhook_for_block(
            user_id=user_id,
            trigger_block=graph.webhook_input_node.block,
            trigger_config=trigger_config_with_credentials,
            for_preset_id=preset_id,
        )
        trigger_inputs_updated = True
        if not new_webhook:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not update trigger configuration: {feedback}",
            )

    try:
        updated = await db.update_preset(
            user_id=user_id,
            preset_id=preset_id,
            inputs=preset.inputs,
            credentials=preset.credentials,
            name=preset.name,
            description=preset.description,
            is_active=preset.is_active,
        )
    except Exception as e:
        logger.exception("Preset update failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    # Update the webhook as well, if necessary
    if trigger_inputs_updated:
        updated = await db.set_preset_webhook(
            user_id, preset_id, new_webhook.id if new_webhook else None
        )

        # Clean up webhook if it is now unused
        if current.webhook_id and (
            current.webhook_id != (new_webhook.id if new_webhook else None)
        ):
            current_webhook = await get_webhook(current.webhook_id)
            credentials = (
                await credentials_manager.get(user_id, current_webhook.credentials_id)
                if current_webhook.credentials_id
                else None
            )
            await get_webhook_manager(
                current_webhook.provider
            ).prune_webhook_if_dangling(user_id, current_webhook.id, credentials)

    return updated


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
    preset = await db.get_preset(user_id, preset_id)
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset #{preset_id} not found for user #{user_id}",
        )

    # Detach and clean up the attached webhook, if any
    if preset.webhook_id:
        webhook = await get_webhook(preset.webhook_id)
        await db.set_preset_webhook(user_id, preset_id, None)

        # Clean up webhook if it is now unused
        credentials = (
            await credentials_manager.get(user_id, webhook.credentials_id)
            if webhook.credentials_id
            else None
        )
        await get_webhook_manager(webhook.provider).prune_webhook_if_dangling(
            user_id, webhook.id, credentials
        )

    try:
        await db.delete_preset(user_id, preset_id)
    except Exception as e:
        logger.exception(
            "Error deleting preset %s for user %s: %s", preset_id, user_id, e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/presets/{preset_id}/execute",
    tags=["presets"],
    summary="Execute a preset",
    description="Execute a preset with the given graph and node input for the current user.",
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
