import logging
from typing import Annotated, Any, Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

import backend.server.v2.library.db as db
import backend.server.v2.library.model as models
from backend.executor.utils import add_graph_execution
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/presets",
    summary="List presets",
    description="Retrieve a paginated list of presets for the current user.",
)
async def list_presets(
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "hint": "Ensure the presets DB table is accessible.",
            },
        )


@router.get(
    "/presets/{preset_id}",
    summary="Get a specific preset",
    description="Retrieve details for a specific preset by its ID.",
)
async def get_preset(
    preset_id: str,
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
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
        if not preset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset {preset_id} not found",
            )
        return preset
    except Exception as e:
        logger.exception(
            "Error retrieving preset %s for user %s: %s", preset_id, user_id, e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Validate preset ID and retry."},
        )


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
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Check preset payload format."},
        )


@router.patch(
    "/presets/{preset_id}",
    summary="Update an existing preset",
    description="Update an existing preset by its ID.",
)
async def update_preset(
    preset_id: str,
    preset: models.LibraryAgentPresetUpdatable,
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
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
    try:
        return await db.update_preset(
            user_id=user_id, preset_id=preset_id, preset=preset
        )
    except Exception as e:
        logger.exception("Preset update failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Check preset data and try again."},
        )


@router.delete(
    "/presets/{preset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a preset",
    description="Delete an existing preset by its ID.",
)
async def delete_preset(
    preset_id: str,
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
) -> None:
    """
    Delete a preset by its ID. Returns 204 No Content on success.

    Args:
        preset_id (str): ID of the preset to delete.
        user_id (str): ID of the authenticated user.

    Raises:
        HTTPException: If an error occurs while deleting the preset.
    """
    try:
        await db.delete_preset(user_id, preset_id)
    except Exception as e:
        logger.exception(
            "Error deleting preset %s for user %s: %s", preset_id, user_id, e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Ensure preset exists before deleting."},
        )


@router.post(
    "/presets/{preset_id}/execute",
    tags=["presets"],
    summary="Execute a preset",
    description="Execute a preset with the given graph and node input for the current user.",
)
async def execute_preset(
    graph_id: str,
    graph_version: int,
    preset_id: str,
    node_input: Annotated[dict[str, Any], Body(..., embed=True, default_factory=dict)],
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
) -> dict[str, Any]:  # FIXME: add proper return type
    """
    Execute a preset given graph parameters, returning the execution ID on success.

    Args:
        graph_id (str): ID of the graph to execute.
        graph_version (int): Version of the graph to execute.
        preset_id (str): ID of the preset to execute.
        node_input (Dict[Any, Any]): Input data for the node.
        user_id (str): ID of the authenticated user.

    Returns:
        Dict[str, Any]: A response containing the execution ID.

    Raises:
        HTTPException: If the preset is not found or an error occurs while executing the preset.
    """
    try:
        preset = await db.get_preset(user_id, preset_id)
        if not preset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Preset not found",
            )

        # Merge input overrides with preset inputs
        merged_node_input = preset.inputs | node_input

        execution = await add_graph_execution(
            graph_id=graph_id,
            user_id=user_id,
            inputs=merged_node_input,
            preset_id=preset_id,
            graph_version=graph_version,
        )

        logger.debug(f"Execution added: {execution} with input: {merged_node_input}")

        return {"id": execution.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Preset execution failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": str(e),
                "hint": "Review preset configuration and graph ID.",
            },
        )
