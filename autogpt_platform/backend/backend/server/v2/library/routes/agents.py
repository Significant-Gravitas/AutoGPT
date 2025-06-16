import logging
from typing import Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

import backend.server.v2.library.db as library_db
import backend.server.v2.library.model as library_model
import backend.server.v2.store.exceptions as store_exceptions
from backend.data.graph import get_graph
from backend.integrations.webhooks.utils import setup_webhook as _setup_webhook

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["library", "private"],
    dependencies=[Depends(autogpt_auth_lib.auth_middleware)],
)


@router.get(
    "",
    responses={
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_library_agents(
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
    search_term: Optional[str] = Query(
        None, description="Search term to filter agents"
    ),
    sort_by: library_model.LibraryAgentSort = Query(
        library_model.LibraryAgentSort.UPDATED_AT,
        description="Criteria to sort results by",
    ),
    page: int = Query(
        1,
        ge=1,
        description="Page number to retrieve (must be >= 1)",
    ),
    page_size: int = Query(
        15,
        ge=1,
        description="Number of agents per page (must be >= 1)",
    ),
) -> library_model.LibraryAgentResponse:
    """
    Get all agents in the user's library (both created and saved).

    Args:
        user_id: ID of the authenticated user.
        search_term: Optional search term to filter agents by name/description.
        filter_by: List of filters to apply (favorites, created by user).
        sort_by: List of sorting criteria (created date, updated date).
        page: Page number to retrieve.
        page_size: Number of agents per page.

    Returns:
        A LibraryAgentResponse containing agents and pagination metadata.

    Raises:
        HTTPException: If a server/database error occurs.
    """
    try:
        return await library_db.list_library_agents(
            user_id=user_id,
            search_term=search_term,
            sort_by=sort_by,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        logger.exception("Listing library agents failed for user %s: %s", user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Inspect database connectivity."},
        ) from e


@router.get("/{library_agent_id}")
async def get_library_agent(
    library_agent_id: str,
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
) -> library_model.LibraryAgent:
    return await library_db.get_library_agent(id=library_agent_id, user_id=user_id)


@router.get(
    "/marketplace/{store_listing_version_id}",
    tags=["store, library"],
    response_model=library_model.LibraryAgent | None,
)
async def get_library_agent_by_store_listing_version_id(
    store_listing_version_id: str,
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
):
    """
    Get Library Agent from Store Listing Version ID.
    """
    try:
        return await library_db.get_library_agent_by_store_version_id(
            store_listing_version_id, user_id
        )
    except Exception as e:
        logger.exception(
            "Retrieving library agent by store version failed for user %s: %s",
            user_id,
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "hint": "Check if the store listing ID is valid.",
            },
        ) from e


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Agent added successfully"},
        404: {"description": "Store listing version not found"},
        500: {"description": "Server error"},
    },
)
async def add_marketplace_agent_to_library(
    store_listing_version_id: str = Body(embed=True),
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
) -> library_model.LibraryAgent:
    """
    Add an agent from the marketplace to the user's library.

    Args:
        store_listing_version_id: ID of the store listing version to add.
        user_id: ID of the authenticated user.

    Returns:
        library_model.LibraryAgent: Agent added to the library

    Raises:
        HTTPException(404): If the listing version is not found.
        HTTPException(500): If a server/database error occurs.
    """
    try:
        return await library_db.add_store_agent_to_library(
            store_listing_version_id=store_listing_version_id,
            user_id=user_id,
        )

    except store_exceptions.AgentNotFoundError:
        logger.warning(
            "Store listing version %s not found when adding to library",
            store_listing_version_id,
        )
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Store listing version {store_listing_version_id} not found",
                "hint": "Confirm the ID provided.",
            },
        )
    except store_exceptions.DatabaseError as e:
        logger.exception("Database error whilst adding agent to library: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Inspect DB logs for details."},
        ) from e
    except Exception as e:
        logger.exception("Unexpected error while adding agent to library: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "hint": "Check server logs for more information.",
            },
        ) from e


@router.put(
    "/{library_agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Agent updated successfully"},
        500: {"description": "Server error"},
    },
)
async def update_library_agent(
    library_agent_id: str,
    payload: library_model.LibraryAgentUpdateRequest,
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
) -> JSONResponse:
    """
    Update the library agent with the given fields.

    Args:
        library_agent_id: ID of the library agent to update.
        payload: Fields to update (auto_update_version, is_favorite, etc.).
        user_id: ID of the authenticated user.

    Returns:
        204 (No Content) on success.

    Raises:
        HTTPException(500): If a server/database error occurs.
    """
    try:
        await library_db.update_library_agent(
            library_agent_id=library_agent_id,
            user_id=user_id,
            auto_update_version=payload.auto_update_version,
            is_favorite=payload.is_favorite,
            is_archived=payload.is_archived,
            is_deleted=payload.is_deleted,
        )
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content={"message": "Agent updated successfully"},
        )
    except store_exceptions.DatabaseError as e:
        logger.exception("Database error while updating library agent: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Verify DB connection."},
        ) from e
    except Exception as e:
        logger.exception("Unexpected error while updating library agent: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Check server logs."},
        ) from e


@router.post("/{library_agent_id}/fork")
async def fork_library_agent(
    library_agent_id: str,
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
) -> library_model.LibraryAgent:
    return await library_db.fork_library_agent(
        library_agent_id=library_agent_id,
        user_id=user_id,
    )


class SetupWebhookRequest(BaseModel):
    provider: str
    credentials_id: str

    name: str
    description: str

    resource: str

    model_config = ConfigDict(extra="allow")


@router.post("/{library_agent_id}/setup_webhook_trigger")
async def setup_webhook(
    library_agent_id: str = Path(..., description="ID of the library agent"),
    params: SetupWebhookRequest = Body(),
    user_id: str = Depends(autogpt_auth_lib.depends.get_user_id),
) -> library_model.LibraryAgentPreset:
    """
    Sets up a webhook for a node and creates an AgentPreset with webhookId set.
    """
    library_agent = await library_db.get_library_agent(
        id=library_agent_id, user_id=user_id
    )
    if not library_agent:
        raise HTTPException(
            status_code=404,
            detail=f"Library agent #{library_agent_id} not found.",
        )

    graph = await get_graph(
        library_agent.graph_id, version=library_agent.graph_version, user_id=user_id
    )
    if not graph:
        raise HTTPException(
            status_code=404,
            detail=f"Graph #{library_agent.graph_id} not found.",
        )
    if not graph.webhook_input_node:
        raise HTTPException(
            status_code=400,
            detail=f"Graph #{library_agent.graph_id} does not have a webhook input node.",
        )

    trigger_webhook_config = graph.webhook_input_node.block.webhook_config
    if not trigger_webhook_config:
        raise HTTPException(
            status_code=400,
            detail=f"Graph #{library_agent.graph_id} does not have a trigger webhook configured.",
        )

    new_webhook = await _setup_webhook(
        user_id=user_id,
        provider=params.provider,
        webhook_type=trigger_webhook_config.webhook_type,
        credentials_id=params.credentials_id,
        resource=trigger_webhook_config.resource_format.format(**params.model_dump()),
        events=["triggered"],
        for_graph_id=library_agent.graph_id,
    )
    # Create preset with webhookId
    preset = await library_db.create_preset(
        user_id=user_id,
        preset=library_model.LibraryAgentPresetCreatable(
            graph_id=library_agent.graph_id,
            graph_version=library_agent.graph_version,
            name=params.name,
            description=params.description,
            inputs=params.inputs,
            is_active=True,
        ),
    )
    # Patch preset to set webhookId (Prisma: update after create)
    preset = await library_db.set_preset_webhook(preset.id, updated_node.webhook_id)
    return preset
