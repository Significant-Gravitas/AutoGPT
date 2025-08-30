import logging
from typing import Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Body, HTTPException, Query, Security, status
from fastapi.responses import Response

import backend.server.v2.library.db as library_db
import backend.server.v2.library.model as library_model
import backend.server.v2.store.exceptions as store_exceptions
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["library", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "",
    summary="List Library Agents",
    responses={
        500: {"description": "Server error", "content": {"application/json": {}}},
    },
)
async def list_library_agents(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
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
        logger.error(f"Could not list library agents for user #{user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get("/{library_agent_id}", summary="Get Library Agent")
async def get_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    return await library_db.get_library_agent(id=library_agent_id, user_id=user_id)


@router.get("/by-graph/{graph_id}")
async def get_library_agent_by_graph_id(
    graph_id: str,
    version: Optional[int] = Query(default=None),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    library_agent = await library_db.get_library_agent_by_graph_id(
        user_id, graph_id, version
    )
    if not library_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library agent for graph #{graph_id} and user #{user_id} not found",
        )
    return library_agent


@router.get(
    "/marketplace/{store_listing_version_id}",
    summary="Get Agent By Store ID",
    tags=["store, library"],
)
async def get_library_agent_by_store_listing_version_id(
    store_listing_version_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent | None:
    """
    Get Library Agent from Store Listing Version ID.
    """
    try:
        return await library_db.get_library_agent_by_store_version_id(
            store_listing_version_id, user_id
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Could not fetch library agent from store version ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post(
    "",
    summary="Add Marketplace Agent",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Agent added successfully"},
        404: {"description": "Store listing version not found"},
        500: {"description": "Server error"},
    },
)
async def add_marketplace_agent_to_library(
    store_listing_version_id: str = Body(embed=True),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
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

    except store_exceptions.AgentNotFoundError as e:
        logger.warning(
            f"Could not find store listing version {store_listing_version_id} "
            "to add to library"
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except store_exceptions.DatabaseError as e:
        logger.error(f"Database error while adding agent to library: {e}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Inspect DB logs for details."},
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error while adding agent to library: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": str(e),
                "hint": "Check server logs for more information.",
            },
        ) from e


@router.patch(
    "/{library_agent_id}",
    summary="Update Library Agent",
    responses={
        200: {"description": "Agent updated successfully"},
        500: {"description": "Server error"},
    },
)
async def update_library_agent(
    library_agent_id: str,
    payload: library_model.LibraryAgentUpdateRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    """
    Update the library agent with the given fields.

    Args:
        library_agent_id: ID of the library agent to update.
        payload: Fields to update (auto_update_version, is_favorite, etc.).
        user_id: ID of the authenticated user.

    Raises:
        HTTPException(500): If a server/database error occurs.
    """
    try:
        return await library_db.update_library_agent(
            library_agent_id=library_agent_id,
            user_id=user_id,
            auto_update_version=payload.auto_update_version,
            is_favorite=payload.is_favorite,
            is_archived=payload.is_archived,
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except store_exceptions.DatabaseError as e:
        logger.error(f"Database error while updating library agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Verify DB connection."},
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error while updating library agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "hint": "Check server logs."},
        ) from e


@router.delete(
    "/{library_agent_id}",
    summary="Delete Library Agent",
    responses={
        204: {"description": "Agent deleted successfully"},
        404: {"description": "Agent not found"},
        500: {"description": "Server error"},
    },
)
async def delete_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    """
    Soft-delete the specified library agent.

    Args:
        library_agent_id: ID of the library agent to delete.
        user_id: ID of the authenticated user.

    Returns:
        204 No Content if successful.

    Raises:
        HTTPException(404): If the agent does not exist.
        HTTPException(500): If a server/database error occurs.
    """
    try:
        await library_db.delete_library_agent(
            library_agent_id=library_agent_id, user_id=user_id
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


@router.post("/{library_agent_id}/fork", summary="Fork Library Agent")
async def fork_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    return await library_db.fork_library_agent(
        library_agent_id=library_agent_id,
        user_id=user_id,
    )
