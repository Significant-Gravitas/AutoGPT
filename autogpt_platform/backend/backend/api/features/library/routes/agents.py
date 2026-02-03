from typing import Literal, Optional

import autogpt_libs.auth as autogpt_auth_lib
from fastapi import APIRouter, Body, HTTPException, Query, Security, status
from fastapi.responses import Response
from prisma.enums import OnboardingStep

from backend.data.onboarding import complete_onboarding_step

from .. import db as library_db
from .. import model as library_model

router = APIRouter(
    prefix="/agents",
    tags=["library", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get(
    "",
    summary="List Library Agents",
    response_model=library_model.LibraryAgentResponse,
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
    folder_id: Optional[str] = Query(
        None,
        description="Filter by folder ID",
    ),
    include_root_only: bool = Query(
        False,
        description="Only return agents without a folder (root-level agents)",
    ),
) -> library_model.LibraryAgentResponse:
    """
    Get all agents in the user's library (both created and saved).
    """
    return await library_db.list_library_agents(
        user_id=user_id,
        search_term=search_term,
        sort_by=sort_by,
        page=page,
        page_size=page_size,
        folder_id=folder_id,
        include_root_only=include_root_only,
    )


@router.get(
    "/favorites",
    summary="List Favorite Library Agents",
)
async def list_favorite_library_agents(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
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
    Get all favorite agents in the user's library.
    """
    return await library_db.list_favorite_library_agents(
        user_id=user_id,
        page=page,
        page_size=page_size,
    )


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
    tags=["store", "library"],
    response_model=library_model.LibraryAgent | None,
)
async def get_library_agent_by_store_listing_version_id(
    store_listing_version_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent | None:
    """
    Get Library Agent from Store Listing Version ID.
    """
    return await library_db.get_library_agent_by_store_version_id(
        store_listing_version_id, user_id
    )


@router.post(
    "",
    summary="Add Marketplace Agent",
    status_code=status.HTTP_201_CREATED,
)
async def add_marketplace_agent_to_library(
    store_listing_version_id: str = Body(embed=True),
    source: Literal["onboarding", "marketplace"] = Body(
        default="marketplace", embed=True
    ),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    """
    Add an agent from the marketplace to the user's library.
    """
    agent = await library_db.add_store_agent_to_library(
        store_listing_version_id=store_listing_version_id,
        user_id=user_id,
    )
    if source != "onboarding":
        await complete_onboarding_step(user_id, OnboardingStep.MARKETPLACE_ADD_AGENT)
    return agent


@router.patch(
    "/{library_agent_id}",
    summary="Update Library Agent",
)
async def update_library_agent(
    library_agent_id: str,
    payload: library_model.LibraryAgentUpdateRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    """
    Update the library agent with the given fields.
    """
    return await library_db.update_library_agent(
        library_agent_id=library_agent_id,
        user_id=user_id,
        auto_update_version=payload.auto_update_version,
        graph_version=payload.graph_version,
        is_favorite=payload.is_favorite,
        is_archived=payload.is_archived,
        settings=payload.settings,
        folder_id=payload.folder_id,
    )


@router.delete(
    "/{library_agent_id}",
    summary="Delete Library Agent",
)
async def delete_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Response:
    """
    Soft-delete the specified library agent.
    """
    await library_db.delete_library_agent(
        library_agent_id=library_agent_id, user_id=user_id
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{library_agent_id}/fork", summary="Fork Library Agent")
async def fork_library_agent(
    library_agent_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> library_model.LibraryAgent:
    return await library_db.fork_library_agent(
        library_agent_id=library_agent_id,
        user_id=user_id,
    )
