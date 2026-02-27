"""
V2 External API - Library Agent Endpoints

Provides access to the user's agent library and agent execution.
"""

import logging

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.data import execution as execution_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.credit import get_user_credit_model
from backend.executor import utils as execution_utils

from ..common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from ..models import (
    AgentGraphRun,
    AgentRunListResponse,
    AgentRunRequest,
    LibraryAgent,
    LibraryAgentListResponse,
    LibraryAgentUpdateRequest,
)
from ..rate_limit import execute_limiter

logger = logging.getLogger(__name__)

agents_router = APIRouter()


# ============================================================================
# Endpoints
# ============================================================================


@agents_router.get(
    path="/agents",
    summary="List library agents",
)
async def list_library_agents(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
    published: bool | None = Query(
        default=None,
        description="Filter by marketplace publish status: "
        "true = published, false = unpublished, omit = all",
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> LibraryAgentListResponse:
    """
    List agents in the user's library.

    The library contains agents the user has created or added from the marketplace.
    Use the `published` filter to show only agents that are/aren't listed on the
    marketplace.
    """
    result = await library_db.list_library_agents(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
        published=published,
    )

    return LibraryAgentListResponse(
        agents=[LibraryAgent.from_internal(a) for a in result.agents],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )


@agents_router.get(
    path="/agents/favorites",
    summary="List favorite agents",
)
async def list_favorite_agents(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> LibraryAgentListResponse:
    """
    List favorite agents in the user's library.
    """
    result = await library_db.list_favorite_library_agents(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
    )

    return LibraryAgentListResponse(
        agents=[LibraryAgent.from_internal(a) for a in result.agents],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )


@agents_router.get(
    path="/agents/{agent_id}",
    summary="Get library agent details",
)
async def get_library_agent(
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryAgent:
    """
    Get detailed information about a specific agent in the user's library.
    """
    try:
        agent = await library_db.get_library_agent(
            id=agent_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found")

    return LibraryAgent.from_internal(agent)


@agents_router.patch(
    path="/agents/{agent_id}",
    summary="Update a library agent",
)
async def update_library_agent(
    request: LibraryAgentUpdateRequest,
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryAgent:
    """
    Update properties of a library agent.

    Only the fields provided in the request body will be updated.
    """
    try:
        updated = await library_db.update_library_agent(
            library_agent_id=agent_id,
            user_id=auth.user_id,
            auto_update_version=request.auto_update_version,
            graph_version=request.graph_version,
            is_favorite=request.is_favorite,
            is_archived=request.is_archived,
            folder_id=request.folder_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found")

    return LibraryAgent.from_internal(updated)


@agents_router.delete(
    path="/agents/{agent_id}",
    summary="Delete a library agent",
    status_code=HTTP_204_NO_CONTENT,
)
async def delete_library_agent(
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> None:
    """
    Remove an agent from the user's library.
    """
    try:
        await library_db.delete_library_agent(
            library_agent_id=agent_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found")


@agents_router.post(
    path="/agents/{agent_id}/fork",
    summary="Fork a library agent",
    status_code=HTTP_201_CREATED,
)
async def fork_library_agent(
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryAgent:
    """
    Fork (clone) a library agent.

    Creates a copy of the agent's graph with new IDs, owned by the
    authenticated user.
    """
    try:
        forked = await library_db.fork_library_agent(
            library_agent_id=agent_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found")

    return LibraryAgent.from_internal(forked)


@agents_router.post(
    path="/agents/{agent_id}/runs",
    summary="Execute an agent",
)
async def execute_agent(
    request: AgentRunRequest,
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.RUN_AGENT)
    ),
) -> AgentGraphRun:
    """
    Execute an agent from the library.

    This creates a new run with the provided inputs. The run executes
    asynchronously and you can poll the run status using GET /runs/{run_id}.
    """
    execute_limiter.check(auth.user_id)

    # Check credit balance
    user_credit_model = await get_user_credit_model(auth.user_id)
    current_balance = await user_credit_model.get_credits(auth.user_id)
    if current_balance <= 0:
        raise HTTPException(
            status_code=402,
            detail="Insufficient balance to execute the agent. Please top up your account.",
        )

    # Get the library agent to find the graph ID and version
    try:
        library_agent = await library_db.get_library_agent(
            id=agent_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found")

    try:
        result = await execution_utils.add_graph_execution(
            graph_id=library_agent.graph_id,
            user_id=auth.user_id,
            inputs=request.inputs,
            graph_version=library_agent.graph_version,
            graph_credentials_inputs=request.credentials_inputs,
        )

        return AgentGraphRun.from_internal(result)

    except Exception as e:
        logger.error(f"Failed to execute agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@agents_router.get(
    path="/agents/{agent_id}/runs",
    summary="List runs for an agent",
)
async def list_agent_runs(
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
) -> AgentRunListResponse:
    """
    List execution runs for a specific agent.
    """
    # Get the library agent to find the graph ID
    try:
        library_agent = await library_db.get_library_agent(
            id=agent_id,
            user_id=auth.user_id,
        )
    except Exception:
        raise HTTPException(status_code=404, detail=f"Agent #{agent_id} not found")

    result = await execution_db.get_graph_executions_paginated(
        graph_id=library_agent.graph_id,
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
    )

    return AgentRunListResponse(
        runs=[AgentGraphRun.from_internal(e) for e in result.executions],
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_count=result.pagination.total_items,
        total_pages=result.pagination.total_pages,
    )
