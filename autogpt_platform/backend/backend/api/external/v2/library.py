"""
V2 External API - Library Endpoints

Provides access to the user's agent library and agent execution.
"""

import logging

from fastapi import APIRouter, HTTPException, Path, Query, Security
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.data import execution as execution_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.credit import get_user_credit_model
from backend.executor import utils as execution_utils

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import (
    ExecuteAgentRequest,
    LibraryAgent,
    LibraryAgentsResponse,
    Run,
    RunsListResponse,
)

logger = logging.getLogger(__name__)

library_router = APIRouter()


# ============================================================================
# Endpoints
# ============================================================================


@library_router.get(
    path="/agents",
    summary="List library agents",
    response_model=LibraryAgentsResponse,
)
async def list_library_agents(
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
) -> LibraryAgentsResponse:
    """
    List agents in the user's library.

    The library contains agents the user has created or added from the marketplace.
    """
    result = await library_db.list_library_agents(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
    )

    return LibraryAgentsResponse(
        agents=[LibraryAgent.from_internal(a) for a in result.agents],
        total_count=result.pagination.total_items,
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_pages=result.pagination.total_pages,
    )


@library_router.get(
    path="/agents/favorites",
    summary="List favorite agents",
    response_model=LibraryAgentsResponse,
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
) -> LibraryAgentsResponse:
    """
    List favorite agents in the user's library.
    """
    result = await library_db.list_favorite_library_agents(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
    )

    return LibraryAgentsResponse(
        agents=[LibraryAgent.from_internal(a) for a in result.agents],
        total_count=result.pagination.total_items,
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_pages=result.pagination.total_pages,
    )


@library_router.post(
    path="/agents/{agent_id}/runs",
    summary="Execute an agent",
    response_model=Run,
)
async def execute_agent(
    request: ExecuteAgentRequest,
    agent_id: str = Path(description="Library agent ID"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.RUN_AGENT)
    ),
) -> Run:
    """
    Execute an agent from the library.

    This creates a new run with the provided inputs. The run executes
    asynchronously and you can poll the run status using GET /runs/{run_id}.
    """
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

        return Run.from_internal(result)

    except Exception as e:
        logger.error(f"Failed to execute agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@library_router.get(
    path="/agents/{agent_id}/runs",
    summary="List runs for an agent",
    response_model=RunsListResponse,
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
) -> RunsListResponse:
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

    return RunsListResponse(
        runs=[Run.from_internal(e) for e in result.executions],
        total_count=result.pagination.total_items,
        page=result.pagination.current_page,
        page_size=result.pagination.page_size,
        total_pages=result.pagination.total_pages,
    )
