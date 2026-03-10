"""V2 External API - Library Agent Endpoints"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Security
from prisma.enums import APIKeyPermission
from starlette import status

from backend.api.external.middleware import require_permission
from backend.api.features.library import db as library_db
from backend.data import graph as graph_db
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.credit import get_user_credit_model
from backend.executor import utils as execution_utils

from ..common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from ..integrations.helpers import get_credential_requirements
from ..models import (
    AgentGraphRun,
    AgentRunRequest,
    CredentialRequirementsResponse,
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
    published: Optional[bool] = Query(
        default=None,
        description="Filter by marketplace publish status",
    ),
    favorite: Optional[bool] = Query(
        default=None,
        description="Filter by `isFavorite` attribute",
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryAgentListResponse:
    """List agents in the user's library."""
    result = await library_db.list_library_agents(
        user_id=auth.user_id,
        page=page,
        page_size=page_size,
        published=published,
        favorite=favorite,
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
    summary="Get library agent",
)
async def get_library_agent(
    agent_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_LIBRARY)
    ),
) -> LibraryAgent:
    """Get detailed information about a specific agent in the user's library."""
    agent = await library_db.get_library_agent(
        id=agent_id,
        user_id=auth.user_id,
    )
    return LibraryAgent.from_internal(agent)


@agents_router.patch(
    path="/agents/{agent_id}",
    summary="Update library agent",
)
async def update_library_agent(
    request: LibraryAgentUpdateRequest,
    agent_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryAgent:
    """Update properties of a library agent."""
    updated = await library_db.update_library_agent(
        library_agent_id=agent_id,
        user_id=auth.user_id,
        auto_update_version=request.auto_update_version,
        graph_version=request.graph_version,
        is_favorite=request.is_favorite,
        is_archived=request.is_archived,
        folder_id=request.folder_id,
    )
    return LibraryAgent.from_internal(updated)


@agents_router.delete(
    path="/agents/{agent_id}",
    summary="Delete library agent",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_library_agent(
    agent_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> None:
    """Remove an agent from the user's library."""
    await library_db.delete_library_agent(
        library_agent_id=agent_id,
        user_id=auth.user_id,
    )


@agents_router.post(
    path="/agents/{agent_id}/fork",
    summary="Fork library agent",
    status_code=status.HTTP_201_CREATED,
)
async def fork_library_agent(
    agent_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.WRITE_LIBRARY)
    ),
) -> LibraryAgent:
    """Fork (clone) a library agent."""
    forked = await library_db.fork_library_agent(
        library_agent_id=agent_id,
        user_id=auth.user_id,
    )
    return LibraryAgent.from_internal(forked)


@agents_router.post(
    path="/agents/{agent_id}/runs",
    summary="Execute library agent",
)
async def execute_agent(
    request: AgentRunRequest,
    agent_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.RUN_AGENT)
    ),
) -> AgentGraphRun:
    """Execute an agent from the library."""
    execute_limiter.check(auth.user_id)

    # Check credit balance
    user_credit_model = await get_user_credit_model(auth.user_id)
    current_balance = await user_credit_model.get_credits(auth.user_id)
    if current_balance <= 0:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient balance to execute the agent. Please top up your account.",
        )

    # Get the library agent to find the graph ID and version
    library_agent = await library_db.get_library_agent(
        id=agent_id,
        user_id=auth.user_id,
    )

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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@agents_router.get(
    path="/agents/{agent_id}/credentials",
    summary="Get agent credentials",
)
async def list_agent_credential_requirements(
    agent_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> CredentialRequirementsResponse:
    """List credential requirements and matching user credentials for a library agent."""
    library_agent = await library_db.get_library_agent(agent_id, user_id=auth.user_id)

    graph = await graph_db.get_graph(
        graph_id=library_agent.graph_id,
        version=library_agent.graph_version,
        user_id=auth.user_id,
        include_subgraphs=True,
    )
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph for agent #{agent_id} not found",
        )

    requirements = await get_credential_requirements(
        graph.credentials_input_schema, auth.user_id
    )
    return CredentialRequirementsResponse(requirements=requirements)
