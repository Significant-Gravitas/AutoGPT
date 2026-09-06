"""V2 External API - Library Agent Endpoints"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from prisma.enums import APIKeyPermission
from starlette import status

from backend.api.features.library import db as library_db
from backend.data import graph as graph_db
from backend.executor import utils as execution_utils

from ..idempotency import idempotency_key, idempotent_run, replayed_run
from ..integrations.helpers import get_credential_requirements
from ..models import (
    AgentGraphRun,
    AgentRunRequest,
    CredentialRequirement,
    LibraryAgent,
    LibraryAgentUpdateRequest,
)
from ..pagination import Page, PageRequest, page_request
from ..rate_limit import graph_exec_limiter
from ..tenancy import TenantContext, in_tenant, require_permission
from .helpers import assert_can_pay

logger = logging.getLogger(__name__)

agents_router = APIRouter(tags=["library"])


# ============================================================================
# Endpoints
# ============================================================================


@agents_router.get(
    path="/agents",
    summary="List library agents",
    operation_id="listLibraryAgents",
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
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_LIBRARY)),
) -> Page[LibraryAgent]:
    """List agents in the user's library."""
    result = await library_db.list_library_agents(
        user_id=auth.user_id,
        page=page.page,
        page_size=page.limit,
        published=published,
        favorite=favorite,
        organization_id=auth.organization_id,
    )

    return page.paged(
        [LibraryAgent.from_internal(a) for a in result.agents],
        total_count=result.pagination.total_items,
    )


@agents_router.get(
    path="/agents/{agent_id}",
    summary="Get library agent",
    operation_id="getLibraryAgent",
)
async def get_library_agent(
    agent_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_LIBRARY)),
) -> LibraryAgent:
    """Get detailed information about a specific agent in the user's library."""
    agent = in_tenant(
        await library_db.get_library_agent(id=agent_id, user_id=auth.user_id),
        auth,
        f"Agent #{agent_id}",
    )
    return LibraryAgent.from_internal(agent)


@agents_router.patch(
    path="/agents/{agent_id}",
    summary="Update library agent",
    operation_id="updateLibraryAgent",
)
async def update_library_agent(
    request: LibraryAgentUpdateRequest,
    agent_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_LIBRARY)),
) -> LibraryAgent:
    """Update properties of a library agent."""
    await _assert_agent_in_tenant(agent_id, auth)

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
    operation_id="deleteLibraryAgent",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_library_agent(
    agent_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_LIBRARY)),
) -> None:
    """Remove an agent from the user's library."""
    await _assert_agent_in_tenant(agent_id, auth)

    await library_db.delete_library_agent(
        library_agent_id=agent_id,
        user_id=auth.user_id,
    )


@agents_router.post(
    path="/agents/{agent_id}/fork",
    summary="Fork library agent",
    operation_id="forkLibraryAgent",
    status_code=status.HTTP_201_CREATED,
)
async def fork_library_agent(
    agent_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_LIBRARY)),
) -> LibraryAgent:
    """Fork (clone) a library agent.

    Creates a deep copy of the agent's underlying graph and all its nodes,
    assigning new IDs. The cloned graph is added to the user's library as
    an independent agent that can be modified without affecting the original.
    """
    await _assert_agent_in_tenant(agent_id, auth)

    forked = await library_db.fork_library_agent(
        library_agent_id=agent_id,
        user_id=auth.user_id,
    )
    return LibraryAgent.from_internal(forked)


@agents_router.post(
    path="/agents/{agent_id}/runs",
    summary="Execute library agent",
    operation_id="executeLibraryAgent",
    status_code=status.HTTP_202_ACCEPTED,
)
async def execute_agent(
    request: AgentRunRequest,
    agent_id: str,
    idempotency: Annotated[Optional[str], Depends(idempotency_key)] = None,
    auth: TenantContext = Security(require_permission(APIKeyPermission.RUN_AGENT)),
) -> AgentGraphRun:
    """
    Execute an agent from the library.

    Send an `Idempotency-Key` to make a retry safe: a second request carrying the
    same key returns the run the first one started rather than starting another.

    **Rate limit:** 60 requests per minute per user.
    """
    await graph_exec_limiter.check(auth.user_id)

    async with idempotent_run(idempotency, auth.user_id) as claim:
        if claim.existing_run_id:
            return await replayed_run(claim, auth)

        await assert_can_pay(auth)

        library_agent = in_tenant(
            await library_db.get_library_agent(id=agent_id, user_id=auth.user_id),
            auth,
            f"Agent #{agent_id}",
        )

        result = await execution_utils.add_graph_execution(
            graph_id=library_agent.graph_id,
            user_id=auth.user_id,
            inputs=request.inputs,
            graph_version=library_agent.graph_version,
            graph_credentials_inputs=request.credentials_inputs,
            organization_id=auth.organization_id,
            team_id=auth.team_id,
        )
        await claim.record(result.id)
        return AgentGraphRun.from_internal(result)


@agents_router.get(
    path="/agents/{agent_id}/credentials",
    summary="Get library agent credential requirements",
    operation_id="getCredentialRequirementsForLibraryAgent",
)
async def list_agent_credential_requirements(
    agent_id: str,
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(
        require_permission(APIKeyPermission.READ_INTEGRATIONS)
    ),
) -> Page[CredentialRequirement]:
    """List credential requirements and matching user credentials for a library agent."""
    library_agent = in_tenant(
        await library_db.get_library_agent(agent_id, user_id=auth.user_id),
        auth,
        f"Agent #{agent_id}",
    )

    graph = await graph_db.get_graph(
        graph_id=library_agent.graph_id,
        version=library_agent.graph_version,
        user_id=auth.user_id,
        include_subgraphs=True,
        organization_id=auth.organization_id,
    )
    if not graph:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph for agent #{agent_id} not found",
        )

    requirements = await get_credential_requirements(
        graph.credentials_input_schema, auth.user_id
    )
    return page.slice(requirements)


async def _assert_agent_in_tenant(agent_id: str, auth: TenantContext) -> None:
    """404 before mutating an agent the credentials cannot reach."""
    in_tenant(
        await library_db.get_library_agent(id=agent_id, user_id=auth.user_id),
        auth,
        f"Agent #{agent_id}",
    )
