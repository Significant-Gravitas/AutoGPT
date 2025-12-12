"""
Agent Execution endpoints for external OAuth clients.

Allows external applications to:
- Execute agents using granted credentials
- Poll execution status
- Cancel running executions
- Get available capabilities

External apps can only use credentials they have been granted access to.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Security
from prisma.enums import AgentExecutionStatus
from pydantic import BaseModel, Field

from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.db import prisma
from backend.data.execution import ExecutionContext, GrantResolverContext
from backend.executor.utils import add_graph_execution
from backend.integrations.grant_resolver import (
    GrantValidationError,
    create_resolver_from_oauth_token,
)
from backend.integrations.webhook_notifier import validate_webhook_url
from backend.server.external.oauth_middleware import OAuthTokenInfo, require_scope

logger = logging.getLogger(__name__)

execution_router = APIRouter(prefix="/executions", tags=["executions"])


# ================================================================
# Request/Response Models
# ================================================================


class ExecuteAgentRequest(BaseModel):
    """Request to execute an agent."""

    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Input values for the agent",
    )
    grant_ids: Optional[list[str]] = Field(
        default=None,
        description="Specific grant IDs to use. If not provided, uses all available grants.",
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="URL to receive execution status webhooks",
    )


class ExecuteAgentResponse(BaseModel):
    """Response from starting an agent execution."""

    execution_id: str
    status: str
    message: str


class ExecutionStatusResponse(BaseModel):
    """Response with execution status."""

    execution_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    outputs: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class GrantInfo(BaseModel):
    """Summary of a credential grant for capabilities."""

    grant_id: str
    provider: str
    scopes: list[str]


class CapabilitiesResponse(BaseModel):
    """Response describing what the client can do."""

    user_id: str
    client_id: str
    grants: list[GrantInfo]
    available_scopes: list[str]


# ================================================================
# Endpoints
# ================================================================


@execution_router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities(
    token: OAuthTokenInfo = Security(require_scope("agents:execute")),
) -> CapabilitiesResponse:
    """
    Get the capabilities available to this client for the authenticated user.

    Returns information about:
    - Available credential grants (NOT credential values)
    - Scopes the client has access to
    """
    try:
        resolver = await create_resolver_from_oauth_token(
            user_id=token.user_id,
            client_public_id=token.client_id,
        )
        credentials_info = await resolver.get_available_credentials()

        grants = [
            GrantInfo(
                grant_id=info["grant_id"],
                provider=info["provider"],
                scopes=info["granted_scopes"],
            )
            for info in credentials_info
        ]

        return CapabilitiesResponse(
            user_id=token.user_id,
            client_id=token.client_id,
            grants=grants,
            available_scopes=token.scopes,
        )
    except GrantValidationError:
        # No grants available is not an error, just empty capabilities
        return CapabilitiesResponse(
            user_id=token.user_id,
            client_id=token.client_id,
            grants=[],
            available_scopes=token.scopes,
        )


@execution_router.post(
    "/agents/{agent_id}/execute",
    response_model=ExecuteAgentResponse,
)
async def execute_agent(
    agent_id: str,
    request: ExecuteAgentRequest,
    token: OAuthTokenInfo = Security(require_scope("agents:execute")),
) -> ExecuteAgentResponse:
    """
    Execute an agent using granted credentials.

    The agent must be accessible to the user, and the client must have
    valid credential grants that satisfy the agent's requirements.

    Args:
        agent_id: The agent (graph) ID to execute
        request: Execution parameters including inputs and optional grant IDs
    """
    # Verify the agent exists and user has access
    # First try to get the latest version
    graph = await graph_db.get_graph(
        graph_id=agent_id,
        version=None,
        user_id=token.user_id,
    )

    if not graph:
        # Try to find it in the store (public agents)
        graph = await graph_db.get_graph(
            graph_id=agent_id,
            version=None,
            user_id=None,
            skip_access_check=True,
        )
        if not graph:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found or not accessible",
            )

    # Initialize the grant resolver to validate grants exist
    # The resolver context will be passed to the execution engine
    grant_resolver_context = None
    try:
        resolver = await create_resolver_from_oauth_token(
            user_id=token.user_id,
            client_public_id=token.client_id,
            grant_ids=request.grant_ids,
        )
        # Get available credentials info to build resolver context
        credentials_info = await resolver.get_available_credentials()
        grant_resolver_context = GrantResolverContext(
            client_db_id=resolver.client_id,
            grant_ids=[c["grant_id"] for c in credentials_info],
        )
    except GrantValidationError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Grant validation failed: {str(e)}",
        )

    try:
        # Build execution context with grant resolver info
        execution_context = ExecutionContext(
            grant_resolver_context=grant_resolver_context,
        )

        # Execute the agent with grant resolver context
        graph_exec = await add_graph_execution(
            graph_id=agent_id,
            user_id=token.user_id,
            inputs=request.inputs,
            graph_version=graph.version,
            execution_context=execution_context,
        )

        # Log the execution for audit
        logger.info(
            f"External execution started: agent={agent_id}, "
            f"execution={graph_exec.id}, client={token.client_id}, "
            f"user={token.user_id}"
        )

        # Register webhook if provided
        if request.webhook_url:
            # Get client to check webhook domains
            client = await prisma.oauthclient.find_unique(
                where={"clientId": token.client_id}
            )
            if client:
                if not validate_webhook_url(request.webhook_url, client.webhookDomains):
                    raise HTTPException(
                        status_code=400,
                        detail="Webhook URL not in allowed domains for this client",
                    )

                # Store webhook registration with client's webhook secret
                await prisma.executionwebhook.create(
                    data={  # type: ignore[typeddict-item]
                        "executionId": graph_exec.id,
                        "webhookUrl": request.webhook_url,
                        "clientId": client.id,
                        "userId": token.user_id,
                        "secret": client.webhookSecret,
                    }
                )
                logger.info(
                    f"Registered webhook for execution {graph_exec.id}: {request.webhook_url}"
                )

        return ExecuteAgentResponse(
            execution_id=graph_exec.id,
            status="queued",
            message="Agent execution has been queued",
        )

    except ValueError as e:
        # Client error - invalid input or configuration
        logger.warning(
            f"Invalid execution request: agent={agent_id}, "
            f"client={token.client_id}, error={str(e)}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}",
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception:
        # Server error - log full exception but don't expose details to client
        logger.exception(
            f"Unexpected error starting execution: agent={agent_id}, "
            f"client={token.client_id}"
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while starting execution",
        )


@execution_router.get(
    "/{execution_id}",
    response_model=ExecutionStatusResponse,
)
async def get_execution_status(
    execution_id: str,
    token: OAuthTokenInfo = Security(require_scope("agents:execute")),
) -> ExecutionStatusResponse:
    """
    Get the status of an agent execution.

    Returns current status, outputs (if completed), and any error messages.
    """
    graph_exec = await execution_db.get_graph_execution(
        user_id=token.user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )

    if not graph_exec:
        raise HTTPException(
            status_code=404,
            detail=f"Execution {execution_id} not found",
        )

    # Build response
    outputs = None
    error = None

    if graph_exec.status == AgentExecutionStatus.COMPLETED:
        outputs = graph_exec.outputs
    elif graph_exec.status == AgentExecutionStatus.FAILED:
        # Get error from execution stats
        # Note: Currently no standard error field in stats, but could be added
        error = "Execution failed"

    return ExecutionStatusResponse(
        execution_id=execution_id,
        status=graph_exec.status.value,
        started_at=graph_exec.started_at,
        completed_at=graph_exec.ended_at,
        outputs=outputs,
        error=error,
    )


@execution_router.post("/{execution_id}/cancel")
async def cancel_execution(
    execution_id: str,
    token: OAuthTokenInfo = Security(require_scope("agents:execute")),
) -> dict:
    """
    Cancel a running agent execution.

    Only executions in QUEUED or RUNNING status can be cancelled.
    """
    graph_exec = await execution_db.get_graph_execution(
        user_id=token.user_id,
        execution_id=execution_id,
        include_node_executions=False,
    )

    if not graph_exec:
        raise HTTPException(
            status_code=404,
            detail=f"Execution {execution_id} not found",
        )

    # Check if execution can be cancelled
    if graph_exec.status not in [
        AgentExecutionStatus.QUEUED,
        AgentExecutionStatus.RUNNING,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel execution with status {graph_exec.status.value}",
        )

    # Update execution status to TERMINATED
    # Note: This is a simplified implementation. A full implementation would
    # need to signal the executor to stop processing.
    await prisma.agentgraphexecution.update(
        where={"id": execution_id},
        data={"executionStatus": AgentExecutionStatus.TERMINATED},
    )

    logger.info(
        f"Execution terminated: execution={execution_id}, "
        f"client={token.client_id}, user={token.user_id}"
    )

    return {"message": "Execution terminated", "execution_id": execution_id}
