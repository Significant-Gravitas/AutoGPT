"""External API routes for chat tools - stateless HTTP endpoints."""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Security
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.data.api_key import APIKeyInfo
from backend.server.external.middleware import require_permission
from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools import find_agent_tool, run_agent_tool
from backend.server.v2.chat.tools.models import ToolResponseBase

logger = logging.getLogger(__name__)

tools_router = APIRouter(prefix="/tools", tags=["tools"])


# Request models
class FindAgentRequest(BaseModel):
    query: str = Field(..., description="Search query for finding agents")


class RunAgentRequest(BaseModel):
    """Unified request for all agent operations."""

    action: Literal["get_details", "validate", "run", "schedule"] = Field(
        default="run",
        description="Action to perform: get_details, validate, run, or schedule",
    )
    username_agent_slug: str = Field(
        ...,
        description="The marketplace agent slug (e.g., 'username/agent-name')",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of input values for the agent",
    )
    schedule_name: str | None = Field(
        None,
        description="Name for scheduled execution (required for action='schedule')",
    )
    cron: str | None = Field(
        None,
        description="Cron expression (5 fields: minute hour day month weekday)",
    )
    timezone: str = Field(
        default="UTC",
        description="IANA timezone (e.g., 'America/New_York', 'UTC')",
    )


def _create_ephemeral_session(user_id: str | None) -> ChatSession:
    """Create an ephemeral session for stateless API requests."""
    return ChatSession.new(user_id)


@tools_router.post(
    path="/find-agent",
    dependencies=[Security(require_permission(APIKeyPermission.USE_TOOLS))],
)
async def find_agent(
    request: FindAgentRequest,
    api_key: APIKeyInfo = Security(require_permission(APIKeyPermission.USE_TOOLS)),
) -> dict[str, Any]:
    """
    Search for agents in the marketplace based on capabilities and user needs.

    Args:
        request: Search query for finding agents

    Returns:
        List of matching agents or no results response
    """
    session = _create_ephemeral_session(api_key.user_id)
    result = await find_agent_tool._execute(
        user_id=api_key.user_id,
        session=session,
        query=request.query,
    )
    return _response_to_dict(result)


@tools_router.post(
    path="/run-agent",
    dependencies=[Security(require_permission(APIKeyPermission.USE_TOOLS))],
)
async def run_agent(
    request: RunAgentRequest,
    api_key: APIKeyInfo = Security(require_permission(APIKeyPermission.USE_TOOLS)),
) -> dict[str, Any]:
    """
    Unified endpoint for all agent operations.

    Actions:
    - **get_details**: Get agent info, required inputs, and credentials
    - **validate**: Check if credentials and inputs are ready
    - **run**: Execute agent immediately with provided inputs
    - **schedule**: Set up scheduled execution with cron expression

    Workflow:
    1. Call with action="get_details" to see what the agent needs
    2. If credentials are needed, configure them via the platform
    3. Call with action="validate" to confirm readiness
    4. Call with action="run" or action="schedule" with inputs

    For scheduled execution:
    - Cron format: "minute hour day month weekday"
    - Examples: "0 9 * * 1-5" (9am weekdays), "0 0 * * *" (daily at midnight)
    - Timezone: Use IANA timezone names like "America/New_York"

    Args:
        request: Action, agent slug, and optional inputs/schedule config

    Returns:
        Response varies by action - details, validation status, or execution started
    """
    session = _create_ephemeral_session(api_key.user_id)
    result = await run_agent_tool._execute(
        user_id=api_key.user_id,
        session=session,
        action=request.action,
        username_agent_slug=request.username_agent_slug,
        inputs=request.inputs,
        schedule_name=request.schedule_name or "",
        cron=request.cron or "",
        timezone=request.timezone,
    )
    return _response_to_dict(result)


def _response_to_dict(result: ToolResponseBase) -> dict[str, Any]:
    """Convert a tool response to a dictionary for JSON serialization."""
    return result.model_dump()
