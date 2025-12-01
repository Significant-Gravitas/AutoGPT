"""External API routes for chat tools - stateless HTTP endpoints.

Note: These endpoints use ephemeral sessions that are not persisted to Redis.
As a result, session-based rate limiting (max_agent_runs, max_agent_schedules)
is not enforced for external API calls. Each request creates a fresh session
with zeroed counters. Rate limiting for external API consumers should be
handled separately (e.g., via API key quotas).
"""

import logging
from typing import Any

from fastapi import APIRouter, Security
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, Field

from backend.data.api_key import APIKeyInfo
from backend.server.external.middleware import require_permission
from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools import (
    find_agent_tool,
    get_agent_details_tool,
    get_required_setup_info_tool,
    run_agent_tool,
    setup_agent_tool,
)
from backend.server.v2.chat.tools.models import ToolResponseBase

logger = logging.getLogger(__name__)

tools_router = APIRouter(prefix="/tools", tags=["tools"])

# Note: We use Security() as a function parameter dependency (api_key: APIKeyInfo = Security(...))
# rather than in the decorator's dependencies= list. This avoids duplicate permission checks
# while still enforcing auth AND giving us access to the api_key for extracting user_id.


# Request models
class FindAgentRequest(BaseModel):
    query: str = Field(..., description="Search query for finding agents")


class AgentSlugRequest(BaseModel):
    username_agent_slug: str = Field(
        ...,
        description="The marketplace agent slug (e.g., 'username/agent-name')",
    )


class GetRequiredSetupInfoRequest(AgentSlugRequest):
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="The input dictionary you plan to provide",
    )


class RunAgentRequest(AgentSlugRequest):
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of input values for the agent",
    )


class SetupAgentRequest(AgentSlugRequest):
    setup_type: str = Field(
        default="schedule",
        description="Type of setup: 'schedule' for cron, 'webhook' for triggers",
    )
    name: str = Field(..., description="Name for this setup/schedule")
    description: str | None = Field(None, description="Description of this setup")
    cron: str | None = Field(
        None,
        description="Cron expression (5 fields: minute hour day month weekday)",
    )
    timezone: str = Field(
        default="UTC",
        description="IANA timezone (e.g., 'America/New_York', 'UTC')",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary with required inputs for the agent",
    )
    webhook_config: dict[str, Any] | None = Field(
        None,
        description="Webhook configuration (required if setup_type is 'webhook')",
    )


def _create_ephemeral_session(user_id: str | None) -> ChatSession:
    """Create an ephemeral session for stateless API requests.

    Note: These sessions are NOT persisted to Redis, so session-based rate
    limiting (max_agent_runs, max_agent_schedules) will not be enforced
    across requests.
    """
    return ChatSession.new(user_id)


@tools_router.post(path="/find-agent")
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


@tools_router.post(path="/get-agent-details")
async def get_agent_details(
    request: AgentSlugRequest,
    api_key: APIKeyInfo = Security(require_permission(APIKeyPermission.USE_TOOLS)),
) -> dict[str, Any]:
    """
    Get detailed information about a specific agent including inputs,
    credentials required, and execution options.

    Args:
        request: Agent slug in format 'username/agent-name'

    Returns:
        Detailed agent information
    """
    session = _create_ephemeral_session(api_key.user_id)
    result = await get_agent_details_tool._execute(
        user_id=api_key.user_id,
        session=session,
        username_agent_slug=request.username_agent_slug,
    )
    return _response_to_dict(result)


@tools_router.post(path="/get-required-setup-info")
async def get_required_setup_info(
    request: GetRequiredSetupInfoRequest,
    api_key: APIKeyInfo = Security(require_permission(APIKeyPermission.USE_TOOLS)),
) -> dict[str, Any]:
    """
    Check if an agent can be set up with the provided input data and credentials.
    Validates that you have all required inputs before running or scheduling.

    Args:
        request: Agent slug and optional inputs to validate

    Returns:
        Setup requirements and user readiness status
    """
    session = _create_ephemeral_session(api_key.user_id)
    result = await get_required_setup_info_tool._execute(
        user_id=api_key.user_id,
        session=session,
        username_agent_slug=request.username_agent_slug,
        inputs=request.inputs,
    )
    return _response_to_dict(result)


@tools_router.post(path="/run-agent")
async def run_agent(
    request: RunAgentRequest,
    api_key: APIKeyInfo = Security(require_permission(APIKeyPermission.USE_TOOLS)),
) -> dict[str, Any]:
    """
    Run an agent immediately (one-off manual execution).

    IMPORTANT: Before calling this endpoint, first call get-agent-details
    to determine what inputs are required.

    Args:
        request: Agent slug and input values

    Returns:
        Execution started response with execution_id
    """
    session = _create_ephemeral_session(api_key.user_id)
    result = await run_agent_tool._execute(
        user_id=api_key.user_id,
        session=session,
        username_agent_slug=request.username_agent_slug,
        inputs=request.inputs,
    )
    return _response_to_dict(result)


@tools_router.post(path="/setup-agent")
async def setup_agent(
    request: SetupAgentRequest,
    api_key: APIKeyInfo = Security(require_permission(APIKeyPermission.USE_TOOLS)),
) -> dict[str, Any]:
    """
    Set up an agent with credentials and configure it for scheduled execution
    or webhook triggers.

    For SCHEDULED execution:
    - Cron format: "minute hour day month weekday"
    - Examples: "0 9 * * 1-5" (9am weekdays), "0 0 * * *" (daily at midnight)
    - Timezone: Use IANA timezone names like "America/New_York", "Europe/London"

    For WEBHOOK triggers:
    - The agent will be triggered by external events

    Args:
        request: Agent slug, setup type, schedule configuration, and inputs

    Returns:
        Schedule or webhook created response
    """
    session = _create_ephemeral_session(api_key.user_id)
    result = await setup_agent_tool._execute(
        user_id=api_key.user_id,
        session=session,
        username_agent_slug=request.username_agent_slug,
        setup_type=request.setup_type,
        name=request.name,
        description=request.description,
        cron=request.cron,
        timezone=request.timezone,
        inputs=request.inputs,
        webhook_config=request.webhook_config,
    )
    return _response_to_dict(result)


def _response_to_dict(result: ToolResponseBase) -> dict[str, Any]:
    """Convert a tool response to a dictionary for JSON serialization."""
    return result.model_dump()
