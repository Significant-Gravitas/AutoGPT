"""Tool for setting up an agent with credentials and configuration."""

import logging
from typing import Any

from pydantic import BaseModel

from backend.data.graph import get_graph
from backend.data.model import CredentialsMetaInput
from backend.data.user import get_user_by_id
from backend.executor import utils as execution_utils
from backend.server.v2.chat.tools.get_required_setup_info import (
    GetRequiredSetupInfoTool,
)
from backend.server.v2.chat.tools.models import (
    ExecutionStartedResponse,
    SetupInfo,
    SetupRequirementsResponse,
)
from backend.server.v2.library import db as library_db
from backend.server.v2.library import model as library_model
from backend.util.clients import get_scheduler_client
from backend.util.timezone_utils import (
    convert_utc_time_to_user_timezone,
    get_user_timezone_or_utc,
)

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class AgentDetails(BaseModel):
    graph_name: str
    graph_id: str
    graph_version: int
    recommended_schedule_cron: str | None
    required_credentials: dict[str, CredentialsMetaInput]


class SetupAgentTool(BaseTool):
    """Tool for setting up an agent with scheduled execution or webhook triggers."""

    @property
    def name(self) -> str:
        return "setup_agent"

    @property
    def description(self) -> str:
        return """Set up an agent with credentials and configure it for scheduled execution or webhook triggers.
        IMPORTANT: Before calling this tool, you MUST first call get_agent_details to determine what inputs are required.

        For SCHEDULED execution:
        - Cron format: "minute hour day month weekday" (e.g., "0 9 * * 1-5" = 9am weekdays)
        - Common patterns: "0 * * * *" (hourly), "0 0 * * *" (daily at midnight), "0 9 * * 1" (Mondays at 9am)
        - Timezone: Use IANA timezone names like "America/New_York", "Europe/London", "Asia/Tokyo"
        - The 'inputs' parameter must contain ALL required inputs from get_agent_details as a dictionary

        For WEBHOOK triggers:
        - The agent will be triggered by external events
        - Still requires all input values from get_agent_details"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "username_agent_slug": {
                    "type": "string",
                    "description": "The marketplace agent slug (e.g., 'username/agent-name')",
                },
                "setup_type": {
                    "type": "string",
                    "enum": ["schedule", "webhook"],
                    "description": "Type of setup: 'schedule' for cron, 'webhook' for triggers.",
                },
                "name": {
                    "type": "string",
                    "description": "Name for this setup/schedule (e.g., 'Daily Report', 'Weekly Summary')",
                },
                "description": {
                    "type": "string",
                    "description": "Description of this setup",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression (5 fields: minute hour day month weekday). Examples: '0 9 * * 1-5' (9am weekdays), '*/30 * * * *' (every 30 min)",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone (e.g., 'America/New_York', 'Europe/London', 'UTC'). Defaults to UTC if not specified.",
                },
                "inputs": {
                    "type": "object",
                    "description": 'REQUIRED: Dictionary with ALL required inputs from get_agent_details. Format: {"input_name": value}',
                    "additionalProperties": True,
                },
                "webhook_config": {
                    "type": "object",
                    "description": "Webhook configuration (required if setup_type is 'webhook')",
                    "additionalProperties": True,
                },
            },
            "required": ["username_agent_slug", "setup_type"],
        }

    @property
    def requires_auth(self) -> bool:
        """This tool requires authentication."""
        return True

    async def _execute(
        self,
        user_id: str | None,
        session_id: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Set up an agent with configuration.

        Args:
            user_id: Authenticated user ID
            session_id: Chat session ID
            **kwargs: Setup parameters

        Returns:
            JSON formatted setup result

        """
        assert (
            user_id is not None
        ), "User ID is required to run an agent. Superclass enforces authentication."
        setup_type = kwargs.get("setup_type", "schedule").strip()
        if setup_type != "schedule":
            return ErrorResponse(
                message="Only schedule setup is supported at this time",
                session_id=session_id,
            )
        else:
            cron = kwargs.get("cron", "").strip()
            cron_name = kwargs.get("name", "").strip()
            if not cron or not cron_name:
                return ErrorResponse(
                    message="Cron and name are required for schedule setup",
                    session_id=session_id,
                )

        username_agent_slug = kwargs.get("username_agent_slug", "").strip()
        inputs = kwargs.get("inputs", {})

        library_agent = await self._get_or_add_library_agent(
            username_agent_slug, user_id, session_id, **kwargs
        )
        if not isinstance(library_agent, AgentDetails):
            # library agent is an ErrorResponse
            return library_agent

        # At this point we know the user is ready to run the agent
        # So we can execute the agent
        execution = await execution_utils.add_graph_execution(
            graph_id=library_agent.graph_id,
            user_id=user_id,
            inputs=inputs,
        )
        return ExecutionStartedResponse(
            message="Agent execution started",
            session_id=session_id,
            execution_id=execution.id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.graph_name,
        )

    async def _add_graph_execution_schedule(
        self,
        library_agent: library_model.LibraryAgent,
        user_id: str,
        cron: str,
        name: str,
        inputs: dict[str, Any],
        credentials: dict[str, CredentialsMetaInput],
        session_id: str,
        **kwargs,
    ) -> ExecutionStartedResponse | ErrorResponse:
        # Use timezone from request if provided, otherwise fetch from user profile
        user = await get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)

        result = await get_scheduler_client().add_execution_schedule(
            user_id=user_id,
            graph_id=library_agent.graph_id,
            graph_version=library_agent.graph_version,
            name=name,
            cron=cron,
            input_data=inputs,
            input_credentials=credentials,
            user_timezone=user_timezone,
        )

        # Convert the next_run_time back to user timezone for display
        if result.next_run_time:
            result.next_run_time = convert_utc_time_to_user_timezone(
                result.next_run_time, user_timezone
            )
        return ExecutionStartedResponse(
            message="Agent execution started",
            session_id=session_id,
            execution_id=result.id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.name,
        )

    async def _get_or_add_library_agent(
        self, agent_id: str, user_id: str, session_id: str, **kwargs
    ) -> AgentDetails | ErrorResponse:
        # Call _execute directly since we're calling internally from another tool
        response = await GetRequiredSetupInfoTool()._execute(
            user_id, session_id, **kwargs
        )

        if not isinstance(response, SetupRequirementsResponse):
            return ErrorResponse(
                message="Failed to get required setup information",
                session_id=session_id,
            )

        setup_info = SetupInfo.model_validate(response.setup_info)

        if not setup_info.user_readiness.ready_to_run:
            return ErrorResponse(
                message=f"User is not ready to run the agent. User Readiness: {setup_info.user_readiness.model_dump_json()} Requirments: {setup_info.requirements}",
                session_id=session_id,
            )

        graph = await get_graph(agent_id)
        if not graph:
            return ErrorResponse(
                message=f"Graph {agent_id} not found",
                session_id=session_id,
            )

        recommended_schedule_cron = graph.recommended_schedule_cron
        required_credentials = graph.credentials_input_schema

        # Check if we already have a library agent for this graph
        existing_library_agent = await library_db.get_library_agent_by_graph_id(
            graph_id=graph.id, user_id=user_id
        )
        if not existing_library_agent:
            # Now we need to add the graph to the users library
            library_agents: list[library_model.LibraryAgent] = (
                await library_db.create_library_agent(
                    graph=graph,
                    user_id=user_id,
                    create_library_agents_for_sub_graphs=False,
                )
            )
            assert len(library_agents) == 1, "Expected 1 library agent to be created"
            library_agent = library_agents[0]
        else:
            library_agent = existing_library_agent

        return AgentDetails(
            graph_name=graph.name,
            graph_id=library_agent.graph_id,
            graph_version=library_agent.graph_version,
            recommended_schedule_cron=recommended_schedule_cron,
            required_credentials=required_credentials,
        )
