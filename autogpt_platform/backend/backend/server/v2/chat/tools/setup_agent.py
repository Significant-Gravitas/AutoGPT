"""Tool for setting up an agent with credentials and configuration."""

import logging
from typing import Any

from pydantic import BaseModel

from backend.data.graph import get_graph
from backend.data.model import CredentialsMetaInput
from backend.data.user import get_user_by_id
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.server.v2.chat.config import ChatConfig
from backend.server.v2.chat.model import ChatSession
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

config = ChatConfig()
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
        return "schedule_agent"

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
        session: ChatSession,
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

        session_id = session.session_id
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
            username_agent_slug, user_id, session, **kwargs
        )

        if not isinstance(library_agent, AgentDetails):
            # library agent is an ErrorResponse
            return library_agent

        if library_agent and (
            session.successful_agent_schedules.get(library_agent.graph_id, 0)
            if isinstance(library_agent, AgentDetails)
            else 0 >= config.max_agent_schedules
        ):
            return ErrorResponse(
                message="Maximum number of agent schedules reached. You can't schedule this agent again in this chat session.",
                session_id=session.session_id,
            )
        # At this point we know the user is ready to run the agent
        # Create the schedule for the agent
        from backend.server.v2.library import db as library_db

        # Get the library agent model for scheduling
        lib_agent = await library_db.get_library_agent_by_graph_id(
            graph_id=library_agent.graph_id, user_id=user_id
        )
        if not lib_agent:
            return ErrorResponse(
                message=f"Library agent not found for graph {library_agent.graph_id}",
                session_id=session_id,
            )

        return await self._add_graph_execution_schedule(
            library_agent=lib_agent,
            user_id=user_id,
            cron=cron,
            name=cron_name,
            inputs=inputs,
            credentials=library_agent.required_credentials,
            session=session,
        )

    async def _add_graph_execution_schedule(
        self,
        library_agent: library_model.LibraryAgent,
        user_id: str,
        cron: str,
        name: str,
        inputs: dict[str, Any],
        credentials: dict[str, CredentialsMetaInput],
        session: ChatSession,
        **kwargs,
    ) -> ExecutionStartedResponse | ErrorResponse:
        # Use timezone from request if provided, otherwise fetch from user profile
        user = await get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)
        session_id = session.session_id
        # Map required credentials (schema field names) to actual user credential IDs
        # credentials param contains CredentialsMetaInput with schema field names as keys
        # We need to find the user's actual credentials that match the provider/type
        creds_manager = IntegrationCredentialsManager()
        user_credentials = await creds_manager.store.get_all_creds(user_id)

        # Build a mapping from schema field name -> actual credential ID
        resolved_credentials: dict[str, CredentialsMetaInput] = {}
        missing_credentials: list[str] = []

        for field_name, cred_meta in credentials.items():
            # Find a matching credential from the user's credentials
            matching_cred = next(
                (
                    c
                    for c in user_credentials
                    if c.provider == cred_meta.provider and c.type == cred_meta.type
                ),
                None,
            )

            if matching_cred:
                # Use the actual credential ID instead of the schema field name
                # Create a new CredentialsMetaInput with the actual credential ID
                # but keep the same provider/type from the original meta
                resolved_credentials[field_name] = CredentialsMetaInput(
                    id=matching_cred.id,
                    provider=cred_meta.provider,
                    type=cred_meta.type,
                    title=cred_meta.title,
                )
            else:
                missing_credentials.append(
                    f"{cred_meta.title} ({cred_meta.provider}/{cred_meta.type})"
                )

        if missing_credentials:
            return ErrorResponse(
                message=f"Cannot execute agent: missing {len(missing_credentials)} required credential(s). You need to call the get_required_setup_info tool to setup the credentials.",
                session_id=session_id,
            )

        result = await get_scheduler_client().add_execution_schedule(
            user_id=user_id,
            graph_id=library_agent.graph_id,
            graph_version=library_agent.graph_version,
            name=name,
            cron=cron,
            input_data=inputs,
            input_credentials=resolved_credentials,
            user_timezone=user_timezone,
        )

        # Convert the next_run_time back to user timezone for display
        if result.next_run_time:
            result.next_run_time = convert_utc_time_to_user_timezone(
                result.next_run_time, user_timezone
            )

        session.successful_agent_schedules[library_agent.graph_id] = (
            session.successful_agent_schedules.get(library_agent.graph_id, 0) + 1
        )

        return ExecutionStartedResponse(
            message="Agent execution successfully scheduled. Do not run this tool again unless specifically asked to run the agent again.",
            session_id=session_id,
            execution_id=result.id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.name,
        )

    async def _get_or_add_library_agent(
        self, agent_id: str, user_id: str, session: ChatSession, **kwargs
    ) -> AgentDetails | ErrorResponse:
        # Call _execute directly since we're calling internally from another tool
        session_id = session.session_id
        response = await GetRequiredSetupInfoTool()._execute(user_id, session, **kwargs)

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

        # Get the graph using the graph_id and graph_version from the setup response
        if not response.graph_id or not response.graph_version:
            return ErrorResponse(
                message=f"Graph information not available for {agent_id}",
                session_id=session_id,
            )

        graph = await get_graph(
            graph_id=response.graph_id,
            version=response.graph_version,
            user_id=None,  # Public access for store graphs
            include_subgraphs=True,
        )
        if not graph:
            return ErrorResponse(
                message=f"Graph {agent_id} ({response.graph_id}v{response.graph_version}) not found",
                session_id=session_id,
            )

        recommended_schedule_cron = graph.recommended_schedule_cron

        # Extract credentials from the JSON schema properties
        credentials_input_schema = graph.credentials_input_schema
        required_credentials: dict[str, CredentialsMetaInput] = {}
        if (
            isinstance(credentials_input_schema, dict)
            and "properties" in credentials_input_schema
        ):
            for cred_name, cred_schema in credentials_input_schema[
                "properties"
            ].items():
                # Get provider from credentials_provider array or properties.provider.const
                provider = "unknown"
                if (
                    "credentials_provider" in cred_schema
                    and cred_schema["credentials_provider"]
                ):
                    provider = cred_schema["credentials_provider"][0]
                elif (
                    "properties" in cred_schema
                    and "provider" in cred_schema["properties"]
                ):
                    provider = cred_schema["properties"]["provider"].get(
                        "const", "unknown"
                    )

                # Get type from credentials_types array or properties.type.const
                cred_type = "api_key"  # Default
                if (
                    "credentials_types" in cred_schema
                    and cred_schema["credentials_types"]
                ):
                    cred_type = cred_schema["credentials_types"][0]
                elif (
                    "properties" in cred_schema and "type" in cred_schema["properties"]
                ):
                    cred_type = cred_schema["properties"]["type"].get(
                        "const", "api_key"
                    )

                required_credentials[cred_name] = CredentialsMetaInput(
                    id=cred_name,
                    title=cred_schema.get("title", cred_name),
                    provider=provider,  # type: ignore
                    type=cred_type,
                )

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
