"""Unified tool for agent operations: get details, validate, run, and schedule."""

import logging
from typing import Any, Literal

from backend.data.user import get_user_by_id
from backend.executor import utils as execution_utils
from backend.server.v2.chat.config import ChatConfig
from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    AgentDetails,
    AgentDetailsResponse,
    ErrorResponse,
    ExecutionOptions,
    ExecutionStartedResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from backend.server.v2.chat.tools.utils import (
    check_user_has_required_credentials,
    extract_credentials_from_schema,
    fetch_graph_from_store_slug,
    get_or_create_library_agent,
    match_user_credentials_to_graph,
)
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import DatabaseError, NotFoundError
from backend.util.timezone_utils import (
    convert_utc_time_to_user_timezone,
    get_user_timezone_or_utc,
)

logger = logging.getLogger(__name__)
config = ChatConfig()

ActionType = Literal["get_details", "validate", "run", "schedule"]


class RunAgentTool(BaseTool):
    """Unified tool for agent operations.

    Supports four actions:
    - get_details: Get agent information and requirements
    - validate: Check if user has required credentials and inputs
    - run: Execute the agent immediately
    - schedule: Set up scheduled execution with cron
    """

    @property
    def name(self) -> str:
        return "run_agent"

    @property
    def description(self) -> str:
        return """Unified tool for agent operations. Use different actions:

        1. action="get_details": Get agent info and requirements
        2. action="validate": Check if credentials and inputs are ready
        3. action="run": Execute agent immediately with provided inputs
        4. action="schedule": Set up scheduled execution with cron

        WORKFLOW:
        1. First call with action="get_details" to see what the agent needs
        2. If credentials are needed, wait for user to configure them
        3. Call with action="validate" to confirm readiness
        4. Finally call with action="run" or action="schedule" with inputs"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get_details", "validate", "run", "schedule"],
                    "description": "Action to perform: get_details, validate, run, or schedule",
                },
                "username_agent_slug": {
                    "type": "string",
                    "description": "Agent identifier in format 'username/agent-name'",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input values for the agent (required for run/schedule)",
                    "additionalProperties": True,
                },
                "schedule_name": {
                    "type": "string",
                    "description": "Name for scheduled execution (required for schedule)",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression for schedule (5 fields: min hour day month weekday)",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone for schedule (default: UTC)",
                },
            },
            "required": ["action", "username_agent_slug"],
        }

    @property
    def requires_auth(self) -> bool:
        """All actions require authentication."""
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Route to appropriate action handler."""
        action: ActionType = kwargs.get("action", "run")  # type: ignore
        agent_slug = kwargs.get("username_agent_slug", "").strip()
        session_id = session.session_id

        # Validate agent slug format
        if not agent_slug or "/" not in agent_slug:
            return ErrorResponse(
                message="Please provide an agent slug in format 'username/agent-name'",
                session_id=session_id,
            )

        # Auth is required for all actions
        if not user_id:
            return ErrorResponse(
                message="Authentication required. Please sign in to use this tool.",
                session_id=session_id,
            )

        if action == "get_details":
            return await self._get_details(user_id, session, agent_slug, **kwargs)
        elif action == "validate":
            return await self._validate_setup(user_id, session, agent_slug, **kwargs)
        elif action == "run":
            return await self._run_agent(user_id, session, agent_slug, **kwargs)
        elif action == "schedule":
            return await self._schedule_agent(user_id, session, agent_slug, **kwargs)
        else:
            return ErrorResponse(
                message=f"Unknown action: {action}. Use: get_details, validate, run, or schedule",
                session_id=session_id,
            )

    async def _get_details(
        self,
        user_id: str,
        session: ChatSession,
        agent_slug: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Get detailed information about an agent."""
        session_id = session.session_id

        try:
            username, agent_name = agent_slug.split("/", 1)
            graph, store_agent = await fetch_graph_from_store_slug(username, agent_name)

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_slug}' not found in marketplace",
                    session_id=session_id,
                )

            # Extract credentials from schema
            credentials = extract_credentials_from_schema(
                graph.credentials_input_schema
            )

            # Check if user has required credentials
            missing_creds = await check_user_has_required_credentials(
                user_id, credentials
            )

            trigger_info = (
                graph.trigger_setup_info.model_dump()
                if graph.trigger_setup_info
                else None
            )

            agent_details = AgentDetails(
                id=graph.id,
                name=graph.name,
                description=graph.description,
                inputs=graph.input_schema,
                credentials=credentials,
                execution_options=ExecutionOptions(
                    manual=trigger_info is None,
                    scheduled=trigger_info is None,
                    webhook=trigger_info is not None,
                ),
                trigger_info=trigger_info,
            )

            # Build next action hint message
            if missing_creds:
                next_msg = (
                    f"Agent requires {len(missing_creds)} credential(s). "
                    "User needs to configure credentials, then call run_agent with action='validate'."
                )
            elif credentials:
                next_msg = (
                    "Agent requires credentials. User has them configured. "
                    "Call run_agent with action='validate' to confirm readiness."
                )
            else:
                next_msg = "Agent is ready. Call run_agent with action='run' and provide required inputs."

            return AgentDetailsResponse(
                message=f"Found agent '{agent_details.name}'. {next_msg}",
                session_id=session_id,
                agent=agent_details,
                user_authenticated=True,
                graph_id=graph.id,
                graph_version=graph.version,
            )

        except NotFoundError:
            return ErrorResponse(
                message=f"Agent '{agent_slug}' not found",
                session_id=session_id,
            )
        except DatabaseError as e:
            logger.error(f"Database error getting agent details: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to get agent details: {e!s}",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error getting agent details: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to get agent details: {e!s}",
                error=str(e),
                session_id=session_id,
            )

    async def _validate_setup(
        self,
        user_id: str,
        session: ChatSession,
        agent_slug: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Validate that user has required credentials and inputs."""
        session_id = session.session_id
        inputs = kwargs.get("inputs", {})

        try:
            username, agent_name = agent_slug.split("/", 1)
            graph, store_agent = await fetch_graph_from_store_slug(username, agent_name)

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_slug}' not found",
                    session_id=session_id,
                )

            # Extract and check credentials
            credentials = extract_credentials_from_schema(
                graph.credentials_input_schema
            )
            missing_creds = await check_user_has_required_credentials(
                user_id, credentials
            )

            # Check inputs
            required_fields = set(graph.input_schema.get("required", []))
            provided_inputs = set(inputs.keys())
            missing_inputs = required_fields - provided_inputs

            # Build user readiness
            missing_credentials_dict = {c.id: c.model_dump() for c in missing_creds}
            user_readiness = UserReadiness(
                has_all_credentials=len(missing_creds) == 0,
                missing_credentials=missing_credentials_dict,
                ready_to_run=len(missing_inputs) == 0 and len(missing_creds) == 0,
            )

            # Build execution modes
            trigger_info = graph.trigger_setup_info
            execution_modes = []
            if trigger_info is None:
                execution_modes.extend(["manual", "scheduled"])
            else:
                execution_modes.append("webhook")

            # Build inputs list
            inputs_list = []
            if (
                isinstance(graph.input_schema, dict)
                and "properties" in graph.input_schema
            ):
                for field_name, field_schema in graph.input_schema[
                    "properties"
                ].items():
                    inputs_list.append(
                        {
                            "name": field_name,
                            "title": field_schema.get("title", field_name),
                            "type": field_schema.get("type", "string"),
                            "description": field_schema.get("description", ""),
                            "required": field_name
                            in graph.input_schema.get("required", []),
                        }
                    )

            requirements = {
                "credentials": credentials,
                "inputs": inputs_list,
                "execution_modes": execution_modes,
            }

            # Build message
            if missing_creds:
                message = (
                    "User needs to configure credentials before proceeding. "
                    "Wait for confirmation, then call run_agent with action='validate' again."
                )
            elif missing_inputs:
                missing_names = ", ".join(missing_inputs)
                message = (
                    f"Missing required inputs: {missing_names}. "
                    f"Call run_agent with action='run' and provide these inputs."
                )
            else:
                message = (
                    "Agent is ready to run. "
                    "Call run_agent with action='run' (or action='schedule' for scheduled execution)."
                )

            return SetupRequirementsResponse(
                message=message,
                session_id=session_id,
                setup_info=SetupInfo(
                    agent_id=graph.id,
                    agent_name=graph.name,
                    user_readiness=user_readiness,
                    requirements=requirements,
                ),
                graph_id=graph.id,
                graph_version=graph.version,
            )

        except Exception as e:
            logger.error(f"Error validating setup: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to validate setup: {e!s}",
                error=str(e),
                session_id=session_id,
            )

    async def _run_agent(
        self,
        user_id: str,
        session: ChatSession,
        agent_slug: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute an agent immediately."""
        session_id = session.session_id
        inputs = kwargs.get("inputs", {})

        try:
            username, agent_name = agent_slug.split("/", 1)
            graph, store_agent = await fetch_graph_from_store_slug(username, agent_name)

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_slug}' not found",
                    session_id=session_id,
                )

            # Check rate limits
            if session.successful_agent_runs.get(graph.id, 0) >= config.max_agent_runs:
                return ErrorResponse(
                    message="Maximum agent runs reached for this session. Please try again later.",
                    session_id=session_id,
                )

            # Match credentials
            graph_credentials, missing_creds = await match_user_credentials_to_graph(
                user_id, graph
            )

            if missing_creds:
                return ErrorResponse(
                    message=(
                        f"Missing {len(missing_creds)} required credential(s). "
                        "Call run_agent with action='validate' to see what's needed."
                    ),
                    session_id=session_id,
                    details={"missing_credentials": missing_creds},
                )

            # Get or create library agent
            library_agent = await get_or_create_library_agent(graph, user_id)

            # Execute
            execution = await execution_utils.add_graph_execution(
                graph_id=library_agent.graph_id,
                user_id=user_id,
                inputs=inputs,
                graph_credentials_inputs=graph_credentials,
            )

            # Track successful run
            session.successful_agent_runs[library_agent.graph_id] = (
                session.successful_agent_runs.get(library_agent.graph_id, 0) + 1
            )

            library_agent_link = f"/library/agents/{library_agent.id}"
            return ExecutionStartedResponse(
                message=(
                    f"Agent execution started. "
                    f"View at {library_agent_link}. "
                    "Do not run again unless explicitly requested."
                ),
                session_id=session_id,
                execution_id=execution.id,
                graph_id=library_agent.graph_id,
                graph_name=library_agent.name,
                library_agent_id=library_agent.id,
                library_agent_link=library_agent_link,
            )

        except Exception as e:
            logger.error(f"Error running agent: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to run agent: {e!s}",
                error=str(e),
                session_id=session_id,
            )

    async def _schedule_agent(
        self,
        user_id: str,
        session: ChatSession,
        agent_slug: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Set up scheduled execution for an agent."""
        session_id = session.session_id
        inputs = kwargs.get("inputs", {})
        schedule_name = kwargs.get("schedule_name", "").strip()
        cron = kwargs.get("cron", "").strip()
        timezone = kwargs.get("timezone", "UTC").strip()

        # Validate schedule params
        if not schedule_name:
            return ErrorResponse(
                message="schedule_name is required for scheduled execution",
                session_id=session_id,
            )
        if not cron:
            return ErrorResponse(
                message="cron expression is required for scheduled execution",
                session_id=session_id,
            )

        try:
            username, agent_name = agent_slug.split("/", 1)
            graph, store_agent = await fetch_graph_from_store_slug(username, agent_name)

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_slug}' not found",
                    session_id=session_id,
                )

            # Check rate limits
            if (
                session.successful_agent_schedules.get(graph.id, 0)
                >= config.max_agent_schedules
            ):
                return ErrorResponse(
                    message="Maximum agent schedules reached for this session.",
                    session_id=session_id,
                )

            # Match credentials
            graph_credentials, missing_creds = await match_user_credentials_to_graph(
                user_id, graph
            )

            if missing_creds:
                return ErrorResponse(
                    message=(
                        f"Missing {len(missing_creds)} required credential(s). "
                        "Call run_agent with action='validate' to see what's needed."
                    ),
                    session_id=session_id,
                    details={"missing_credentials": missing_creds},
                )

            # Get or create library agent
            library_agent = await get_or_create_library_agent(graph, user_id)

            # Get user timezone
            user = await get_user_by_id(user_id)
            user_timezone = get_user_timezone_or_utc(
                user.timezone if user else timezone
            )

            # Create schedule
            result = await get_scheduler_client().add_execution_schedule(
                user_id=user_id,
                graph_id=library_agent.graph_id,
                graph_version=library_agent.graph_version,
                name=schedule_name,
                cron=cron,
                input_data=inputs,
                input_credentials=graph_credentials,
                user_timezone=user_timezone,
            )

            # Convert next_run_time to user timezone for display
            if result.next_run_time:
                result.next_run_time = convert_utc_time_to_user_timezone(
                    result.next_run_time, user_timezone
                )

            # Track successful schedule
            session.successful_agent_schedules[library_agent.graph_id] = (
                session.successful_agent_schedules.get(library_agent.graph_id, 0) + 1
            )

            library_agent_link = f"/library/agents/{library_agent.id}"
            return ExecutionStartedResponse(
                message=(
                    f"Agent scheduled successfully as '{schedule_name}'. "
                    f"View at {library_agent_link}. "
                    "Do not schedule again unless explicitly requested."
                ),
                session_id=session_id,
                execution_id=result.id,
                graph_id=library_agent.graph_id,
                graph_name=library_agent.name,
                library_agent_id=library_agent.id,
                library_agent_link=library_agent_link,
            )

        except Exception as e:
            logger.error(f"Error scheduling agent: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to schedule agent: {e!s}",
                error=str(e),
                session_id=session_id,
            )
