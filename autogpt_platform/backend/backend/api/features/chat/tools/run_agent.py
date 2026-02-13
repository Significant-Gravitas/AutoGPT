"""Unified tool for agent operations with automatic state detection."""

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.api.features.chat.config import ChatConfig
from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tracking import (
    track_agent_run_success,
    track_agent_scheduled,
)
from backend.api.features.library import db as library_db
from backend.data.graph import GraphModel
from backend.data.model import CredentialsMetaInput
from backend.data.user import get_user_by_id
from backend.executor import utils as execution_utils
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import DatabaseError, NotFoundError
from backend.util.timezone_utils import (
    convert_utc_time_to_user_timezone,
    get_user_timezone_or_utc,
)

from .base import BaseTool
from .helpers import get_inputs_from_schema
from .models import (
    AgentDetails,
    AgentDetailsResponse,
    ErrorResponse,
    ExecutionOptions,
    ExecutionStartedResponse,
    InputValidationErrorResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from .utils import (
    build_missing_credentials_from_graph,
    extract_credentials_from_schema,
    fetch_graph_from_store_slug,
    get_or_create_library_agent,
    match_user_credentials_to_graph,
)

logger = logging.getLogger(__name__)
config = ChatConfig()

# Constants for response messages
MSG_DO_NOT_RUN_AGAIN = "Do not run again unless explicitly requested."
MSG_DO_NOT_SCHEDULE_AGAIN = "Do not schedule again unless explicitly requested."
MSG_ASK_USER_FOR_VALUES = (
    "Ask the user what values to use, or call again with use_defaults=true "
    "to run with default values."
)
MSG_WHAT_VALUES_TO_USE = (
    "What values would you like to use, or would you like to run with defaults?"
)


class RunAgentInput(BaseModel):
    """Input parameters for the run_agent tool."""

    username_agent_slug: str = ""
    library_agent_id: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    use_defaults: bool = False
    schedule_name: str = ""
    cron: str = ""
    timezone: str = "UTC"

    @field_validator(
        "username_agent_slug",
        "library_agent_id",
        "schedule_name",
        "cron",
        "timezone",
        mode="before",
    )
    @classmethod
    def strip_strings(cls, v: Any) -> Any:
        """Strip whitespace from string fields."""
        return v.strip() if isinstance(v, str) else v


class RunAgentTool(BaseTool):
    """Unified tool for agent operations with automatic state detection.

    The tool automatically determines what to do based on provided parameters:
    1. Fetches agent details (always, silently)
    2. Checks if required inputs are provided
    3. Checks if user has required credentials
    4. Runs immediately OR schedules (if cron is provided)

    The response tells the caller what's missing or confirms execution.
    """

    @property
    def name(self) -> str:
        return "run_agent"

    @property
    def description(self) -> str:
        return """Run or schedule an agent from the marketplace or user's library.

        The tool automatically handles the setup flow:
        - Returns missing inputs if required fields are not provided
        - Returns missing credentials if user needs to configure them
        - Executes immediately if all requirements are met
        - Schedules execution if cron expression is provided

        Identify the agent using either:
        - username_agent_slug: Marketplace format 'username/agent-name'
        - library_agent_id: ID of an agent in the user's library

        For scheduled execution, provide: schedule_name, cron, and optionally timezone."""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "username_agent_slug": {
                    "type": "string",
                    "description": "Agent identifier in format 'username/agent-name'",
                },
                "library_agent_id": {
                    "type": "string",
                    "description": "Library agent ID from user's library",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input values for the agent",
                    "additionalProperties": True,
                },
                "use_defaults": {
                    "type": "boolean",
                    "description": "Set to true to run with default values (user must confirm)",
                },
                "schedule_name": {
                    "type": "string",
                    "description": "Name for scheduled execution (triggers scheduling mode)",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression (5 fields: min hour day month weekday)",
                },
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone for schedule (default: UTC)",
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        """All operations require authentication."""
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute the tool with automatic state detection."""
        params = RunAgentInput(**kwargs)
        session_id = session.session_id

        # Validate at least one identifier is provided
        has_slug = params.username_agent_slug and "/" in params.username_agent_slug
        has_library_id = bool(params.library_agent_id)

        if not has_slug and not has_library_id:
            return ErrorResponse(
                message=(
                    "Please provide either a username_agent_slug "
                    "(format 'username/agent-name') or a library_agent_id"
                ),
                session_id=session_id,
            )

        # Auth is required
        if not user_id:
            return ErrorResponse(
                message="Authentication required. Please sign in to use this tool.",
                session_id=session_id,
            )

        # Determine if this is a schedule request
        is_schedule = bool(params.schedule_name or params.cron)

        try:
            # Step 1: Fetch agent details
            graph: GraphModel | None = None
            library_agent = None

            # Priority: library_agent_id if provided
            if has_library_id:
                library_agent = await library_db.get_library_agent(
                    params.library_agent_id, user_id
                )
                if not library_agent:
                    return ErrorResponse(
                        message=f"Library agent '{params.library_agent_id}' not found",
                        session_id=session_id,
                    )
                # Get the graph from the library agent
                from backend.data.graph import get_graph

                graph = await get_graph(
                    library_agent.graph_id,
                    library_agent.graph_version,
                    user_id=user_id,
                )
            else:
                # Fetch from marketplace slug
                username, agent_name = params.username_agent_slug.split("/", 1)
                graph, _ = await fetch_graph_from_store_slug(username, agent_name)

            if not graph:
                identifier = (
                    params.library_agent_id
                    if has_library_id
                    else params.username_agent_slug
                )
                return ErrorResponse(
                    message=f"Agent '{identifier}' not found",
                    session_id=session_id,
                )

            # Step 2: Check credentials
            graph_credentials, missing_creds = await match_user_credentials_to_graph(
                user_id, graph
            )

            if missing_creds:
                # Return credentials needed response with input data info
                # The UI handles credential setup automatically, so the message
                # focuses on asking about input data
                requirements_creds_dict = build_missing_credentials_from_graph(
                    graph, None
                )
                missing_credentials_dict = build_missing_credentials_from_graph(
                    graph, graph_credentials
                )
                requirements_creds_list = list(requirements_creds_dict.values())

                return SetupRequirementsResponse(
                    message=self._build_inputs_message(graph, MSG_WHAT_VALUES_TO_USE),
                    session_id=session_id,
                    setup_info=SetupInfo(
                        agent_id=graph.id,
                        agent_name=graph.name,
                        user_readiness=UserReadiness(
                            has_all_credentials=False,
                            missing_credentials=missing_credentials_dict,
                            ready_to_run=False,
                        ),
                        requirements={
                            "credentials": requirements_creds_list,
                            "inputs": get_inputs_from_schema(graph.input_schema),
                            "execution_modes": self._get_execution_modes(graph),
                        },
                    ),
                    graph_id=graph.id,
                    graph_version=graph.version,
                )

            # Step 3: Check inputs
            # Get all available input fields from schema
            input_properties = graph.input_schema.get("properties", {})
            required_fields = set(graph.input_schema.get("required", []))
            provided_inputs = set(params.inputs.keys())
            valid_fields = set(input_properties.keys())

            # Check for unknown input fields
            unrecognized_fields = provided_inputs - valid_fields
            if unrecognized_fields:
                return InputValidationErrorResponse(
                    message=(
                        f"Unknown input field(s) provided: {', '.join(sorted(unrecognized_fields))}. "
                        f"Agent was not executed. Please use the correct field names from the schema."
                    ),
                    session_id=session_id,
                    unrecognized_fields=sorted(unrecognized_fields),
                    inputs=graph.input_schema,
                    graph_id=graph.id,
                    graph_version=graph.version,
                )

            # If agent has inputs but none were provided AND use_defaults is not set,
            # always show what's available first so user can decide
            if input_properties and not provided_inputs and not params.use_defaults:
                credentials = extract_credentials_from_schema(
                    graph.credentials_input_schema
                )
                return AgentDetailsResponse(
                    message=self._build_inputs_message(graph, MSG_ASK_USER_FOR_VALUES),
                    session_id=session_id,
                    agent=self._build_agent_details(graph, credentials),
                    user_authenticated=True,
                    graph_id=graph.id,
                    graph_version=graph.version,
                )

            # Check if required inputs are missing (and not using defaults)
            missing_inputs = required_fields - provided_inputs

            if missing_inputs and not params.use_defaults:
                # Return agent details with missing inputs info
                credentials = extract_credentials_from_schema(
                    graph.credentials_input_schema
                )
                return AgentDetailsResponse(
                    message=(
                        f"Agent '{graph.name}' is missing required inputs: "
                        f"{', '.join(missing_inputs)}. "
                        "Please provide these values to run the agent."
                    ),
                    session_id=session_id,
                    agent=self._build_agent_details(graph, credentials),
                    user_authenticated=True,
                    graph_id=graph.id,
                    graph_version=graph.version,
                )

            # Step 4: Execute or Schedule
            if is_schedule:
                return await self._schedule_agent(
                    user_id=user_id,
                    session=session,
                    graph=graph,
                    graph_credentials=graph_credentials,
                    inputs=params.inputs,
                    schedule_name=params.schedule_name,
                    cron=params.cron,
                    timezone=params.timezone,
                )
            else:
                return await self._run_agent(
                    user_id=user_id,
                    session=session,
                    graph=graph,
                    graph_credentials=graph_credentials,
                    inputs=params.inputs,
                )

        except NotFoundError as e:
            return ErrorResponse(
                message=f"Agent '{params.username_agent_slug}' not found",
                error=str(e) if str(e) else "not_found",
                session_id=session_id,
            )
        except DatabaseError as e:
            logger.error(f"Database error: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to process request: {e!s}",
                error=str(e),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error processing agent request: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to process request: {e!s}",
                error=str(e),
                session_id=session_id,
            )

    def _get_execution_modes(self, graph: GraphModel) -> list[str]:
        """Get available execution modes for the graph."""
        trigger_info = graph.trigger_setup_info
        if trigger_info is None:
            return ["manual", "scheduled"]
        return ["webhook"]

    def _build_inputs_message(
        self,
        graph: GraphModel,
        suffix: str,
    ) -> str:
        """Build a message describing available inputs for an agent."""
        inputs_list = get_inputs_from_schema(graph.input_schema)
        required_names = [i["name"] for i in inputs_list if i["required"]]
        optional_names = [i["name"] for i in inputs_list if not i["required"]]

        message_parts = [f"Agent '{graph.name}' accepts the following inputs:"]
        if required_names:
            message_parts.append(f"Required: {', '.join(required_names)}.")
        if optional_names:
            message_parts.append(
                f"Optional (have defaults): {', '.join(optional_names)}."
            )
        if not inputs_list:
            message_parts = [f"Agent '{graph.name}' has no required inputs."]
        message_parts.append(suffix)

        return " ".join(message_parts)

    def _build_agent_details(
        self,
        graph: GraphModel,
        credentials: list[CredentialsMetaInput],
    ) -> AgentDetails:
        """Build AgentDetails from a graph."""
        trigger_info = (
            graph.trigger_setup_info.model_dump() if graph.trigger_setup_info else None
        )
        return AgentDetails(
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

    async def _run_agent(
        self,
        user_id: str,
        session: ChatSession,
        graph: GraphModel,
        graph_credentials: dict[str, CredentialsMetaInput],
        inputs: dict[str, Any],
    ) -> ToolResponseBase:
        """Execute an agent immediately."""
        session_id = session.session_id

        # Check rate limits
        if session.successful_agent_runs.get(graph.id, 0) >= config.max_agent_runs:
            return ErrorResponse(
                message="Maximum agent runs reached for this session. Please try again later.",
                session_id=session_id,
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

        # Track in PostHog
        track_agent_run_success(
            user_id=user_id,
            session_id=session_id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.name,
            execution_id=execution.id,
            library_agent_id=library_agent.id,
        )

        library_agent_link = f"/library/agents/{library_agent.id}"
        return ExecutionStartedResponse(
            message=(
                f"Agent '{library_agent.name}' execution started successfully. "
                f"View at {library_agent_link}. "
                f"{MSG_DO_NOT_RUN_AGAIN}"
            ),
            session_id=session_id,
            execution_id=execution.id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.name,
            library_agent_id=library_agent.id,
            library_agent_link=library_agent_link,
        )

    async def _schedule_agent(
        self,
        user_id: str,
        session: ChatSession,
        graph: GraphModel,
        graph_credentials: dict[str, CredentialsMetaInput],
        inputs: dict[str, Any],
        schedule_name: str,
        cron: str,
        timezone: str,
    ) -> ToolResponseBase:
        """Set up scheduled execution for an agent."""
        session_id = session.session_id

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

        # Check rate limits
        if (
            session.successful_agent_schedules.get(graph.id, 0)
            >= config.max_agent_schedules
        ):
            return ErrorResponse(
                message="Maximum agent schedules reached for this session.",
                session_id=session_id,
            )

        # Get or create library agent
        library_agent = await get_or_create_library_agent(graph, user_id)

        # Get user timezone
        user = await get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else timezone)

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

        # Track in PostHog
        track_agent_scheduled(
            user_id=user_id,
            session_id=session_id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.name,
            schedule_id=result.id,
            schedule_name=schedule_name,
            cron=cron,
            library_agent_id=library_agent.id,
        )

        library_agent_link = f"/library/agents/{library_agent.id}"
        return ExecutionStartedResponse(
            message=(
                f"Agent '{library_agent.name}' scheduled successfully as '{schedule_name}'. "
                f"View at {library_agent_link}. "
                f"{MSG_DO_NOT_SCHEDULE_AGAIN}"
            ),
            session_id=session_id,
            execution_id=result.id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.name,
            library_agent_id=library_agent.id,
            library_agent_link=library_agent_link,
        )
