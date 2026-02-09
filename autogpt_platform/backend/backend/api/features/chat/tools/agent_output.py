"""Tool for retrieving agent execution outputs from user's library."""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, field_validator

from backend.api.features.chat.model import ChatSession
from backend.api.features.library import db as library_db
from backend.api.features.library.model import LibraryAgent
from backend.data import execution as execution_db
from backend.data.execution import ExecutionStatus, GraphExecution, GraphExecutionMeta

from .base import BaseTool
from .models import (
    AgentOutputResponse,
    ErrorResponse,
    ExecutionOutputInfo,
    NoResultsResponse,
    ToolResponseBase,
)
from .utils import fetch_graph_from_store_slug

logger = logging.getLogger(__name__)


class AgentOutputInput(BaseModel):
    """Input parameters for the agent_output tool."""

    agent_name: str = ""
    library_agent_id: str = ""
    store_slug: str = ""
    execution_id: str = ""
    run_time: str = "latest"

    @field_validator(
        "agent_name",
        "library_agent_id",
        "store_slug",
        "execution_id",
        "run_time",
        mode="before",
    )
    @classmethod
    def strip_strings(cls, v: Any) -> Any:
        """Strip whitespace from string fields."""
        return v.strip() if isinstance(v, str) else v


def parse_time_expression(
    time_expr: str | None,
) -> tuple[datetime | None, datetime | None]:
    """
    Parse time expression into datetime range (start, end).

    Supports: "latest", "yesterday", "today", "last week", "last 7 days",
    "last month", "last 30 days", ISO date "YYYY-MM-DD", ISO datetime.
    """
    if not time_expr or time_expr.lower() == "latest":
        return None, None

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    expr = time_expr.lower().strip()

    # Relative time expressions lookup
    relative_times: dict[str, tuple[datetime, datetime]] = {
        "yesterday": (today_start - timedelta(days=1), today_start),
        "today": (today_start, now),
        "last week": (now - timedelta(days=7), now),
        "last 7 days": (now - timedelta(days=7), now),
        "last month": (now - timedelta(days=30), now),
        "last 30 days": (now - timedelta(days=30), now),
    }
    if expr in relative_times:
        return relative_times[expr]

    # Try ISO date format (YYYY-MM-DD)
    date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", expr)
    if date_match:
        try:
            year, month, day = map(int, date_match.groups())
            start = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
            return start, start + timedelta(days=1)
        except ValueError:
            # Invalid date components (e.g., month=13, day=32)
            pass

    # Try ISO datetime
    try:
        parsed = datetime.fromisoformat(expr.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed - timedelta(hours=1), parsed + timedelta(hours=1)
    except ValueError:
        return None, None


class AgentOutputTool(BaseTool):
    """Tool for retrieving execution outputs from user's library agents."""

    @property
    def name(self) -> str:
        return "view_agent_output"

    @property
    def description(self) -> str:
        return """Retrieve execution outputs from agents in the user's library.

        Identify the agent using one of:
        - agent_name: Fuzzy search in user's library
        - library_agent_id: Exact library agent ID
        - store_slug: Marketplace format 'username/agent-name'

        Select which run to retrieve using:
        - execution_id: Specific execution ID
        - run_time: 'latest' (default), 'yesterday', 'last week', or ISO date 'YYYY-MM-DD'
        """

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Agent name to search for in user's library (fuzzy match)",
                },
                "library_agent_id": {
                    "type": "string",
                    "description": "Exact library agent ID",
                },
                "store_slug": {
                    "type": "string",
                    "description": "Marketplace identifier: 'username/agent-slug'",
                },
                "execution_id": {
                    "type": "string",
                    "description": "Specific execution ID to retrieve",
                },
                "run_time": {
                    "type": "string",
                    "description": (
                        "Time filter: 'latest', 'yesterday', 'last week', or 'YYYY-MM-DD'"
                    ),
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _resolve_agent(
        self,
        user_id: str,
        agent_name: str | None,
        library_agent_id: str | None,
        store_slug: str | None,
    ) -> tuple[LibraryAgent | None, str | None]:
        """
        Resolve agent from provided identifiers.
        Returns (library_agent, error_message).
        """
        # Priority 1: Exact library agent ID
        if library_agent_id:
            try:
                agent = await library_db.get_library_agent(library_agent_id, user_id)
                return agent, None
            except Exception as e:
                logger.warning(f"Failed to get library agent by ID: {e}")
                return None, f"Library agent '{library_agent_id}' not found"

        # Priority 2: Store slug (username/agent-name)
        if store_slug and "/" in store_slug:
            username, agent_slug = store_slug.split("/", 1)
            graph, _ = await fetch_graph_from_store_slug(username, agent_slug)
            if not graph:
                return None, f"Agent '{store_slug}' not found in marketplace"

            # Find in user's library by graph_id
            agent = await library_db.get_library_agent_by_graph_id(user_id, graph.id)
            if not agent:
                return (
                    None,
                    f"Agent '{store_slug}' is not in your library. "
                    "Add it first to see outputs.",
                )
            return agent, None

        # Priority 3: Fuzzy name search in library
        if agent_name:
            try:
                response = await library_db.list_library_agents(
                    user_id=user_id,
                    search_term=agent_name,
                    page_size=5,
                )
                if not response.agents:
                    return (
                        None,
                        f"No agents matching '{agent_name}' found in your library",
                    )

                # Return best match (first result from search)
                return response.agents[0], None
            except Exception as e:
                logger.error(f"Error searching library agents: {e}")
                return None, f"Error searching for agent: {e}"

        return (
            None,
            "Please specify an agent name, library_agent_id, or store_slug",
        )

    async def _get_execution(
        self,
        user_id: str,
        graph_id: str,
        execution_id: str | None,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> tuple[GraphExecution | None, list[GraphExecutionMeta], str | None]:
        """
        Fetch execution(s) based on filters.
        Returns (single_execution, available_executions_meta, error_message).
        """
        # If specific execution_id provided, fetch it directly
        if execution_id:
            execution = await execution_db.get_graph_execution(
                user_id=user_id,
                execution_id=execution_id,
                include_node_executions=False,
            )
            if not execution:
                return None, [], f"Execution '{execution_id}' not found"
            return execution, [], None

        # Get completed executions with time filters
        executions = await execution_db.get_graph_executions(
            graph_id=graph_id,
            user_id=user_id,
            statuses=[ExecutionStatus.COMPLETED],
            created_time_gte=time_start,
            created_time_lte=time_end,
            limit=10,
        )

        if not executions:
            return None, [], None  # No error, just no executions

        # If only one execution, fetch full details
        if len(executions) == 1:
            full_execution = await execution_db.get_graph_execution(
                user_id=user_id,
                execution_id=executions[0].id,
                include_node_executions=False,
            )
            return full_execution, [], None

        # Multiple executions - return latest with full details, plus list of available
        full_execution = await execution_db.get_graph_execution(
            user_id=user_id,
            execution_id=executions[0].id,
            include_node_executions=False,
        )
        return full_execution, executions, None

    def _build_response(
        self,
        agent: LibraryAgent,
        execution: GraphExecution | None,
        available_executions: list[GraphExecutionMeta],
        session_id: str | None,
    ) -> AgentOutputResponse:
        """Build the response based on execution data."""
        library_agent_link = f"/library/agents/{agent.id}"

        if not execution:
            return AgentOutputResponse(
                message=f"No completed executions found for agent '{agent.name}'",
                session_id=session_id,
                agent_name=agent.name,
                agent_id=agent.graph_id,
                library_agent_id=agent.id,
                library_agent_link=library_agent_link,
                total_executions=0,
            )

        execution_info = ExecutionOutputInfo(
            execution_id=execution.id,
            status=execution.status.value,
            started_at=execution.started_at,
            ended_at=execution.ended_at,
            outputs=dict(execution.outputs),
            inputs_summary=execution.inputs if execution.inputs else None,
        )

        available_list = None
        if len(available_executions) > 1:
            available_list = [
                {
                    "id": e.id,
                    "status": e.status.value,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                }
                for e in available_executions[:5]
            ]

        message = f"Found execution outputs for agent '{agent.name}'"
        if len(available_executions) > 1:
            message += (
                f". Showing latest of {len(available_executions)} matching executions."
            )

        return AgentOutputResponse(
            message=message,
            session_id=session_id,
            agent_name=agent.name,
            agent_id=agent.graph_id,
            library_agent_id=agent.id,
            library_agent_link=library_agent_link,
            execution=execution_info,
            available_executions=available_list,
            total_executions=len(available_executions) if available_executions else 1,
        )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute the agent_output tool."""
        session_id = session.session_id

        # Parse and validate input
        try:
            input_data = AgentOutputInput(**kwargs)
        except Exception as e:
            logger.error(f"Invalid input: {e}")
            return ErrorResponse(
                message="Invalid input parameters",
                error=str(e),
                session_id=session_id,
            )

        # Ensure user_id is present (should be guaranteed by requires_auth)
        if not user_id:
            return ErrorResponse(
                message="User authentication required",
                session_id=session_id,
            )

        # Check if at least one identifier is provided
        if not any(
            [
                input_data.agent_name,
                input_data.library_agent_id,
                input_data.store_slug,
                input_data.execution_id,
            ]
        ):
            return ErrorResponse(
                message=(
                    "Please specify at least one of: agent_name, "
                    "library_agent_id, store_slug, or execution_id"
                ),
                session_id=session_id,
            )

        # If only execution_id provided, we need to find the agent differently
        if (
            input_data.execution_id
            and not input_data.agent_name
            and not input_data.library_agent_id
            and not input_data.store_slug
        ):
            # Fetch execution directly to get graph_id
            execution = await execution_db.get_graph_execution(
                user_id=user_id,
                execution_id=input_data.execution_id,
                include_node_executions=False,
            )
            if not execution:
                return ErrorResponse(
                    message=f"Execution '{input_data.execution_id}' not found",
                    session_id=session_id,
                )

            # Find library agent by graph_id
            agent = await library_db.get_library_agent_by_graph_id(
                user_id, execution.graph_id
            )
            if not agent:
                return NoResultsResponse(
                    message=(
                        f"Execution found but agent not in your library. "
                        f"Graph ID: {execution.graph_id}"
                    ),
                    session_id=session_id,
                    suggestions=["Add the agent to your library to see more details"],
                )

            return self._build_response(agent, execution, [], session_id)

        # Resolve agent from identifiers
        agent, error = await self._resolve_agent(
            user_id=user_id,
            agent_name=input_data.agent_name or None,
            library_agent_id=input_data.library_agent_id or None,
            store_slug=input_data.store_slug or None,
        )

        if error or not agent:
            return NoResultsResponse(
                message=error or "Agent not found",
                session_id=session_id,
                suggestions=[
                    "Check the agent name or ID",
                    "Make sure the agent is in your library",
                ],
            )

        # Parse time expression
        time_start, time_end = parse_time_expression(input_data.run_time)

        # Fetch execution(s)
        execution, available_executions, exec_error = await self._get_execution(
            user_id=user_id,
            graph_id=agent.graph_id,
            execution_id=input_data.execution_id or None,
            time_start=time_start,
            time_end=time_end,
        )

        if exec_error:
            return ErrorResponse(
                message=exec_error,
                session_id=session_id,
            )

        return self._build_response(agent, execution, available_executions, session_id)
