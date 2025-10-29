"""Tool for running an agent manually (one-off execution)."""

import logging
from typing import Any

from backend.data.graph import get_graph
from backend.executor import utils as execution_utils
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.get_required_setup_info import (
    GetRequiredSetupInfoTool,
)
from backend.server.v2.chat.tools.models import (
    ErrorResponse,
    ExecutionStartedResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
)
from backend.server.v2.library import db as library_db
from backend.server.v2.library import model as library_model

logger = logging.getLogger(__name__)


class RunAgentTool(BaseTool):
    """Tool for executing an agent manually with immediate results."""

    @property
    def name(self) -> str:
        return "run_agent"

    @property
    def description(self) -> str:
        return """Run an agent immediately (one-off manual execution).
        IMPORTANT: Before calling this tool, you MUST first call get_agent_details to determine what inputs are required.
        The 'inputs' parameter must be a dictionary containing ALL required input values identified by get_agent_details.
        Example: If get_agent_details shows required inputs 'search_query' and 'max_results', you must pass:
        inputs={"search_query": "user's query", "max_results": 10}"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "username_agent_slug": {
                    "type": "string",
                    "description": "The ID of the agent to run (graph ID or marketplace slug)",
                },
                "inputs": {
                    "type": "object",
                    "description": 'REQUIRED: Dictionary of input values. Must include ALL required inputs from get_agent_details. Format: {"input_name": value}',
                    "additionalProperties": True,
                },
            },
            "required": ["username_agent_slug"],
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
        """Execute an agent manually.

        Args:
            user_id: Authenticated user ID
            session_id: Chat session ID
            **kwargs: Execution parameters

        Returns:
            JSON formatted execution result

        """

        assert (
            user_id is not None
        ), "User ID is required to run an agent. Superclass enforces authentication."

        username_agent_slug = kwargs.get("username_agent_slug", "").strip()
        inputs = kwargs.get("inputs", {})

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

        graph = await get_graph(username_agent_slug)

        if not graph:
            return ErrorResponse(
                message=f"Graph {username_agent_slug} not found",
                session_id=session_id,
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
            graph_name=library_agent.name,
        )
