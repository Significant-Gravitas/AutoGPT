"""Tool for running an agent manually (one-off execution)."""

import asyncio
import logging
from typing import Any

import prisma.enums

from backend.data import graph as graph_db
from backend.data.credit import get_user_credit_model
from backend.data.execution import get_graph_execution, get_graph_execution_meta
from backend.data.model import CredentialsMetaInput
from backend.executor import utils as execution_utils
from backend.server.v2.library import db as library_db

from .base import BaseTool
from .models import (
    ErrorResponse,
    ExecutionStartedResponse,
    InsufficientCreditsResponse,
    ToolResponseBase,
    ValidationErrorResponse,
)

logger = logging.getLogger(__name__)


class RunAgentTool(BaseTool):
    """Tool for executing an agent manually with immediate results."""

    @property
    def name(self) -> str:
        return "run_agent"

    @property
    def description(self) -> str:
        return "Run an agent immediately (one-off manual execution). Use this when the user wants to run an agent right now without setting up a schedule or webhook."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the agent to run (graph ID or marketplace slug)",
                },
                "agent_version": {
                    "type": "integer",
                    "description": "Optional version number of the agent",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input values for the agent execution",
                    "additionalProperties": True,
                },
                "credentials": {
                    "type": "object",
                    "description": "Credentials for the agent (if needed)",
                    "additionalProperties": True,
                },
                "wait_for_result": {
                    "type": "boolean",
                    "description": "Whether to wait for execution to complete (max 30s)",
                    "default": False,
                },
            },
            "required": ["agent_id"],
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
        agent_id = kwargs.get("agent_id", "").strip()
        agent_version = kwargs.get("agent_version")
        inputs = kwargs.get("inputs", {})
        credentials = kwargs.get("credentials", {})
        wait_for_result = kwargs.get("wait_for_result", False)

        if not agent_id:
            return ErrorResponse(
                message="Please provide an agent ID",
                session_id=session_id,
            )

        try:
            # Check credit balance
            credit_model = get_user_credit_model()
            balance = await credit_model.get_credits(user_id)

            if balance <= 0:
                return InsufficientCreditsResponse(
                    message="Insufficient credits. Please top up your account.",
                    balance=balance,
                    session_id=session_id,
                )

            # Get graph (check library first, then marketplace)
            graph = await graph_db.get_graph(
                graph_id=agent_id,
                version=agent_version,
                user_id=user_id,
                include_subgraphs=True,
            )

            if not graph:
                # Try as marketplace agent
                graph = await graph_db.get_graph(
                    graph_id=agent_id,
                    version=agent_version,
                    user_id=None,  # Public access
                    include_subgraphs=True,
                )

                # Add to library if from marketplace
                if graph:
                    logger.info(f"Adding marketplace agent {agent_id} to user library")
                    await library_db.create_library_agent(
                        graph=graph,
                        user_id=user_id,
                        create_library_agents_for_sub_graphs=True,
                    )

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            # Convert credentials to CredentialsMetaInput format
            input_credentials = {}
            for key, value in credentials.items():
                if isinstance(value, dict):
                    input_credentials[key] = CredentialsMetaInput(**value)
                else:
                    # Assume it's a credential ID
                    input_credentials[key] = CredentialsMetaInput(
                        id=value,
                        type="api_key",
                    )

            # Execute the graph
            logger.info(
                f"Executing agent {graph.name} (ID: {graph.id}) for user {user_id}"
            )

            graph_exec = await execution_utils.add_graph_execution(
                graph_id=graph.id,
                user_id=user_id,
                inputs=inputs,
                graph_version=graph.version,
                graph_credentials_inputs=input_credentials,
            )

            result = ExecutionStartedResponse(
                message=f"Agent '{graph.name}' execution started",
                execution_id=graph_exec.id,
                graph_id=graph.id,
                graph_name=graph.name,
                status="QUEUED",
                session_id=session_id,
            )

            # Optionally wait for completion (with timeout)
            if wait_for_result:
                logger.info(f"Waiting for execution {graph_exec.id} to complete...")
                start_time = asyncio.get_event_loop().time()
                timeout = 30  # 30 seconds max wait

                while asyncio.get_event_loop().time() - start_time < timeout:
                    # Get execution status
                    exec_status = await get_graph_execution_meta(user_id, graph_exec.id)

                    if exec_status and exec_status.status in [
                        prisma.enums.AgentExecutionStatus.COMPLETED,
                        prisma.enums.AgentExecutionStatus.FAILED,
                    ]:
                        result.status = exec_status.status.value
                        result.ended_at = (
                            exec_status.ended_at.isoformat()
                            if exec_status.ended_at
                            else None
                        )

                        if (
                            exec_status.status
                            == prisma.enums.AgentExecutionStatus.COMPLETED
                        ):
                            result.message = "Agent completed successfully"

                            # Try to get outputs
                            try:
                                full_exec = await get_graph_execution(
                                    user_id=user_id,
                                    execution_id=graph_exec.id,
                                    include_node_executions=True,
                                )
                                if (
                                    full_exec
                                    and hasattr(full_exec, "output_data")
                                    and full_exec.output_data
                                ):
                                    result.outputs = full_exec.output_data
                            except Exception as e:
                                logger.warning(f"Failed to get execution outputs: {e}")
                        else:
                            result.message = "Agent execution failed"
                            if (
                                hasattr(exec_status, "stats")
                                and exec_status.stats
                                and hasattr(exec_status.stats, "error")
                            ):
                                result.error = exec_status.stats.error
                        break

                    # Wait before checking again
                    await asyncio.sleep(2)
                else:
                    # Timeout reached
                    result.status = "RUNNING"
                    result.message = "Execution still running. Check status later."
                    result.timeout_reached = True

            return result

        except Exception as e:
            logger.error(f"Error executing agent: {e}", exc_info=True)

            # Check for specific error types
            if "validation" in str(e).lower():
                return ValidationErrorResponse(
                    message="Input validation failed",
                    error=str(e),
                    session_id=session_id,
                )

            return ErrorResponse(
                message=f"Failed to execute agent: {e!s}",
                session_id=session_id,
            )
