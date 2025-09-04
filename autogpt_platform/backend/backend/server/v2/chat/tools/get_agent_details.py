"""Tool for getting detailed information about a specific agent."""

import logging
from typing import Any

from backend.data import graph as graph_db
from backend.server.v2.store import db as store_db

from .base import BaseTool
from .models import (
    AgentDetails,
    AgentDetailsNeedLoginResponse,
    AgentDetailsResponse,
    CredentialRequirement,
    ErrorResponse,
    ExecutionOptions,
    InputField,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class GetAgentDetailsTool(BaseTool):
    """Tool for getting detailed information about an agent."""

    @property
    def name(self) -> str:
        return "get_agent_details"

    @property
    def description(self) -> str:
        return "Get detailed information about a specific agent including inputs, credentials required, and execution options."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent ID (graph ID) or marketplace slug (username/agent_name)",
                },
                "agent_version": {
                    "type": "integer",
                    "description": "Optional specific version of the agent (defaults to latest)",
                },
            },
            "required": ["agent_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session_id: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Get detailed information about an agent.

        Args:
            user_id: User ID (may be anonymous)
            session_id: Chat session ID
            agent_id: Agent ID or slug
            agent_version: Optional version number

        Returns:
            Pydantic response model

        """
        agent_id = kwargs.get("agent_id", "").strip()
        agent_version = kwargs.get("agent_version")

        if not agent_id:
            return ErrorResponse(
                message="Please provide an agent ID",
                session_id=session_id,
            )

        try:
            # First try to get as library agent if user is authenticated
            graph = None
            in_library = False
            is_marketplace = False

            if user_id and not user_id.startswith("anon_"):
                try:
                    # Try to get from user's library
                    graph = await graph_db.get_graph(
                        graph_id=agent_id,
                        version=agent_version,
                        user_id=user_id,
                        include_subgraphs=True,
                    )
                    if graph:
                        in_library = True
                        logger.info(f"Found agent {agent_id} in user library")
                except Exception as e:
                    logger.debug(f"Agent not in library: {e}")

            # If not found in library, try marketplace
            if not graph:
                # Check if it's a slug format (username/agent_name)
                if "/" in agent_id:
                    try:
                        # Get from marketplace by slug
                        store_agent = await store_db.get_store_agent_by_slug(agent_id)
                        if store_agent:
                            graph = await graph_db.get_graph(
                                graph_id=store_agent.graph_id,
                                version=agent_version or store_agent.graph_version,
                                user_id=store_agent.creator_id,  # Get with creator's permissions
                                include_subgraphs=True,
                            )
                            is_marketplace = True
                            logger.info(f"Found agent {agent_id} in marketplace")
                    except Exception as e:
                        logger.debug(f"Failed to get from marketplace: {e}")
                else:
                    # Try direct graph ID lookup (public access)
                    try:
                        graph = await graph_db.get_graph(
                            graph_id=agent_id,
                            version=agent_version,
                            user_id=None,  # Public access attempt
                            include_subgraphs=True,
                        )
                        is_marketplace = True
                    except Exception as e:
                        logger.debug(f"Failed public graph lookup: {e}")

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            # Parse input schema
            input_fields = {}
            if hasattr(graph, "input_schema") and graph.input_schema:
                if isinstance(graph.input_schema, dict):
                    properties = graph.input_schema.get("properties", {})
                    required = graph.input_schema.get("required", [])

                    input_required = []
                    input_optional = []

                    for key, schema in properties.items():
                        field = InputField(
                            name=key,
                            type=schema.get("type", "string"),
                            description=schema.get("description", ""),
                            required=key in required,
                            default=schema.get("default"),
                            options=schema.get("enum"),
                            format=schema.get("format"),
                        )

                        if key in required:
                            input_required.append(field)
                        else:
                            input_optional.append(field)

                    input_fields = {
                        "schema": graph.input_schema,
                        "required": input_required,
                        "optional": input_optional,
                    }

            # Parse credential requirements
            credentials = []
            needs_auth = False
            if (
                hasattr(graph, "credentials_input_schema")
                and graph.credentials_input_schema
            ):
                for cred_key, cred_schema in graph.credentials_input_schema.items():
                    cred_req = CredentialRequirement(
                        provider=cred_key,
                        required=True,
                    )

                    # Extract provider details if available
                    if isinstance(cred_schema, dict):
                        if "provider" in cred_schema:
                            cred_req.provider = cred_schema["provider"]
                        if "scopes" in cred_schema:
                            cred_req.scopes = cred_schema["scopes"]
                        if "type" in cred_schema:
                            cred_req.type = cred_schema["type"]
                        if "description" in cred_schema:
                            cred_req.description = cred_schema["description"]

                    credentials.append(cred_req)
                needs_auth = True

            # Determine execution options
            execution_options = ExecutionOptions(
                manual=True,  # Always support manual execution
                scheduled=True,  # Most agents support scheduling
                webhook=False,  # Check for webhook support
            )

            # Check for webhook/trigger support
            if hasattr(graph, "has_external_trigger"):
                execution_options.webhook = graph.has_external_trigger
            elif hasattr(graph, "webhook_input_node") and graph.webhook_input_node:
                execution_options.webhook = True

            # Build trigger info if available
            trigger_info = None
            if hasattr(graph, "trigger_setup_info") and graph.trigger_setup_info:
                trigger_info = {
                    "supported": True,
                    "config": (
                        graph.trigger_setup_info.dict()
                        if hasattr(graph.trigger_setup_info, "dict")
                        else graph.trigger_setup_info
                    ),
                }

            # Build stats if available
            stats = None
            if hasattr(graph, "executions_count"):
                stats = {
                    "total_runs": graph.executions_count,
                    "last_run": (
                        graph.last_execution.isoformat()
                        if hasattr(graph, "last_execution") and graph.last_execution
                        else None
                    ),
                }

            # Create agent details
            details = AgentDetails(
                id=graph.id,
                name=graph.name,
                description=graph.description,
                version=graph.version,
                is_latest=graph.is_active if hasattr(graph, "is_active") else True,
                in_library=in_library,
                is_marketplace=is_marketplace,
                inputs=input_fields,
                credentials=credentials,
                execution_options=execution_options,
                trigger_info=trigger_info,
                stats=stats,
            )

            # Check if anonymous user needs to log in
            if needs_auth and (not user_id or user_id.startswith("anon_")):
                return AgentDetailsNeedLoginResponse(
                    message="This agent requires credentials. Please sign in to set up and run this agent.",
                    session_id=session_id,
                    agent=details,
                    agent_info={
                        "agent_id": agent_id,
                        "agent_version": agent_version,
                        "name": details.name,
                        "graph_id": graph.id,
                    },
                )

            return AgentDetailsResponse(
                message=f"Agent '{graph.name}' details loaded successfully",
                session_id=session_id,
                agent=details,
                user_authenticated=not (not user_id or user_id.startswith("anon_")),
            )

        except Exception as e:
            logger.error(f"Error getting agent details: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to get agent details: {e!s}",
                error=str(e),
                session_id=session_id,
            )
