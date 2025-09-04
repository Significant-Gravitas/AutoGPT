"""Tool for getting required setup information for an agent."""

import logging
from typing import Any

from backend.data import graph as graph_db
from backend.integrations.creds_manager import IntegrationCredentialsManager

from .base import BaseTool
from .models import (
    ErrorResponse,
    ExecutionModeInfo,
    InputField,
    SetupInfo,
    SetupRequirementInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class GetRequiredSetupInfoTool(BaseTool):
    """Tool for getting required setup information including credentials and inputs."""

    @property
    def name(self) -> str:
        return "get_required_setup_info"

    @property
    def description(self) -> str:
        return "Get information about required credentials, inputs, and configuration needed to set up an agent. Requires authentication."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The agent ID (graph ID) to get setup requirements for",
                },
                "agent_version": {
                    "type": "integer",
                    "description": "Optional specific version of the agent",
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
        """Get required setup information for an agent.

        Args:
            user_id: Authenticated user ID
            session_id: Chat session ID
            agent_id: Agent/Graph ID
            agent_version: Optional version

        Returns:
            JSON formatted setup requirements

        """
        agent_id = kwargs.get("agent_id", "").strip()
        agent_version = kwargs.get("agent_version")

        if not agent_id:
            return ErrorResponse(
                message="Please provide an agent ID",
                session_id=session_id,
            )

        try:
            # Get the graph with subgraphs for complete analysis
            graph = await graph_db.get_graph(
                graph_id=agent_id,
                version=agent_version,
                user_id=user_id,
                include_subgraphs=True,
            )

            if not graph:
                # Try to get from marketplace/public
                graph = await graph_db.get_graph(
                    graph_id=agent_id,
                    version=agent_version,
                    user_id=None,
                    include_subgraphs=True,
                )

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            setup_info = SetupInfo(
                agent_id=graph.id,
                agent_name=graph.name,
                version=graph.version,
            )

            # Get credential manager
            creds_manager = IntegrationCredentialsManager()

            # Analyze credential requirements
            if (
                hasattr(graph, "credentials_input_schema")
                and graph.credentials_input_schema
            ):
                user_credentials = {}
                try:
                    # Get user's existing credentials
                    user_creds_list = await creds_manager.list_credentials(user_id)
                    user_credentials = {c.provider: c for c in user_creds_list}
                except Exception as e:
                    logger.warning(f"Failed to get user credentials: {e}")

                for cred_key, cred_schema in graph.credentials_input_schema.items():
                    cred_req = SetupRequirementInfo(
                        key=cred_key,
                        provider=cred_key,
                        required=True,
                        user_has=False,
                    )

                    # Parse credential schema
                    if isinstance(cred_schema, dict):
                        if "provider" in cred_schema:
                            cred_req.provider = cred_schema["provider"]
                        if "type" in cred_schema:
                            cred_req.type = cred_schema["type"]  # oauth, api_key
                        if "scopes" in cred_schema:
                            cred_req.scopes = cred_schema["scopes"]
                        if "description" in cred_schema:
                            cred_req.description = cred_schema["description"]

                    # Check if user has this credential
                    provider_name = cred_req.provider
                    if provider_name in user_credentials:
                        cred_req.user_has = True
                        cred_req.credential_id = user_credentials[provider_name].id
                    else:
                        setup_info.user_readiness.missing_credentials.append(
                            provider_name
                        )

                    setup_info.requirements["credentials"].append(cred_req)

            # Analyze input requirements
            if hasattr(graph, "input_schema") and graph.input_schema:
                if isinstance(graph.input_schema, dict):
                    properties = graph.input_schema.get("properties", {})
                    required = graph.input_schema.get("required", [])

                    for key, schema in properties.items():
                        input_req = InputField(
                            name=key,
                            type=schema.get("type", "string"),
                            required=key in required,
                            description=schema.get("description", ""),
                        )

                        # Add default value if present
                        if "default" in schema:
                            input_req.default = schema["default"]

                        # Add enum values if present
                        if "enum" in schema:
                            input_req.options = schema["enum"]

                        # Add format hints
                        if "format" in schema:
                            input_req.format = schema["format"]

                        setup_info.requirements["inputs"].append(input_req)

            # Determine supported execution modes
            execution_modes = []

            # Manual execution is always supported
            execution_modes.append(
                ExecutionModeInfo(
                    type="manual",
                    description="Run the agent immediately with provided inputs",
                    supported=True,
                )
            )

            # Check for scheduled execution support
            execution_modes.append(
                ExecutionModeInfo(
                    type="scheduled",
                    description="Run the agent on a recurring schedule (cron)",
                    supported=True,
                    config_required={
                        "cron": "Cron expression (e.g., '0 9 * * 1' for Mondays at 9 AM)",
                        "timezone": "User timezone (converted to UTC)",
                    },
                )
            )

            # Check for webhook support
            webhook_supported = False
            if hasattr(graph, "has_external_trigger"):
                webhook_supported = graph.has_external_trigger
            elif hasattr(graph, "webhook_input_node") and graph.webhook_input_node:
                webhook_supported = True

            if webhook_supported:
                webhook_mode = ExecutionModeInfo(
                    type="webhook",
                    description="Trigger the agent via external webhook",
                    supported=True,
                    config_required={},
                )

                # Add trigger setup info if available
                if hasattr(graph, "trigger_setup_info") and graph.trigger_setup_info:
                    webhook_mode.trigger_info = (
                        graph.trigger_setup_info.dict()
                        if hasattr(graph.trigger_setup_info, "dict")
                        else graph.trigger_setup_info
                    )

                execution_modes.append(webhook_mode)
            else:
                execution_modes.append(
                    ExecutionModeInfo(
                        type="webhook",
                        description="Webhook triggers not supported for this agent",
                        supported=False,
                    )
                )

            setup_info.requirements["execution_modes"] = execution_modes

            # Check overall readiness
            has_all_creds = len(setup_info.user_readiness.missing_credentials) == 0
            setup_info.user_readiness.has_all_credentials = has_all_creds

            # Agent is ready if all required credentials are present
            setup_info.user_readiness.ready_to_run = has_all_creds

            # Add setup instructions
            if not setup_info.user_readiness.ready_to_run:
                instructions = []
                if setup_info.user_readiness.missing_credentials:
                    instructions.append(
                        f"Add credentials for: {', '.join(setup_info.user_readiness.missing_credentials)}",
                    )
                setup_info.setup_instructions = instructions
            else:
                setup_info.setup_instructions = ["Agent is ready to set up and run!"]

            return SetupRequirementsResponse(
                message=f"Setup requirements for '{graph.name}' retrieved successfully",
                setup_info=setup_info,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error getting setup requirements: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to get setup requirements: {e!s}",
                session_id=session_id,
            )
