"""Tool for getting required setup information for an agent."""

import logging
from typing import Any

from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.get_agent_details import GetAgentDetailsTool
from backend.server.v2.chat.tools.models import (
    AgentDetailsResponse,
    ErrorResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)

logger = logging.getLogger(__name__)


class GetRequiredSetupInfoTool(BaseTool):
    """Tool for getting required setup information including credentials and inputs."""

    @property
    def name(self) -> str:
        return "get_required_setup_info"

    @property
    def description(self) -> str:
        return """Check if an agent can be set up with the provided input data and credentials.
        Call this AFTER get_agent_details to validate that you have all required inputs.
        Pass the input dictionary you plan to use with run_agent or setup_agent to verify it's complete."""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "username_agent_slug": {
                    "type": "string",
                    "description": "The marketplace agent slug (e.g., 'username/agent-name' or just 'agent-name' to search)",
                },
                "inputs": {
                    "type": "object",
                    "description": "The input dictionary you plan to provide. Should contain ALL required inputs from get_agent_details",
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
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """
        Retrieve and validate the required setup information for running or configuring an agent.

        This checks all required credentials and input fields based on the agent details,
        and verifies user readiness to run the agent based on provided inputs and available credentials.

        Args:
            user_id: The authenticated user's ID (must not be None; authentication required).
            session_id: The chat session ID.
            agent_id: The agent's marketplace slug (e.g. 'username/agent-name'). Also accepts Graph ID.
            agent_version: (Optional) Specific agent/graph version (if applicable).

        Returns:
            SetupRequirementsResponse containing:
                - agent and graph info,
                - credential and input requirements,
                - user readiness and missing credentials/fields,
                - setup instructions.
        """
        assert (
            user_id is not None
        ), "GetRequiredSetupInfoTool - This should never happen user_id is None when auth is required"
        session_id = session.session_id
        # Call _execute directly since we're calling internally from another tool
        agent_details = await GetAgentDetailsTool()._execute(user_id, session, **kwargs)

        if isinstance(agent_details, ErrorResponse):
            return agent_details

        if not isinstance(agent_details, AgentDetailsResponse):
            return ErrorResponse(
                message="Failed to get agent details",
                session_id=session_id,
            )

        available_creds = await IntegrationCredentialsManager().store.get_all_creds(
            user_id
        )
        required_credentials = []

        # Check if user has credentials matching the required provider/type
        for c in agent_details.agent.credentials:
            # Check if any available credential matches this provider and type
            has_matching_cred = any(
                cred.provider == c.provider and cred.type == c.type
                for cred in available_creds
            )
            if not has_matching_cred:
                required_credentials.append(c)

        required_fields = set(agent_details.agent.inputs.get("required", []))
        provided_inputs = kwargs.get("inputs", {})
        missing_inputs = required_fields - set(provided_inputs.keys())

        missing_credentials = {c.id: c.model_dump() for c in required_credentials}

        user_readiness = UserReadiness(
            has_all_credentials=len(required_credentials) == 0,
            missing_credentials=missing_credentials,
            ready_to_run=len(missing_inputs) == 0 and len(required_credentials) == 0,
        )
        # Convert execution options to list of available modes
        exec_opts = agent_details.agent.execution_options
        execution_modes = []
        if exec_opts.manual:
            execution_modes.append("manual")
        if exec_opts.scheduled:
            execution_modes.append("scheduled")
        if exec_opts.webhook:
            execution_modes.append("webhook")

        # Convert input schema to list of input field info
        inputs_list = []
        if (
            isinstance(agent_details.agent.inputs, dict)
            and "properties" in agent_details.agent.inputs
        ):
            for field_name, field_schema in agent_details.agent.inputs[
                "properties"
            ].items():
                inputs_list.append(
                    {
                        "name": field_name,
                        "title": field_schema.get("title", field_name),
                        "type": field_schema.get("type", "string"),
                        "description": field_schema.get("description", ""),
                        "required": field_name
                        in agent_details.agent.inputs.get("required", []),
                    }
                )

        requirements = {
            "credentials": agent_details.agent.credentials,
            "inputs": inputs_list,
            "execution_modes": execution_modes,
        }
        message = ""
        if len(agent_details.agent.credentials) > 0:
            message = "The user needs to enter credentials before proceeding. Please wait until you have a message informing you that the credentials have been entered."
        elif len(inputs_list) > 0:
            message = (
                "The user needs to enter inputs before proceeding. Please wait until you have a message informing you that the inputs have been entered. The inputs are: "
                + ", ".join([input["name"] for input in inputs_list])
            )
        else:
            message = "The agent is ready to run. Please call the run_agent tool with the agent ID."

        return SetupRequirementsResponse(
            message=message,
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=agent_details.agent.id,
                agent_name=agent_details.agent.name,
                user_readiness=user_readiness,
                requirements=requirements,
            ),
            graph_id=agent_details.graph_id,
            graph_version=agent_details.graph_version,
        )
