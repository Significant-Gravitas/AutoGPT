"""Tool for running an agent manually (one-off execution)."""

import logging
from typing import Any

from backend.data.graph import get_graph
from backend.data.model import CredentialsMetaInput
from backend.executor import utils as execution_utils
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.server.v2.chat.config import ChatConfig
from backend.server.v2.chat.model import ChatSession
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
config = ChatConfig()


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
        session: ChatSession,
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

        session_id = session.session_id
        username_agent_slug = kwargs.get("username_agent_slug", "").strip()
        inputs = kwargs.get("inputs", {})

        # Call _execute directly since we're calling internally from another tool
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
                message=f"Graph information not available for {username_agent_slug}",
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
                message=f"Graph {username_agent_slug} ({response.graph_id}v{response.graph_version}) not found",
                session_id=session_id,
            )

        if graph and (
            session.successful_agent_runs.get(graph.id, 0) >= config.max_agent_runs
        ):
            return ErrorResponse(
                message="Maximum number of agent schedules reached. You can't schedule this agent again in this chat session.",
                session_id=session.session_id,
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

        # Build credentials mapping for the graph
        graph_credentials_inputs: dict[str, CredentialsMetaInput] = {}

        # Get aggregated credentials requirements from the graph
        aggregated_creds = graph.aggregate_credentials_inputs()
        logger.debug(
            f"Matching credentials for graph {graph.id}: {len(aggregated_creds)} required"
        )

        if aggregated_creds:
            # Get all available credentials for the user
            creds_manager = IntegrationCredentialsManager()
            available_creds = await creds_manager.store.get_all_creds(user_id)

            # Track unmatched credentials for error reporting
            missing_creds: list[str] = []

            # For each required credential field, find a matching user credential
            # field_info.provider is a frozenset because aggregate_credentials_inputs()
            # combines requirements from multiple nodes. A credential matches if its
            # provider is in the set of acceptable providers.
            for credential_field_name, (
                credential_requirements,
                _node_fields,
            ) in aggregated_creds.items():
                # Find first matching credential by provider and type
                matching_cred = next(
                    (
                        cred
                        for cred in available_creds
                        if cred.provider in credential_requirements.provider
                        and cred.type in credential_requirements.supported_types
                    ),
                    None,
                )

                if matching_cred:
                    # Use Pydantic validation to ensure type safety
                    try:
                        graph_credentials_inputs[credential_field_name] = (
                            CredentialsMetaInput(
                                id=matching_cred.id,
                                provider=matching_cred.provider,  # type: ignore
                                type=matching_cred.type,
                                title=matching_cred.title,
                            )
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to create CredentialsMetaInput for field '{credential_field_name}': "
                            f"provider={matching_cred.provider}, type={matching_cred.type}, "
                            f"credential_id={matching_cred.id}",
                            exc_info=True,
                        )
                        missing_creds.append(
                            f"{credential_field_name} (validation failed: {e})"
                        )
                else:
                    missing_creds.append(
                        f"{credential_field_name} "
                        f"(requires provider in {list(credential_requirements.provider)}, "
                        f"type in {list(credential_requirements.supported_types)})"
                    )

            # Fail fast if any required credentials are missing
            if missing_creds:
                logger.warning(
                    f"Cannot execute agent - missing credentials: {missing_creds}"
                )
                return ErrorResponse(
                    message=f"Cannot execute agent: missing {len(missing_creds)} required credential(s). You need to call the get_required_setup_info tool to setup the credentials."
                    f"Please set up the following credentials: {', '.join(missing_creds)}",
                    session_id=session_id,
                    details={"missing_credentials": missing_creds},
                )

            logger.info(
                f"Credential matching complete: {len(graph_credentials_inputs)}/{len(aggregated_creds)} matched"
            )

        # At this point we know the user is ready to run the agent
        # So we can execute the agent
        execution = await execution_utils.add_graph_execution(
            graph_id=library_agent.graph_id,
            user_id=user_id,
            inputs=inputs,
            graph_credentials_inputs=graph_credentials_inputs,
        )

        session.successful_agent_runs[library_agent.graph_id] = (
            session.successful_agent_runs.get(library_agent.graph_id, 0) + 1
        )

        return ExecutionStartedResponse(
            message="Agent execution successfully started. Do not run this tool again unless specifically asked to run the agent again.",
            session_id=session_id,
            execution_id=execution.id,
            graph_id=library_agent.graph_id,
            graph_name=library_agent.name,
        )
