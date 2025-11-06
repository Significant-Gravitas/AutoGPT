"""Tool for getting detailed information about a specific agent."""

import logging
from typing import Any

from backend.data import graph as graph_db
from backend.data.model import CredentialsMetaInput
from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    AgentDetails,
    AgentDetailsResponse,
    ErrorResponse,
    ExecutionOptions,
    ToolResponseBase,
)
from backend.server.v2.store import db as store_db
from backend.util.exceptions import DatabaseError, NotFoundError

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
                "username_agent_slug": {
                    "type": "string",
                    "description": "The marketplace agent slug (e.g., 'username/agent-name')",
                },
            },
            "required": ["username_agent_slug"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Get detailed information about an agent.

        Args:
            user_id: User ID (may be anonymous)
            session_id: Chat session ID
            username_agent_slug: Agent ID or slug

        Returns:
            Pydantic response model

        """
        agent_id = kwargs.get("username_agent_slug", "").strip()
        session_id = session.session_id
        if not agent_id or "/" not in agent_id:
            return ErrorResponse(
                message="Please provide an agent ID in format 'creator/agent-name'",
                session_id=session_id,
            )

        try:
            # Always try to get from marketplace first
            graph = None
            store_agent = None

            # Check if it's a slug format (username/agent_name)
            try:
                # Parse username/agent_name from slug
                username, agent_name = agent_id.split("/", 1)
                store_agent = await store_db.get_store_agent_details(
                    username, agent_name
                )
                logger.info(f"Found agent {agent_id} in marketplace")
            except NotFoundError as e:
                logger.debug(f"Failed to get from marketplace: {e}")
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )
            except DatabaseError as e:
                logger.error(f"Failed to get from marketplace: {e}")
                return ErrorResponse(
                    message=f"Failed to get agent details: {e!s}",
                    session_id=session_id,
                )

            # If we found a store agent, get its graph
            if store_agent:
                try:
                    # Use get_available_graph to get the graph from store listing version
                    graph_meta = await store_db.get_available_graph(
                        store_agent.store_listing_version_id
                    )
                    # Now get the full graph with that ID
                    graph = await graph_db.get_graph(
                        graph_id=graph_meta.id,
                        version=graph_meta.version,
                        user_id=None,  # Public access
                        include_subgraphs=True,
                    )

                except NotFoundError as e:
                    logger.error(f"Failed to get graph for store agent: {e}")
                    return ErrorResponse(
                        message=f"Failed to get graph for store agent: {e!s}",
                        session_id=session_id,
                    )
                except DatabaseError as e:
                    logger.error(f"Failed to get graph for store agent: {e}")
                    return ErrorResponse(
                        message=f"Failed to get graph for store agent: {e!s}",
                        session_id=session_id,
                    )

            if not graph:
                return ErrorResponse(
                    message=f"Agent '{agent_id}' not found",
                    session_id=session_id,
                )

            credentials_input_schema = graph.credentials_input_schema

            # Extract credentials from the JSON schema properties
            credentials = []
            if (
                isinstance(credentials_input_schema, dict)
                and "properties" in credentials_input_schema
            ):
                for cred_name, cred_schema in credentials_input_schema[
                    "properties"
                ].items():
                    # Extract credential metadata from the schema
                    # The schema properties contain provider info and other metadata

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
                        "properties" in cred_schema
                        and "type" in cred_schema["properties"]
                    ):
                        cred_type = cred_schema["properties"]["type"].get(
                            "const", "api_key"
                        )

                    credentials.append(
                        CredentialsMetaInput(
                            id=cred_name,
                            title=cred_schema.get("title", cred_name),
                            provider=provider,  # type: ignore
                            type=cred_type,
                        )
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
                    # Currently a graph with a webhook can only be triggered by a webhook
                    manual=trigger_info is None,
                    scheduled=trigger_info is None,
                    webhook=trigger_info is not None,
                ),
                trigger_info=trigger_info,
            )

            return AgentDetailsResponse(
                message=f"Found agent '{agent_details.name}'. You do not need to run this tool again for this agent.",
                session_id=session_id,
                agent=agent_details,
                user_authenticated=user_id is not None,
                graph_id=graph.id,
                graph_version=graph.version,
            )

        except Exception as e:
            logger.error(f"Error getting agent details: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to get agent details: {e!s}",
                error=str(e),
                session_id=session_id,
            )
