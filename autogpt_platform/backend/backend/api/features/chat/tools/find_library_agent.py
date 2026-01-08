"""Tool for searching agents in the user's library."""

import logging
from typing import Any

from backend.api.features.chat.model import ChatSession
from backend.api.features.library import db as library_db
from backend.util.exceptions import DatabaseError

from .base import BaseTool
from .models import (
    AgentCarouselResponse,
    AgentInfo,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class FindLibraryAgentTool(BaseTool):
    """Tool for searching agents in the user's library."""

    @property
    def name(self) -> str:
        return "find_library_agent"

    @property
    def description(self) -> str:
        return (
            "Search for agents in the user's library. Use this to find agents "
            "the user has already added to their library, including agents they "
            "created or added from the marketplace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query to find agents by name or description. "
                        "Use keywords for best results."
                    ),
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Search for agents in the user's library.

        Args:
            user_id: User ID (required)
            session: Chat session
            query: Search query

        Returns:
            AgentCarouselResponse: List of agents found in the library
            NoResultsResponse: No agents found
            ErrorResponse: Error message
        """
        query = kwargs.get("query", "").strip()
        session_id = session.session_id

        if not query:
            return ErrorResponse(
                message="Please provide a search query",
                session_id=session_id,
            )

        if not user_id:
            return ErrorResponse(
                message="User authentication required to search library",
                session_id=session_id,
            )

        agents = []
        try:
            logger.info(f"Searching user library for: {query}")
            library_results = await library_db.list_library_agents(
                user_id=user_id,
                search_term=query,
                page_size=10,
            )

            logger.info(
                f"Find library agents tool found {len(library_results.agents)} agents"
            )

            for agent in library_results.agents:
                agents.append(
                    AgentInfo(
                        id=agent.id,
                        name=agent.name,
                        description=agent.description or "",
                        source="library",
                        in_library=True,
                        creator=agent.creator_name,
                        status=agent.status.value,
                        can_access_graph=agent.can_access_graph,
                        has_external_trigger=agent.has_external_trigger,
                        new_output=agent.new_output,
                        graph_id=agent.graph_id,
                    ),
                )

        except DatabaseError as e:
            logger.error(f"Error searching library agents: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to search library. Please try again.",
                error=str(e),
                session_id=session_id,
            )

        if not agents:
            return NoResultsResponse(
                message=(
                    f"No agents found matching '{query}' in your library. "
                    "Try different keywords or use find_agent to search the marketplace."
                ),
                session_id=session_id,
                suggestions=[
                    "Try more general terms",
                    "Use find_agent to search the marketplace",
                    "Check your library at /library",
                ],
            )

        title = (
            f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} "
            f"in your library for '{query}'"
        )

        return AgentCarouselResponse(
            message=(
                "Found agents in the user's library. You can provide a link to "
                "view an agent at: /library/agents/{agent_id}. "
                "Use agent_output to get execution results, or run_agent to execute."
            ),
            title=title,
            agents=agents,
            count=len(agents),
            session_id=session_id,
        )
