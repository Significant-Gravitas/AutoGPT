"""Tool for discovering agents from marketplace and user library."""

import logging
from typing import Any

from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    AgentCarouselResponse,
    AgentInfo,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)
from backend.server.v2.store import db as store_db
from backend.util.exceptions import DatabaseError, NotFoundError

logger = logging.getLogger(__name__)


class FindAgentTool(BaseTool):
    """Tool for discovering agents based on user needs."""

    @property
    def name(self) -> str:
        return "find_agent"

    @property
    def description(self) -> str:
        return (
            "Discover agents from the marketplace based on capabilities and user needs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what the user wants to accomplish. Use single keywords for best results.",
                },
            },
            "required": ["query"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Search for agents in the marketplace.

        Args:
            user_id: User ID (may be anonymous)
            session_id: Chat session ID
            query: Search query

        Returns:
            AgentCarouselResponse: List of agents found in the marketplace
            NoResultsResponse: No agents found in the marketplace
            ErrorResponse: Error message
        """
        query = kwargs.get("query", "").strip()
        session_id = session.session_id
        if not query:
            return ErrorResponse(
                message="Please provide a search query",
                session_id=session_id,
            )
        agents = []
        try:
            logger.info(f"Searching marketplace for: {query}")
            store_results = await store_db.get_store_agents(
                search_query=query,
                page_size=5,
            )

            logger.info(f"Find agents tool found {len(store_results.agents)} agents")
            for agent in store_results.agents:
                agent_id = f"{agent.creator}/{agent.slug}"
                logger.info(f"Building agent ID = {agent_id}")
                agents.append(
                    AgentInfo(
                        id=agent_id,
                        name=agent.agent_name,
                        description=agent.description or "",
                        source="marketplace",
                        in_library=False,
                        creator=agent.creator,
                        category="general",
                        rating=agent.rating,
                        runs=agent.runs,
                        is_featured=False,
                    ),
                )
        except NotFoundError:
            pass
        except DatabaseError as e:
            logger.error(f"Error searching agents: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to search for agents. Please try again.",
                error=str(e),
                session_id=session_id,
            )
        if not agents:
            return NoResultsResponse(
                message=f"No agents found matching '{query}'. Try different keywords or browse the marketplace. If you have 3 consecutive find_agent tool calls results and found no agents. Please stop trying and ask the user if there is anything else you can help with.",
                session_id=session_id,
                suggestions=[
                    "Try more general terms",
                    "Browse categories in the marketplace",
                    "Check spelling",
                ],
            )

        # Return formatted carousel
        title = (
            f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} for '{query}'"
        )
        return AgentCarouselResponse(
            message="Now you have found some options for the user to choose from. Please ask the user if they would like to use any of these agents. If they do, please call the get_agent_details tool for this agent.",
            title=title,
            agents=agents,
            count=len(agents),
            session_id=session_id,
        )
