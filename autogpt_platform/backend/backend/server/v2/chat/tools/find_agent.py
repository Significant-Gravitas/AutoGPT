"""Tool for discovering agents from marketplace and user library."""

import logging
from typing import Any

from backend.server.v2.store import db as store_db

from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    AgentCarouselResponse,
    AgentInfo,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)

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
                    "description": "Search query describing what the user wants to accomplish",
                },
            },
            "required": ["query"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session_id: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Search for agents in the marketplace.

        Args:
            user_id: User ID (may be anonymous)
            session_id: Chat session ID
            query: Search query

        Returns:
            Pydantic response model

        """
        query = kwargs.get("query", "").strip()

        if not query:
            return ErrorResponse(
                message="Please provide a search query",
                session_id=session_id,
            )

        try:
            # Search marketplace agents
            logger.info(f"Searching marketplace for: {query}")

            # Use search_store_agents for vector search - limit to 5 agents
            store_results = await store_db.search_store_agents(
                search_query=query,
                limit=5,
            )
            logger.info(f"Find agents tool found {len(store_results.agents)} agents")
            # Format marketplace agents
            agents = []
            for agent in store_results.agents:
                # Build the full agent ID as username/slug for marketplace lookup
                # Ensure we're using the slug from the agent, not any other ID field
                agent_slug = agent.slug
                agent_creator = agent.creator
                agent_id = f"{agent_creator}/{agent_slug}"
                logger.info(f"Building agent ID: creator={agent_creator}, slug={agent_slug}, full_id={agent_id}")
                agents.append(
                    AgentInfo(
                        id=agent_id,  # Use username/slug format for marketplace agents
                        name=agent.agent_name,
                        description=agent.description or "",
                        source="marketplace",
                        in_library=False,
                        creator=agent.creator,
                        category="general",  # StoreAgent doesn't have categories
                        rating=agent.rating,
                        runs=agent.runs,
                        is_featured=False,  # StoreAgent doesn't have is_featured
                    ),
                )

            if not agents:
                return NoResultsResponse(
                    message=f"No agents found matching '{query}'. Try different keywords or browse the marketplace.",
                    session_id=session_id,
                    suggestions=[
                        "Try more general terms",
                        "Browse categories in the marketplace",
                        "Check spelling",
                    ],
                )

            # Return formatted carousel
            title = f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} for '{query}'"
            return AgentCarouselResponse(
                message=title,
                title=title,
                agents=agents,
                count=len(agents),
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error searching agents: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to search for agents. Please try again.",
                error=str(e),
                session_id=session_id,
            )

if __name__ == "__main__":
    import asyncio
    import prisma

    find_agent_tool = FindAgentTool()
    print(find_agent_tool.parameters)


    async def main():
        await prisma.Prisma().connect()
        agents = await find_agent_tool.execute(query="Linkedin", user_id="user", session_id="session")
        print(agents)
        await prisma.Prisma().disconnect()
    asyncio.run(main())