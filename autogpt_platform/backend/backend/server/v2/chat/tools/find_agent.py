"""Tool for discovering agents from marketplace and user library."""

import logging
from typing import Any

from backend.server.v2.library import db as library_db
from backend.server.v2.store import db as store_db

from .base import BaseTool
from .models import (
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
        return "Discover agents based on capabilities and user needs. Searches both the marketplace and user's library if logged in."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what the user wants to accomplish",
                },
                "include_user_library": {
                    "type": "boolean",
                    "description": "Whether to include agents from user's library (default: true)",
                    "default": True,
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
        """Search for agents in marketplace and optionally user's library.

        Args:
            user_id: User ID (may be anonymous)
            session_id: Chat session ID
            query: Search query
            include_user_library: Whether to include library agents

        Returns:
            Pydantic response model

        """
        query = kwargs.get("query", "").strip()
        include_user_library = kwargs.get("include_user_library", True)

        if not query:
            return ErrorResponse(
                message="Please provide a search query",
                session_id=session_id,
            )

        try:
            all_agents = []

            # Search marketplace agents
            logger.info(f"Searching marketplace for: {query}")
            try:
                # Search store with query
                store_results = await store_db.search_store_agents(
                    search_query=query,
                    limit=15,  # Leave room for library agents
                )

                # Format marketplace agents
                for agent in store_results.agents:
                    all_agents.append(
                        AgentInfo(
                            id=agent.slug,  # Use slug for marketplace agents
                            name=agent.agent_name,
                            description=agent.description or "",
                            source="marketplace",
                            in_library=False,  # Will update if found in library
                            creator=agent.creator_username,
                            category=(
                                agent.categories[0] if agent.categories else "general"
                            ),
                            rating=agent.rating,
                            runs=agent.runs,
                            is_featured=agent.is_featured,
                        ),
                    )
            except Exception as e:
                logger.warning(f"Marketplace search failed: {e}")
                # Continue even if marketplace fails

            # Search user's library if authenticated
            if include_user_library and user_id and not user_id.startswith("anon_"):
                logger.info(f"Searching library for user {user_id}")
                try:
                    library_results = await library_db.list_library_agents(
                        user_id=user_id,
                        search_query=query,
                        page=1,
                        page_size=10,
                    )

                    # Track library graph IDs
                    library_graph_ids = set()

                    for agent in library_results.agents:
                        library_graph_ids.add(agent.graph_id)

                        # Check if already in results (from marketplace)
                        existing_agent = None
                        for idx, existing in enumerate(all_agents):
                            if (
                                hasattr(existing, "graph_id")
                                and existing.graph_id == agent.graph_id
                            ):
                                existing_agent = existing
                                # Update the existing agent to mark as in library
                                all_agents[idx].in_library = True
                                break

                        if not existing_agent:
                            # Add library-only agent
                            all_agents.append(
                                AgentInfo(
                                    id=agent.id,
                                    name=agent.name,
                                    description=agent.description,
                                    source="library",
                                    in_library=True,
                                    creator=agent.creator_name,
                                    status=(
                                        agent.status.value
                                        if hasattr(agent.status, "value")
                                        else str(agent.status)
                                    ),
                                    graph_id=agent.graph_id,
                                    can_access_graph=agent.can_access_graph,
                                    has_external_trigger=agent.has_external_trigger,
                                    new_output=agent.new_output,
                                ),
                            )

                    # Update marketplace agents that are in library
                    for agent in all_agents:
                        if (
                            hasattr(agent, "graph_id")
                            and agent.graph_id in library_graph_ids
                        ):
                            agent.in_library = True

                except Exception as e:
                    logger.warning(f"Library search failed: {e}")
                    # Continue with marketplace results only

            # Sort results: library first, then by relevance/rating
            all_agents.sort(
                key=lambda a: (
                    not a.in_library,  # Library agents first
                    -(a.rating or 0),  # Then by rating
                    -(a.runs or 0),  # Then by popularity
                ),
            )

            # Limit total results
            all_agents = all_agents[:20]

            if not all_agents:
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
            title = f"Found {len(all_agents)} agent{'s' if len(all_agents) != 1 else ''} for '{query}'"
            return AgentCarouselResponse(
                message=title,
                title=title,
                agents=all_agents,
                count=len(all_agents),
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error searching agents: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to search for agents. Please try again.",
                error=str(e),
                session_id=session_id,
            )
