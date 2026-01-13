"""Shared agent search functionality for find_agent and find_library_agent tools."""

import logging
from typing import Literal

from backend.api.features.library import db as library_db
from backend.api.features.store import db as store_db
from backend.util.exceptions import DatabaseError, NotFoundError

from .models import (
    AgentInfo,
    AgentsFoundResponse,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

SearchSource = Literal["marketplace", "library"]


async def search_agents(
    query: str,
    source: SearchSource,
    session_id: str | None,
    user_id: str | None = None,
) -> ToolResponseBase:
    """
    Search for agents in marketplace or user library.

    Args:
        query: Search query string
        source: "marketplace" or "library"
        session_id: Chat session ID
        user_id: User ID (required for library search)

    Returns:
        AgentsFoundResponse, NoResultsResponse, or ErrorResponse
    """
    if not query:
        return ErrorResponse(
            message="Please provide a search query", session_id=session_id
        )

    if source == "library" and not user_id:
        return ErrorResponse(
            message="User authentication required to search library",
            session_id=session_id,
        )

    agents: list[AgentInfo] = []
    try:
        if source == "marketplace":
            logger.info(f"Searching marketplace for: {query}")
            results = await store_db.get_store_agents(search_query=query, page_size=5)
            for agent in results.agents:
                agents.append(
                    AgentInfo(
                        id=f"{agent.creator}/{agent.slug}",
                        name=agent.agent_name,
                        description=agent.description or "",
                        source="marketplace",
                        in_library=False,
                        creator=agent.creator,
                        category="general",
                        rating=agent.rating,
                        runs=agent.runs,
                        is_featured=False,
                    )
                )
        else:  # library
            logger.info(f"Searching user library for: {query}")
            results = await library_db.list_library_agents(
                user_id=user_id,  # type: ignore[arg-type]
                search_term=query,
                page_size=10,
            )
            for agent in results.agents:
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
                    )
                )
        logger.info(f"Found {len(agents)} agents in {source}")
    except NotFoundError:
        pass
    except DatabaseError as e:
        logger.error(f"Error searching {source}: {e}", exc_info=True)
        return ErrorResponse(
            message=f"Failed to search {source}. Please try again.",
            error=str(e),
            session_id=session_id,
        )

    if not agents:
        suggestions = (
            [
                "Try more general terms",
                "Browse categories in the marketplace",
                "Check spelling",
            ]
            if source == "marketplace"
            else [
                "Try different keywords",
                "Use find_agent to search the marketplace",
                "Check your library at /library",
            ]
        )
        no_results_msg = (
            f"No agents found matching '{query}'. Try different keywords or browse the marketplace."
            if source == "marketplace"
            else f"No agents matching '{query}' found in your library."
        )
        return NoResultsResponse(
            message=no_results_msg, session_id=session_id, suggestions=suggestions
        )

    title = f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} "
    title += (
        f"for '{query}'"
        if source == "marketplace"
        else f"in your library for '{query}'"
    )

    message = (
        "Now you have found some options for the user to choose from. "
        "You can add a link to a recommended agent at: /marketplace/agent/agent_id "
        "Please ask the user if they would like to use any of these agents."
        if source == "marketplace"
        else "Found agents in the user's library. You can provide a link to view an agent at: "
        "/library/agents/{agent_id}. Use agent_output to get execution results, or run_agent to execute."
    )

    return AgentsFoundResponse(
        message=message,
        title=title,
        agents=agents,
        count=len(agents),
        session_id=session_id,
    )
