"""Shared agent search functionality for find_agent and find_library_agent tools."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.api.features.library.model import LibraryAgent

from backend.data.db_accessors import library_db, store_db
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

_UUID_PATTERN = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$",
    re.IGNORECASE,
)

# Keywords that should be treated as "list all" rather than a literal search
_LIST_ALL_KEYWORDS = frozenset({"all", "*", "everything", "any", ""})


async def search_agents(
    query: str,
    source: SearchSource,
    session_id: str | None = None,
    user_id: str | None = None,
) -> ToolResponseBase:
    """
    Search for agents in marketplace or user library.

    For library searches, keywords like "all", "*", "everything", or an empty
    query will list all agents without filtering.

    Args:
        query: Search query string. Special keywords list all library agents.
        source: "marketplace" or "library"
        session_id: Chat session ID
        user_id: User ID (required for library search)

    Returns:
        AgentsFoundResponse, NoResultsResponse, or ErrorResponse
    """
    # Normalize list-all keywords to empty string for library searches
    if source == "library" and query.lower().strip() in _LIST_ALL_KEYWORDS:
        query = ""

    if source == "marketplace" and not query:
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
            results = await store_db().get_store_agents(search_query=query, page_size=5)
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
        else:
            if _is_uuid(query):
                logger.info(f"Query looks like UUID, trying direct lookup: {query}")
                agent = await _get_library_agent_by_id(user_id, query)  # type: ignore[arg-type]
                if agent:
                    agents.append(agent)
                    logger.info(f"Found agent by direct ID lookup: {agent.name}")

            if not agents:
                search_term = query or None
                logger.info(
                    f"{'Listing all agents in' if not query else 'Searching'} "
                    f"user library{'' if not query else f' for: {query}'}"
                )
                results = await library_db().list_library_agents(
                    user_id=user_id,  # type: ignore[arg-type]
                    search_term=search_term,
                    page_size=50 if not query else 10,
                )
                for agent in results.agents:
                    agents.append(_library_agent_to_info(agent))
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
        if source == "marketplace":
            suggestions = [
                "Try more general terms",
                "Browse categories in the marketplace",
                "Check spelling",
            ]
            no_results_msg = (
                f"No agents found matching '{query}'. Let the user know they can "
                "try different keywords or browse the marketplace. Also let them "
                "know you can create a custom agent for them based on their needs."
            )
        elif not query:
            # User asked to list all but library is empty
            suggestions = [
                "Browse the marketplace to find and add agents",
                "Use find_agent to search the marketplace",
            ]
            no_results_msg = (
                "Your library is empty. Let the user know they can browse the "
                "marketplace to find agents, or you can create a custom agent "
                "for them based on their needs."
            )
        else:
            suggestions = [
                "Try different keywords",
                "Use find_agent to search the marketplace",
                "Check your library at /library",
            ]
            no_results_msg = (
                f"No agents matching '{query}' found in your library. Let the "
                "user know you can create a custom agent for them based on "
                "their needs."
            )
        return NoResultsResponse(
            message=no_results_msg, session_id=session_id, suggestions=suggestions
        )

    if source == "marketplace":
        title = (
            f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} for '{query}'"
        )
    elif not query:
        title = f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} in your library"
    else:
        title = f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} in your library for '{query}'"

    message = (
        "Now you have found some options for the user to choose from. "
        "You can add a link to a recommended agent at: /marketplace/agent/agent_id "
        "Please ask the user if they would like to use any of these agents. "
        "Let the user know we can create a custom agent for them based on their needs."
        if source == "marketplace"
        else "Found agents in the user's library. You can provide a link to view "
        "an agent at: /library/agents/{agent_id}. Use agent_output to get "
        "execution results, or run_agent to execute. Let the user know we can "
        "create a custom agent for them based on their needs."
    )

    return AgentsFoundResponse(
        message=message,
        title=title,
        agents=agents,
        count=len(agents),
        session_id=session_id,
    )


def _is_uuid(text: str) -> bool:
    """Check if text is a valid UUID v4."""
    return bool(_UUID_PATTERN.match(text.strip()))


def _library_agent_to_info(agent: LibraryAgent) -> AgentInfo:
    """Convert a library agent model to an AgentInfo."""
    return AgentInfo(
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


async def _get_library_agent_by_id(user_id: str, agent_id: str) -> AgentInfo | None:
    """Fetch a library agent by ID (library agent ID or graph_id).

    Tries multiple lookup strategies:
    1. First by graph_id (AgentGraph primary key)
    2. Then by library agent ID (LibraryAgent primary key)
    """
    lib_db = library_db()

    try:
        agent = await lib_db.get_library_agent_by_graph_id(user_id, agent_id)
        if agent:
            logger.debug(f"Found library agent by graph_id: {agent.name}")
            return _library_agent_to_info(agent)
    except NotFoundError:
        logger.debug(f"Library agent not found by graph_id: {agent_id}")
    except DatabaseError:
        raise
    except Exception as e:
        logger.warning(
            f"Could not fetch library agent by graph_id {agent_id}: {e}",
            exc_info=True,
        )

    try:
        agent = await lib_db.get_library_agent(agent_id, user_id)
        if agent:
            logger.debug(f"Found library agent by library_id: {agent.name}")
            return _library_agent_to_info(agent)
    except NotFoundError:
        logger.debug(f"Library agent not found by library_id: {agent_id}")
    except DatabaseError:
        raise
    except Exception as e:
        logger.warning(
            f"Could not fetch library agent by library_id {agent_id}: {e}",
            exc_info=True,
        )

    return None
