"""Shared agent search functionality for find_agent and find_library_agent tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.api.features.library.model import LibraryAgent
    from backend.api.features.store.model import StoreAgent, StoreAgentDetails

from backend.data.db_accessors import library_db, store_db
from backend.util.exceptions import DatabaseError, NotFoundError

from .models import (
    AgentInfo,
    AgentsFoundResponse,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)
from .utils import is_creator_slug, is_uuid

logger = logging.getLogger(__name__)

SearchSource = Literal["marketplace", "library"]

# Keywords that should be treated as "list all" rather than a literal search
_LIST_ALL_KEYWORDS = frozenset({"all", "*", "everything", "any", ""})


async def search_agents(
    query: str,
    source: SearchSource,
    session_id: str | None = None,
    user_id: str | None = None,
) -> ToolResponseBase:
    """Search for agents in marketplace or user library."""
    if source == "marketplace":
        return await _search_marketplace(query, session_id)
    else:
        return await _search_library(query, session_id, user_id)


async def _search_marketplace(query: str, session_id: str | None) -> ToolResponseBase:
    """Search marketplace agents, with direct creator/slug lookup fallback."""
    query = query.strip()
    if not query:
        return ErrorResponse(
            message="Please provide a search query", session_id=session_id
        )

    agents: list[AgentInfo] = []
    try:
        # Direct lookup if query matches "creator/slug" pattern
        if is_creator_slug(query):
            logger.info(f"Query looks like creator/slug, trying direct lookup: {query}")
            creator, slug = query.split("/", 1)
            agent_info = await _get_marketplace_agent_by_slug(creator, slug)
            if agent_info:
                agents.append(agent_info)

        if not agents:
            logger.info(f"Searching marketplace for: {query}")
            results = await store_db().get_store_agents(search_query=query, page_size=5)
            for agent in results.agents:
                agents.append(_marketplace_agent_to_info(agent))
    except NotFoundError:
        pass
    except DatabaseError as e:
        logger.error(f"Error searching marketplace: {e}", exc_info=True)
        return ErrorResponse(
            message="Failed to search marketplace. Please try again.",
            error=str(e),
            session_id=session_id,
        )

    if not agents:
        return NoResultsResponse(
            message=(
                f"No agents found matching '{query}'. Let the user know they can "
                "try different keywords or browse the marketplace. Also let them "
                "know you can create a custom agent for them based on their needs."
            ),
            suggestions=[
                "Try more general terms",
                "Browse categories in the marketplace",
                "Check spelling",
            ],
            session_id=session_id,
        )

    return AgentsFoundResponse(
        message=(
            "Now you have found some options for the user to choose from. "
            "You can add a link to a recommended agent at: /marketplace/agent/agent_id "
            "Please ask the user if they would like to use any of these agents. "
            "Let the user know we can create a custom agent for them based on their needs."
        ),
        title=f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} for '{query}'",
        agents=agents,
        count=len(agents),
        session_id=session_id,
    )


async def _search_library(
    query: str, session_id: str | None, user_id: str | None
) -> ToolResponseBase:
    """Search user's library agents, with direct UUID lookup fallback."""
    if not user_id:
        return ErrorResponse(
            message="User authentication required to search library",
            session_id=session_id,
        )

    query = query.strip()
    # Normalize list-all keywords to empty string
    if query.lower() in _LIST_ALL_KEYWORDS:
        query = ""

    agents: list[AgentInfo] = []
    try:
        if is_uuid(query):
            logger.info(f"Query looks like UUID, trying direct lookup: {query}")
            agent = await _get_library_agent_by_id(user_id, query)
            if agent:
                agents.append(agent)

        if not agents:
            logger.info(
                f"{'Listing all agents in' if not query else 'Searching'} "
                f"user library{'' if not query else f' for: {query}'}"
            )
            results = await library_db().list_library_agents(
                user_id=user_id,
                search_term=query or None,
                page_size=50 if not query else 10,
            )
            for agent in results.agents:
                agents.append(_library_agent_to_info(agent))
    except NotFoundError:
        pass
    except DatabaseError as e:
        logger.error(f"Error searching library: {e}", exc_info=True)
        return ErrorResponse(
            message="Failed to search library. Please try again.",
            error=str(e),
            session_id=session_id,
        )

    if not agents:
        if not query:
            return NoResultsResponse(
                message=(
                    "Your library is empty. Let the user know they can browse the "
                    "marketplace to find agents, or you can create a custom agent "
                    "for them based on their needs."
                ),
                suggestions=[
                    "Browse the marketplace to find and add agents",
                    "Use find_agent to search the marketplace",
                ],
                session_id=session_id,
            )
        return NoResultsResponse(
            message=(
                f"No agents matching '{query}' found in your library. Let the "
                "user know you can create a custom agent for them based on "
                "their needs."
            ),
            suggestions=[
                "Try different keywords",
                "Use find_agent to search the marketplace",
                "Check your library at /library",
            ],
            session_id=session_id,
        )

    if not query:
        title = f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} in your library"
    else:
        title = f"Found {len(agents)} agent{'s' if len(agents) != 1 else ''} in your library for '{query}'"

    return AgentsFoundResponse(
        message=(
            "Found agents in the user's library. You can provide a link to view "
            "an agent at: /library/agents/{agent_id}. Use agent_output to get "
            "execution results, or run_agent to execute. Let the user know we can "
            "create a custom agent for them based on their needs."
        ),
        title=title,
        agents=agents,
        count=len(agents),
        session_id=session_id,
    )


def _marketplace_agent_to_info(agent: StoreAgent | StoreAgentDetails) -> AgentInfo:
    """Convert a marketplace agent (StoreAgent or StoreAgentDetails) to an AgentInfo."""
    return AgentInfo(
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
        graph_version=agent.graph_version,
        input_schema=agent.input_schema,
        output_schema=agent.output_schema,
    )


async def _get_marketplace_agent_by_slug(creator: str, slug: str) -> AgentInfo | None:
    """Fetch a marketplace agent by creator/slug identifier."""
    try:
        details = await store_db().get_store_agent_details(creator, slug)
        return _marketplace_agent_to_info(details)
    except NotFoundError:
        pass
    except DatabaseError:
        raise
    except Exception as e:
        logger.warning(
            f"Could not fetch marketplace agent {creator}/{slug}: {e}",
            exc_info=True,
        )
    return None


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
            return _library_agent_to_info(agent)
    except NotFoundError:
        pass
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
            return _library_agent_to_info(agent)
    except NotFoundError:
        pass
    except DatabaseError:
        raise
    except Exception as e:
        logger.warning(
            f"Could not fetch library agent by library_id {agent_id}: {e}",
            exc_info=True,
        )

    return None
