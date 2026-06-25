"""Shared agent search functionality for find_agent and find_library_agent tools."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from backend.api.features.library.model import LibraryAgent
    from backend.api.features.store.model import StoreAgent, StoreAgentDetails

from backend.api.features.library.search import hybrid_search_library_agents
from backend.copilot.tracking import track_library_check_outcome
from backend.data.db_accessors import graph_db, library_db, store_db
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
    include_graph: bool = False,
) -> ToolResponseBase:
    """Search for agents in marketplace or user library."""
    if source == "marketplace":
        return await _search_marketplace(query, session_id)
    else:
        return await _search_library(query, session_id, user_id, include_graph)


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
    query: str,
    session_id: str | None,
    user_id: str | None,
    include_graph: bool = False,
) -> ToolResponseBase:
    """Search user's library agents by name/description.

    For UUID-shaped queries this also tries a direct id lookup as a
    convenience. The strict, no-fuzzy-fallback by-id path lives in
    ``lookup_library_agent_by_id`` (dispatched by the tool when ``agent_id``
    is given) — prefer that when the exact id is known.
    """
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
        # Safety net for a UUID passed as the query; the preferred by-id path
        # is the tool's explicit agent_id → lookup_library_agent_by_id.
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
                # Hide trigger agents — they aren't reusable as sub-agents
                # (parent-coupled, single-purpose). AutoPilot accesses
                # them via list_agent_triggers instead.
                is_hidden=False,
                # Load nodes so has_external_trigger / trigger_setup_info are
                # populated — lets AutoPilot recognise (and set up) webhook
                # triggers from the listing without re-reading the full graph.
                include_nodes=True,
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

    truncation_notice: str | None = None
    if include_graph and agents:
        truncation_notice = await _enrich_agents_with_graph(agents, user_id)

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

    message = (
        "Found agents in the user's library. You can provide a link to view "
        "an agent at: /library/agents/{agent_id}. Use view_agent_output to get "
        "execution results, or run_agent to execute. Let the user know we can "
        "create a custom agent for them based on their needs."
    )
    if any(a.trigger_info for a in agents):
        message += (
            "\n\nSome agents have a webhook trigger (see their "
            "`trigger_info`). To set up or activate "
            "such a trigger, call setup_agent_webhook_trigger and pass the "
            "config_schema fields as `trigger_config` — you don't need the full "
            "graph for this, and must NOT edit the trigger node's values in the "
            "graph (that changes the agent's global default for everyone)."
        )
    if truncation_notice:
        message += f"\n\nNote: {truncation_notice}"

    return AgentsFoundResponse(
        message=message,
        title=title,
        agents=agents,
        count=len(agents),
        session_id=session_id,
    )


_MAX_GRAPH_FETCHES = 10


_GRAPH_FETCH_TIMEOUT = 15  # seconds


async def _enrich_agents_with_graph(
    agents: list[AgentInfo], user_id: str
) -> str | None:
    """Fetch and attach full Graph (nodes + links) to each agent in-place.

    Only the first ``_MAX_GRAPH_FETCHES`` agents with a ``graph_id`` are
    enriched.  If some agents are skipped, a truncation notice is returned
    so the caller can surface it to the copilot.

    Graphs are fetched with ``for_export=True`` so that credentials, API keys,
    and other secrets in ``input_default`` are stripped before the data reaches
    the LLM context.

    Returns a truncation notice string when some agents were skipped, or
    ``None`` when all eligible agents were enriched.
    """
    with_graph_id = [a for a in agents if a.graph_id]
    fetchable = with_graph_id[:_MAX_GRAPH_FETCHES]
    if not fetchable:
        return None

    gdb = graph_db()

    async def _fetch(agent: AgentInfo) -> None:
        graph_id = agent.graph_id
        if not graph_id:
            return
        try:
            graph = await gdb.get_graph(
                graph_id,
                version=agent.graph_version,
                user_id=user_id,
                for_export=True,
            )
            if graph is None:
                logger.warning("Graph not found for agent %s", graph_id)
            agent.graph = graph
        except Exception as e:
            logger.warning("Failed to fetch graph for agent %s: %s", graph_id, e)

    try:
        await asyncio.wait_for(
            asyncio.gather(*[_fetch(a) for a in fetchable]),
            timeout=_GRAPH_FETCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "include_graph: timed out after %ds fetching graphs", _GRAPH_FETCH_TIMEOUT
        )

    skipped = len(with_graph_id) - len(fetchable)
    if skipped > 0:
        logger.warning(
            "include_graph: fetched graphs for %d/%d agents "
            "(_MAX_GRAPH_FETCHES=%d, %d skipped)",
            len(fetchable),
            len(with_graph_id),
            _MAX_GRAPH_FETCHES,
            skipped,
        )
        return (
            f"Graph data included for {len(fetchable)} of "
            f"{len(with_graph_id)} eligible agents (limit: {_MAX_GRAPH_FETCHES}). "
            f"To fetch graphs for remaining agents, narrow your search to a "
            f"specific agent by UUID."
        )
    return None


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
        trigger_info=agent.trigger_setup_info,
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


async def search_library_for_creation(
    goal_summary: str,
    session_id: str | None,
    user_id: str | None,
) -> ToolResponseBase:
    """Hybrid (semantic + lexical) library search used by the create-agent
    similarity gate.

    Unlike ``_search_library`` (substring), this is intended to surface
    *functionally similar* agents the user may want to reuse before
    creating a new one. The response message instructs the LLM to ask the
    user before proceeding to ``create_agent``.
    """
    if not user_id:
        return ErrorResponse(
            message="User authentication required to search library",
            session_id=session_id,
        )

    goal_summary = (goal_summary or "").strip()
    if not goal_summary:
        # Soft-fail instead of an error response so the UI doesn't render
        # "Error finding agents" and the gate still recognises this as a
        # valid call (the tool *was* invoked).
        return NoResultsResponse(
            message=(
                "No `goal_summary` was provided, so no similarity check "
                "ran. If the user is asking for a new agent, retry "
                "find_library_agent with for_creation=true and a "
                "goal_summary describing what they want. If the user has "
                "since clarified they want a new agent regardless, "
                "proceed with create_agent and pass "
                "library_check_ack=true."
            ),
            suggestions=[
                "Retry with for_creation=true and goal_summary=<user's goal>",
                "Proceed with create_agent + library_check_ack=true",
            ],
            session_id=session_id,
        )

    try:
        matches = await hybrid_search_library_agents(
            query=goal_summary, user_id=user_id
        )
    except DatabaseError as e:
        # logger.error → captured by Sentry's LoggingIntegration so a
        # flaky DB doesn't silently disable the gate.
        logger.error(f"Error during hybrid library search: {e}", exc_info=True)
        track_library_check_outcome(
            user_id=user_id, session_id=session_id, outcome="soft_failed"
        )
        return NoResultsResponse(
            message=(
                "Could not run the library similarity check (database "
                "error). Proceeding to create_agent is safe; pass "
                "library_check_ack=true to satisfy the gate."
            ),
            suggestions=["Proceed with create_agent + library_check_ack=true"],
            session_id=session_id,
        )
    except Exception as e:
        # Embedding service down / pgvector edge case — degrade gracefully
        # but log at ERROR so Sentry captures the silent feature-disable.
        logger.error(f"Hybrid library search failed unexpectedly: {e}", exc_info=True)
        track_library_check_outcome(
            user_id=user_id, session_id=session_id, outcome="soft_failed"
        )
        return NoResultsResponse(
            message=(
                "Could not run the library similarity check. Proceeding "
                "to create_agent is safe; pass library_check_ack=true to "
                "satisfy the gate."
            ),
            suggestions=["Proceed with create_agent + library_check_ack=true"],
            session_id=session_id,
        )

    if not matches:
        track_library_check_outcome(
            user_id=user_id, session_id=session_id, outcome="no_matches"
        )
        return NoResultsResponse(
            message=(
                "No functionally similar agents found in the user's library. "
                "You may proceed to create a new agent: call `create_agent` "
                "with `library_check_ack=true` to satisfy the similarity "
                "gate."
            ),
            suggestions=[
                "Proceed with create_agent (no similar library agent to reuse)",
            ],
            session_id=session_id,
        )

    agents = await _load_and_format_matched_agents(matches, user_id)

    if not agents:
        track_library_check_outcome(
            user_id=user_id, session_id=session_id, outcome="no_matches"
        )
        return NoResultsResponse(
            message=(
                "No functionally similar agents found in the user's library. "
                "You may proceed to create a new agent: call `create_agent` "
                "with `library_check_ack=true` to satisfy the similarity "
                "gate."
            ),
            suggestions=[
                "Proceed with create_agent (no similar library agent to reuse)",
            ],
            session_id=session_id,
        )

    top_score = max((a.match_score or 0.0) for a in agents)
    track_library_check_outcome(
        user_id=user_id,
        session_id=session_id,
        outcome="matches_shown",
        matches_count=len(agents),
        top_score=top_score,
    )
    return AgentsFoundResponse(
        message=(
            "Found agents in the user's library that may already match the "
            "user's goal. Present them with their `match_score` (a float in "
            "[0, 1]; format as `[N% match]` for the user) and ask whether "
            "they want to reuse one of these instead of creating a new "
            "agent. Use run_agent to execute a chosen existing agent. ONLY "
            "call `create_agent` with `library_check_ack=true` if the user "
            "explicitly chooses to build a new one anyway."
        ),
        title=(
            f"Found {len(agents)} potentially similar agent"
            f"{'s' if len(agents) != 1 else ''} in your library"
        ),
        agents=agents,
        count=len(agents),
        session_id=session_id,
    )


async def _load_and_format_matched_agents(
    matches: list[dict[str, Any]], user_id: str
) -> list[AgentInfo]:
    """Resolve hybrid-search matches to ``AgentInfo`` rows with ``match_score``
    set from the search's ``combined_score`` (pre-BM25, always in [0, 1];
    ``relevance`` is post-BM25 and can go negative on near-duplicate corpora).
    Skips matches that can no longer be loaded; propagates ``DatabaseError``."""
    lib_db = library_db()
    agents: list[AgentInfo] = []
    for match in matches:
        content_id = match.get("content_id")
        if not content_id:
            continue
        try:
            library_agent = await lib_db.get_library_agent(content_id, user_id)
        except NotFoundError:
            continue
        except DatabaseError:
            raise
        except Exception as e:
            logger.warning(
                f"Could not fetch matched library agent {content_id}: {e}",
                exc_info=True,
            )
            continue

        info = _library_agent_to_info(library_agent)
        info.match_score = match.get("combined_score") or 0.0
        agents.append(info)
    return agents


async def lookup_library_agent_by_id(
    agent_id: str,
    session_id: str | None,
    user_id: str | None,
    include_graph: bool = False,
) -> ToolResponseBase:
    """Strict direct resolution of one library agent by id.

    Resolves an exact ``library_agent_id`` or ``graph_id`` and never falls back
    to a fuzzy name search — returns ``NoResultsResponse`` when the id doesn't
    resolve. Backs ``find_library_agent``'s ``agent_id`` parameter.
    """
    if not user_id:
        return ErrorResponse(
            message="User authentication required to fetch a library agent",
            session_id=session_id,
        )

    try:
        agent = await _get_library_agent_by_id(user_id, agent_id)
    except DatabaseError as e:
        logger.error(f"Error fetching library agent {agent_id}: {e}", exc_info=True)
        return ErrorResponse(
            message="Failed to fetch the library agent. Please try again.",
            error=str(e),
            session_id=session_id,
        )

    if agent is None:
        return NoResultsResponse(
            message=(
                f"No library agent found with id '{agent_id}'. It may have been "
                "deleted or you may not have access. Retry find_library_agent "
                "with a name query, or create a custom agent."
            ),
            suggestions=[
                "Retry find_library_agent with a name/description query",
                "Check your library at /library",
            ],
            session_id=session_id,
        )

    truncation_notice: str | None = None
    if include_graph:
        truncation_notice = await _enrich_agents_with_graph([agent], user_id)

    message = (
        "Found the requested library agent. Link to it at "
        "/library/agents/{agent_id}. Use view_agent_output for execution "
        "results, or run_agent to execute it."
    )
    if truncation_notice:
        message = f"{message}\n\nNote: {truncation_notice}"

    return AgentsFoundResponse(
        message=message,
        title=f"Loaded agent '{agent.name}'",
        agents=[agent],
        count=1,
        session_id=session_id,
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
