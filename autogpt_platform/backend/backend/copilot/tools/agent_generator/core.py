"""Core agent generation functions."""

import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict

from backend.data.db_accessors import graph_db, library_db, store_db
from backend.data.graph import Graph, Link, Node
from backend.util.exceptions import DatabaseError, NotFoundError

from .service import (
    customize_template_external,
    decompose_goal_external,
    generate_agent_external,
    generate_agent_patch_external,
    is_external_service_configured,
)

logger = logging.getLogger(__name__)


class ExecutionSummary(TypedDict):
    """Summary of a single execution for quality assessment."""

    status: str
    correctness_score: NotRequired[float]
    activity_summary: NotRequired[str]


class LibraryAgentSummary(TypedDict):
    """Summary of a library agent for sub-agent composition.

    Includes recent executions to help the LLM decide whether to use this agent.
    Each execution shows status, correctness_score (0-1), and activity_summary.
    """

    graph_id: str
    graph_version: int
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    recent_executions: NotRequired[list[ExecutionSummary]]


class MarketplaceAgentSummary(TypedDict):
    """Summary of a marketplace agent for sub-agent composition."""

    name: str
    description: str
    sub_heading: str
    creator: str
    is_marketplace_agent: bool


class DecompositionStep(TypedDict, total=False):
    """A single step in decomposed instructions."""

    description: str
    action: str
    block_name: str
    tool: str
    name: str


class DecompositionResult(TypedDict, total=False):
    """Result from decompose_goal - can be instructions, questions, or error."""

    type: str
    steps: list[DecompositionStep]
    questions: list[dict[str, Any]]
    error: str
    error_type: str


AgentSummary = LibraryAgentSummary | MarketplaceAgentSummary | dict[str, Any]


def _to_dict_list(
    agents: Sequence[AgentSummary] | Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert typed agent summaries to plain dicts for external service calls."""
    if agents is None:
        return None
    return [dict(a) for a in agents]


class AgentGeneratorNotConfiguredError(Exception):
    """Raised when the external Agent Generator service is not configured."""

    pass


def _check_service_configured() -> None:
    """Check if the external Agent Generator service is configured.

    Raises:
        AgentGeneratorNotConfiguredError: If the service is not configured.
    """
    if not is_external_service_configured():
        raise AgentGeneratorNotConfiguredError(
            "Agent Generator service is not configured. "
            "Set AGENTGENERATOR_HOST environment variable to enable agent generation."
        )


_UUID_PATTERN = re.compile(
    r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}",
    re.IGNORECASE,
)


def extract_uuids_from_text(text: str) -> list[str]:
    """Extract all UUID v4 strings from text.

    Args:
        text: Text that may contain UUIDs (e.g., user's goal description)

    Returns:
        List of unique UUIDs found in the text (lowercase)
    """
    matches = _UUID_PATTERN.findall(text)
    return list({m.lower() for m in matches})


async def get_library_agent_by_id(
    user_id: str, agent_id: str
) -> LibraryAgentSummary | None:
    """Fetch a specific library agent by its ID (library agent ID or graph_id).

    This function tries multiple lookup strategies:
    1. First tries to find by graph_id (AgentGraph primary key)
    2. If not found, tries to find by library agent ID (LibraryAgent primary key)

    This handles both cases:
    - User provides graph_id (e.g., from AgentExecutorBlock)
    - User provides library agent ID (e.g., from library URL)

    Args:
        user_id: The user ID
        agent_id: The ID to look up (can be graph_id or library agent ID)

    Returns:
        LibraryAgentSummary if found, None otherwise
    """
    db = library_db()
    try:
        agent = await db.get_library_agent_by_graph_id(user_id, agent_id)
        if agent:
            logger.debug(f"Found library agent by graph_id: {agent.name}")
            return LibraryAgentSummary(
                graph_id=agent.graph_id,
                graph_version=agent.graph_version,
                name=agent.name,
                description=agent.description,
                input_schema=agent.input_schema,
                output_schema=agent.output_schema,
            )
    except DatabaseError:
        raise
    except Exception as e:
        logger.debug(f"Could not fetch library agent by graph_id {agent_id}: {e}")

    try:
        agent = await db.get_library_agent(agent_id, user_id)
        if agent:
            logger.debug(f"Found library agent by library_id: {agent.name}")
            return LibraryAgentSummary(
                graph_id=agent.graph_id,
                graph_version=agent.graph_version,
                name=agent.name,
                description=agent.description,
                input_schema=agent.input_schema,
                output_schema=agent.output_schema,
            )
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


get_library_agent_by_graph_id = get_library_agent_by_id


async def get_library_agents_by_ids(
    user_id: str,
    agent_ids: list[str],
) -> list[LibraryAgentSummary]:
    """Fetch multiple library agents by their IDs.

    Args:
        user_id: The user ID
        agent_ids: List of agent IDs (can be graph_ids or library agent IDs)

    Returns:
        List of LibraryAgentSummary for found agents (silently skips not found)
    """
    agents: list[LibraryAgentSummary] = []
    for agent_id in agent_ids:
        try:
            agent = await get_library_agent_by_id(user_id, agent_id)
            if agent:
                agents.append(agent)
                logger.debug(f"Fetched library agent by ID: {agent['name']}")
            else:
                logger.warning(f"Library agent not found for ID: {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to fetch library agent {agent_id}: {e}")
            continue

    logger.info(f"Fetched {len(agents)}/{len(agent_ids)} library agents by ID")
    return agents


async def get_library_agents_for_generation(
    user_id: str,
    search_query: str | None = None,
    exclude_graph_id: str | None = None,
    max_results: int = 15,
) -> list[LibraryAgentSummary]:
    """Fetch user's library agents formatted for Agent Generator.

    Uses search-based fetching to return relevant agents instead of all agents.
    This is more scalable for users with large libraries.

    Includes recent_executions list to help the LLM assess agent quality:
    - Each execution has status, correctness_score (0-1), and activity_summary
    - This gives the LLM concrete examples of recent performance

    Args:
        user_id: The user ID
        search_query: Optional search term to find relevant agents (user's goal/description)
        exclude_graph_id: Optional graph ID to exclude (prevents circular references)
        max_results: Maximum number of agents to return (default 15)

    Returns:
        List of LibraryAgentSummary with schemas and recent executions for sub-agent composition
    """
    search_term = search_query.strip() if search_query else None
    if search_term and len(search_term) > 100:
        raise ValueError(
            f"Search query is too long ({len(search_term)} chars, max 100). "
            f"Please use a shorter, more specific search term."
        )

    try:
        response = await library_db().list_library_agents(
            user_id=user_id,
            search_term=search_term,
            page=1,
            page_size=max_results,
            include_executions=True,
        )

        results: list[LibraryAgentSummary] = []
        for agent in response.agents:
            if exclude_graph_id is not None and agent.graph_id == exclude_graph_id:
                continue

            summary = LibraryAgentSummary(
                graph_id=agent.graph_id,
                graph_version=agent.graph_version,
                name=agent.name,
                description=agent.description,
                input_schema=agent.input_schema,
                output_schema=agent.output_schema,
            )
            if agent.recent_executions:
                exec_summaries: list[ExecutionSummary] = []
                for ex in agent.recent_executions:
                    exec_sum = ExecutionSummary(status=ex.status)
                    if ex.correctness_score is not None:
                        exec_sum["correctness_score"] = ex.correctness_score
                    if ex.activity_summary:
                        exec_sum["activity_summary"] = ex.activity_summary
                    exec_summaries.append(exec_sum)
                summary["recent_executions"] = exec_summaries
            results.append(summary)
        return results
    except DatabaseError:
        raise
    except Exception as e:
        logger.warning(f"Failed to fetch library agents: {e}")
        return []


async def search_marketplace_agents_for_generation(
    search_query: str,
    max_results: int = 10,
) -> list[LibraryAgentSummary]:
    """Search marketplace agents formatted for Agent Generator.

    Fetches marketplace agents and their full schemas so they can be used
    as sub-agents in generated workflows.

    Args:
        search_query: Search term to find relevant public agents
        max_results: Maximum number of agents to return (default 10)

    Returns:
        List of LibraryAgentSummary with full input/output schemas
    """
    search_term = search_query.strip()
    if len(search_term) > 100:
        raise ValueError(
            f"Search query is too long ({len(search_term)} chars, max 100). "
            f"Please use a shorter, more specific search term."
        )

    try:
        response = await store_db().get_store_agents(
            search_query=search_term,
            page=1,
            page_size=max_results,
        )

        agents_with_graphs = [
            agent for agent in response.agents if agent.agent_graph_id
        ]

        if not agents_with_graphs:
            return []

        graph_ids = [agent.agent_graph_id for agent in agents_with_graphs]
        graphs = await graph_db().get_store_listed_graphs(graph_ids)

        results: list[LibraryAgentSummary] = []
        for agent in agents_with_graphs:
            graph_id = agent.agent_graph_id
            if graph_id and graph_id in graphs:
                graph = graphs[graph_id]
                results.append(
                    LibraryAgentSummary(
                        graph_id=graph.id,
                        graph_version=graph.version,
                        name=agent.agent_name,
                        description=agent.description,
                        input_schema=graph.input_schema,
                        output_schema=graph.output_schema,
                    )
                )
        return results
    except Exception as e:
        logger.warning(f"Failed to search marketplace agents: {e}")
        return []


async def get_all_relevant_agents_for_generation(
    user_id: str,
    search_query: str | None = None,
    exclude_graph_id: str | None = None,
    include_library: bool = True,
    include_marketplace: bool = True,
    max_library_results: int = 15,
    max_marketplace_results: int = 10,
) -> list[AgentSummary]:
    """Fetch relevant agents from library and/or marketplace.

    Searches both user's library and marketplace by default.
    Explicitly mentioned UUIDs in the search query are always looked up.

    Args:
        user_id: The user ID
        search_query: Search term to find relevant agents (user's goal/description)
        exclude_graph_id: Optional graph ID to exclude (prevents circular references)
        include_library: Whether to search user's library (default True)
        include_marketplace: Whether to also search marketplace (default True)
        max_library_results: Max library agents to return (default 15)
        max_marketplace_results: Max marketplace agents to return (default 10)

    Returns:
        List of AgentSummary with full schemas (both library and marketplace agents)
    """
    agents: list[AgentSummary] = []
    seen_graph_ids: set[str] = set()

    if search_query:
        mentioned_uuids = extract_uuids_from_text(search_query)
        for graph_id in mentioned_uuids:
            if graph_id == exclude_graph_id:
                continue
            agent = await get_library_agent_by_graph_id(user_id, graph_id)
            agent_graph_id = agent.get("graph_id") if agent else None
            if agent and agent_graph_id and agent_graph_id not in seen_graph_ids:
                agents.append(agent)
                seen_graph_ids.add(agent_graph_id)
                logger.debug(
                    f"Found explicitly mentioned agent: {agent.get('name') or 'Unknown'}"
                )

    if include_library:
        library_agents = await get_library_agents_for_generation(
            user_id=user_id,
            search_query=search_query,
            exclude_graph_id=exclude_graph_id,
            max_results=max_library_results,
        )
        for agent in library_agents:
            graph_id = agent.get("graph_id")
            if graph_id and graph_id not in seen_graph_ids:
                agents.append(agent)
                seen_graph_ids.add(graph_id)

    if include_marketplace and search_query:
        marketplace_agents = await search_marketplace_agents_for_generation(
            search_query=search_query,
            max_results=max_marketplace_results,
        )
        for agent in marketplace_agents:
            graph_id = agent.get("graph_id")
            if graph_id and graph_id not in seen_graph_ids:
                agents.append(agent)
                seen_graph_ids.add(graph_id)

    return agents


def extract_search_terms_from_steps(
    decomposition_result: DecompositionResult | dict[str, Any],
) -> list[str]:
    """Extract search terms from decomposed instruction steps.

    Analyzes the decomposition result to extract relevant keywords
    for additional library agent searches.

    Args:
        decomposition_result: Result from decompose_goal containing steps

    Returns:
        List of unique search terms extracted from steps
    """
    search_terms: list[str] = []

    if decomposition_result.get("type") != "instructions":
        return search_terms

    steps = decomposition_result.get("steps", [])
    if not steps:
        return search_terms

    step_keys: list[str] = ["description", "action", "block_name", "tool", "name"]

    for step in steps:
        for key in step_keys:
            value = step.get(key)  # type: ignore[union-attr]
            if isinstance(value, str) and len(value) > 3:
                search_terms.append(value)

    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in search_terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    return unique_terms


async def enrich_library_agents_from_steps(
    user_id: str,
    decomposition_result: DecompositionResult | dict[str, Any],
    existing_agents: Sequence[AgentSummary] | Sequence[dict[str, Any]],
    exclude_graph_id: str | None = None,
    include_marketplace: bool = True,
    max_additional_results: int = 10,
) -> list[AgentSummary] | list[dict[str, Any]]:
    """Enrich library agents list with additional searches based on decomposed steps.

    This implements two-phase search: after decomposition, we search for additional
    relevant agents based on the specific steps identified.

    Args:
        user_id: The user ID
        decomposition_result: Result from decompose_goal containing steps
        existing_agents: Already fetched library agents from initial search
        exclude_graph_id: Optional graph ID to exclude
        include_marketplace: Whether to also search marketplace
        max_additional_results: Max additional agents per search term (default 10)

    Returns:
        Combined list of library agents (existing + newly discovered)
    """
    search_terms = extract_search_terms_from_steps(decomposition_result)

    if not search_terms:
        return list(existing_agents)

    existing_ids: set[str] = set()
    existing_names: set[str] = set()

    for agent in existing_agents:
        agent_name = agent.get("name")
        if agent_name and isinstance(agent_name, str):
            existing_names.add(agent_name.lower())
        graph_id = agent.get("graph_id")  # type: ignore[call-overload]
        if graph_id and isinstance(graph_id, str):
            existing_ids.add(graph_id)

    all_agents: list[AgentSummary] | list[dict[str, Any]] = list(existing_agents)

    for term in search_terms[:3]:
        try:
            additional_agents = await get_all_relevant_agents_for_generation(
                user_id=user_id,
                search_query=term,
                exclude_graph_id=exclude_graph_id,
                include_marketplace=include_marketplace,
                max_library_results=max_additional_results,
                max_marketplace_results=5,
            )

            for agent in additional_agents:
                agent_name = agent.get("name")
                if not agent_name or not isinstance(agent_name, str):
                    continue
                agent_name_lower = agent_name.lower()

                if agent_name_lower in existing_names:
                    continue

                graph_id = agent.get("graph_id")  # type: ignore[call-overload]
                if graph_id and graph_id in existing_ids:
                    continue

                all_agents.append(agent)
                existing_names.add(agent_name_lower)
                if graph_id and isinstance(graph_id, str):
                    existing_ids.add(graph_id)

        except DatabaseError:
            logger.error(f"Database error searching for agents with term '{term}'")
            raise
        except Exception as e:
            logger.warning(
                f"Failed to search for additional agents with term '{term}': {e}"
            )

    logger.debug(
        f"Enriched library agents: {len(existing_agents)} initial + "
        f"{len(all_agents) - len(existing_agents)} additional = {len(all_agents)} total"
    )

    return all_agents


async def decompose_goal(
    description: str,
    context: str = "",
    library_agents: Sequence[AgentSummary] | None = None,
) -> DecompositionResult | None:
    """Break down a goal into steps or return clarifying questions.

    Args:
        description: Natural language goal description
        context: Additional context (e.g., answers to previous questions)
        library_agents: User's library agents available for sub-agent composition

    Returns:
        DecompositionResult with either:
        - {"type": "clarifying_questions", "questions": [...]}
        - {"type": "instructions", "steps": [...]}
        Or None on error

    Raises:
        AgentGeneratorNotConfiguredError: If the external service is not configured.
    """
    _check_service_configured()
    logger.info("Calling external Agent Generator service for decompose_goal")
    result = await decompose_goal_external(
        description, context, _to_dict_list(library_agents)
    )
    return result  # type: ignore[return-value]


async def generate_agent(
    instructions: DecompositionResult | dict[str, Any],
    library_agents: Sequence[AgentSummary] | Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Generate agent JSON from instructions.

    Args:
        instructions: Structured instructions from decompose_goal
        library_agents: User's library agents available for sub-agent composition

    Returns:
        Agent JSON dict, error dict {"type": "error", ...}, or None on error

    Raises:
        AgentGeneratorNotConfiguredError: If the external service is not configured.
    """
    _check_service_configured()
    logger.info("Calling external Agent Generator service for generate_agent")
    result = await generate_agent_external(
        dict(instructions), _to_dict_list(library_agents)
    )

    if result:
        if isinstance(result, dict) and result.get("type") == "error":
            return result
        if "id" not in result:
            result["id"] = str(uuid.uuid4())
        if "version" not in result:
            result["version"] = 1
        if "is_active" not in result:
            result["is_active"] = True
    return result


class AgentJsonValidationError(Exception):
    """Raised when agent JSON is invalid or missing required fields."""

    pass


def json_to_graph(agent_json: dict[str, Any]) -> Graph:
    """Convert agent JSON dict to Graph model.

    Args:
        agent_json: Agent JSON with nodes and links

    Returns:
        Graph ready for saving

    Raises:
        AgentJsonValidationError: If required fields are missing from nodes or links
    """
    nodes = []
    for idx, n in enumerate(agent_json.get("nodes", [])):
        block_id = n.get("block_id")
        if not block_id:
            node_id = n.get("id", f"index_{idx}")
            raise AgentJsonValidationError(
                f"Node '{node_id}' is missing required field 'block_id'"
            )
        node = Node(
            id=n.get("id", str(uuid.uuid4())),
            block_id=block_id,
            input_default=n.get("input_default", {}),
            metadata=n.get("metadata", {}),
        )
        nodes.append(node)

    links = []
    for idx, link_data in enumerate(agent_json.get("links", [])):
        source_id = link_data.get("source_id")
        sink_id = link_data.get("sink_id")
        source_name = link_data.get("source_name")
        sink_name = link_data.get("sink_name")

        missing_fields = []
        if not source_id:
            missing_fields.append("source_id")
        if not sink_id:
            missing_fields.append("sink_id")
        if not source_name:
            missing_fields.append("source_name")
        if not sink_name:
            missing_fields.append("sink_name")

        if missing_fields:
            link_id = link_data.get("id", f"index_{idx}")
            raise AgentJsonValidationError(
                f"Link '{link_id}' is missing required fields: {', '.join(missing_fields)}"
            )

        link = Link(
            id=link_data.get("id", str(uuid.uuid4())),
            source_id=source_id,
            sink_id=sink_id,
            source_name=source_name,
            sink_name=sink_name,
            is_static=link_data.get("is_static", False),
        )
        links.append(link)

    return Graph(
        id=agent_json.get("id", str(uuid.uuid4())),
        version=agent_json.get("version", 1),
        is_active=agent_json.get("is_active", True),
        name=agent_json.get("name", "Generated Agent"),
        description=agent_json.get("description", ""),
        nodes=nodes,
        links=links,
    )


async def save_agent_to_library(
    agent_json: dict[str, Any], user_id: str, is_update: bool = False
) -> tuple[Graph, Any]:
    """Save agent to database and user's library.

    Args:
        agent_json: Agent JSON dict
        user_id: User ID
        is_update: Whether this is an update to an existing agent

    Returns:
        Tuple of (created Graph, LibraryAgent)
    """
    graph = json_to_graph(agent_json)
    db = library_db()
    if is_update:
        return await db.update_graph_in_library(graph, user_id)
    return await db.create_graph_in_library(graph, user_id)


def graph_to_json(graph: Graph) -> dict[str, Any]:
    """Convert a Graph object to JSON format for the agent generator.

    Args:
        graph: Graph object to convert

    Returns:
        Agent as JSON dict
    """
    nodes = []
    for node in graph.nodes:
        nodes.append(
            {
                "id": node.id,
                "block_id": node.block_id,
                "input_default": node.input_default,
                "metadata": node.metadata,
            }
        )

    links = []
    for node in graph.nodes:
        for link in node.output_links:
            links.append(
                {
                    "id": link.id,
                    "source_id": link.source_id,
                    "sink_id": link.sink_id,
                    "source_name": link.source_name,
                    "sink_name": link.sink_name,
                    "is_static": link.is_static,
                }
            )

    return {
        "id": graph.id,
        "name": graph.name,
        "description": graph.description,
        "version": graph.version,
        "is_active": graph.is_active,
        "nodes": nodes,
        "links": links,
    }


async def get_agent_as_json(
    agent_id: str, user_id: str | None
) -> dict[str, Any] | None:
    """Fetch an agent and convert to JSON format for editing.

    Args:
        agent_id: Graph ID or library agent ID
        user_id: User ID

    Returns:
        Agent as JSON dict or None if not found
    """
    db = graph_db()

    graph = await db.get_graph(agent_id, version=None, user_id=user_id)

    if not graph and user_id:
        try:
            library_agent = await library_db().get_library_agent(agent_id, user_id)
            graph = await db.get_graph(
                library_agent.graph_id, version=None, user_id=user_id
            )
        except NotFoundError:
            pass

    if not graph:
        return None

    return graph_to_json(graph)


async def generate_agent_patch(
    update_request: str,
    current_agent: dict[str, Any],
    library_agents: Sequence[AgentSummary] | None = None,
) -> dict[str, Any] | None:
    """Update an existing agent using natural language.

    The external Agent Generator service handles:
    - Generating the patch
    - Applying the patch
    - Fixing and validating the result

    Args:
        update_request: Natural language description of changes
        current_agent: Current agent JSON
        library_agents: User's library agents available for sub-agent composition

    Returns:
        Updated agent JSON, clarifying questions dict {"type": "clarifying_questions", ...},
        error dict {"type": "error", ...}, or None on error

    Raises:
        AgentGeneratorNotConfiguredError: If the external service is not configured.
    """
    _check_service_configured()
    logger.info("Calling external Agent Generator service for generate_agent_patch")
    return await generate_agent_patch_external(
        update_request,
        current_agent,
        _to_dict_list(library_agents),
    )


async def customize_template(
    template_agent: dict[str, Any],
    modification_request: str,
    context: str = "",
) -> dict[str, Any] | None:
    """Customize a template/marketplace agent using natural language.

    This is used when users want to modify a template or marketplace agent
    to fit their specific needs before adding it to their library.

    The external Agent Generator service handles:
    - Understanding the modification request
    - Applying changes to the template
    - Fixing and validating the result

    Args:
        template_agent: The template agent JSON to customize
        modification_request: Natural language description of customizations
        context: Additional context (e.g., answers to previous questions)

    Returns:
        Customized agent JSON, clarifying questions dict {"type": "clarifying_questions", ...},
        error dict {"type": "error", ...}, or None on unexpected error

    Raises:
        AgentGeneratorNotConfiguredError: If the external service is not configured.
    """
    _check_service_configured()
    logger.info("Calling external Agent Generator service for customize_template")
    return await customize_template_external(
        template_agent, modification_request, context
    )
