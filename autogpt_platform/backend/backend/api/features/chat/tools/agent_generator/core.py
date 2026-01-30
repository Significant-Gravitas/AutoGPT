"""Core agent generation functions."""

import logging
import uuid
from typing import Any

from backend.api.features.library import db as library_db
from backend.data.graph import Graph, Link, Node, create_graph

from .service import (
    decompose_goal_external,
    generate_agent_external,
    generate_agent_patch_external,
    is_external_service_configured,
)

logger = logging.getLogger(__name__)


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


async def get_library_agents_for_generation(
    user_id: str,
    search_query: str | None = None,
    exclude_graph_id: str | None = None,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """Fetch user's library agents formatted for Agent Generator.

    Uses search-based fetching to return relevant agents instead of all agents.
    This is more scalable for users with large libraries.

    Args:
        user_id: The user ID
        search_query: Optional search term to find relevant agents (user's goal/description)
        exclude_graph_id: Optional graph ID to exclude (prevents circular references)
        max_results: Maximum number of agents to return (default 15)

    Returns:
        List of library agent dicts with schemas for sub-agent composition
    """
    try:
        response = await library_db.list_library_agents(
            user_id=user_id,
            search_term=search_query,  # Use search API
            page=1,
            page_size=max_results,
        )

        return [
            {
                "graph_id": agent.graph_id,
                "graph_version": agent.graph_version,
                "name": agent.name,
                "description": agent.description,
                "input_schema": agent.input_schema,
                "output_schema": agent.output_schema,
            }
            for agent in response.agents
            # Exclude the agent being generated/edited to prevent circular references
            if exclude_graph_id is None or agent.graph_id != exclude_graph_id
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch library agents: {e}")
        return []


async def search_marketplace_agents_for_generation(
    search_query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search marketplace agents formatted for Agent Generator.

    Note: This returns basic agent info. Full input/output schemas would require
    additional graph fetches and is a potential future enhancement.

    Args:
        search_query: Search term to find relevant public agents
        max_results: Maximum number of agents to return (default 10)

    Returns:
        List of marketplace agent dicts (without detailed schemas for now)
    """
    from backend.api.features.store import db as store_db

    try:
        response = await store_db.get_store_agents(
            search_query=search_query,
            page=1,
            page_size=max_results,
        )

        # Return basic info - full schemas would require fetching each agent's graph
        return [
            {
                "name": agent.agent_name,
                "description": agent.description,
                "sub_heading": agent.sub_heading,
                "creator": agent.creator,
                "is_marketplace_agent": True,
                # Note: graph_id and schemas not available without additional fetches
            }
            for agent in response.agents
        ]
    except Exception as e:
        logger.warning(f"Failed to search marketplace agents: {e}")
        return []


async def get_all_relevant_agents_for_generation(
    user_id: str,
    search_query: str | None = None,
    exclude_graph_id: str | None = None,
    include_marketplace: bool = True,
    max_library_results: int = 15,
    max_marketplace_results: int = 10,
) -> list[dict[str, Any]]:
    """Fetch relevant agents from library and optionally marketplace.

    Combines search results from user's library and public marketplace,
    with library agents taking priority (they have full schemas).

    Args:
        user_id: The user ID
        search_query: Search term to find relevant agents (user's goal/description)
        exclude_graph_id: Optional graph ID to exclude (prevents circular references)
        include_marketplace: Whether to also search marketplace (default True)
        max_library_results: Max library agents to return (default 15)
        max_marketplace_results: Max marketplace agents to return (default 10)

    Returns:
        List of agent dicts, library agents first (with full schemas),
        then marketplace agents (basic info only)
    """
    agents: list[dict[str, Any]] = []

    # Get library agents (these have full schemas)
    library_agents = await get_library_agents_for_generation(
        user_id=user_id,
        search_query=search_query,
        exclude_graph_id=exclude_graph_id,
        max_results=max_library_results,
    )
    agents.extend(library_agents)

    # Optionally add marketplace agents
    if include_marketplace and search_query:
        marketplace_agents = await search_marketplace_agents_for_generation(
            search_query=search_query,
            max_results=max_marketplace_results,
        )
        # Add marketplace agents that aren't already in library (by name)
        library_names = {a["name"].lower() for a in library_agents if a.get("name")}
        for agent in marketplace_agents:
            agent_name = agent.get("name")
            if agent_name and agent_name.lower() not in library_names:
                agents.append(agent)

    return agents


async def decompose_goal(
    description: str,
    context: str = "",
    library_agents: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Break down a goal into steps or return clarifying questions.

    Args:
        description: Natural language goal description
        context: Additional context (e.g., answers to previous questions)
        library_agents: User's library agents available for sub-agent composition

    Returns:
        Dict with either:
        - {"type": "clarifying_questions", "questions": [...]}
        - {"type": "instructions", "steps": [...]}
        Or None on error

    Raises:
        AgentGeneratorNotConfiguredError: If the external service is not configured.
    """
    _check_service_configured()
    logger.info("Calling external Agent Generator service for decompose_goal")
    return await decompose_goal_external(description, context, library_agents)


async def generate_agent(
    instructions: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
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
    result = await generate_agent_external(instructions, library_agents)
    if result:
        # Check if it's an error response - pass through as-is
        if isinstance(result, dict) and result.get("type") == "error":
            return result
        # Ensure required fields for successful agent generation
        if "id" not in result:
            result["id"] = str(uuid.uuid4())
        if "version" not in result:
            result["version"] = 1
        if "is_active" not in result:
            result["is_active"] = True
    return result


def json_to_graph(agent_json: dict[str, Any]) -> Graph:
    """Convert agent JSON dict to Graph model.

    Args:
        agent_json: Agent JSON with nodes and links

    Returns:
        Graph ready for saving
    """
    nodes = []
    for n in agent_json.get("nodes", []):
        node = Node(
            id=n.get("id", str(uuid.uuid4())),
            block_id=n["block_id"],
            input_default=n.get("input_default", {}),
            metadata=n.get("metadata", {}),
        )
        nodes.append(node)

    links = []
    for link_data in agent_json.get("links", []):
        link = Link(
            id=link_data.get("id", str(uuid.uuid4())),
            source_id=link_data["source_id"],
            sink_id=link_data["sink_id"],
            source_name=link_data["source_name"],
            sink_name=link_data["sink_name"],
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


def _reassign_node_ids(graph: Graph) -> None:
    """Reassign all node and link IDs to new UUIDs.

    This is needed when creating a new version to avoid unique constraint violations.
    """
    # Create mapping from old node IDs to new UUIDs
    id_map = {node.id: str(uuid.uuid4()) for node in graph.nodes}

    # Reassign node IDs
    for node in graph.nodes:
        node.id = id_map[node.id]

    # Update link references to use new node IDs
    for link in graph.links:
        link.id = str(uuid.uuid4())  # Also give links new IDs
        if link.source_id in id_map:
            link.source_id = id_map[link.source_id]
        if link.sink_id in id_map:
            link.sink_id = id_map[link.sink_id]


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
    from backend.data.graph import get_graph_all_versions

    graph = json_to_graph(agent_json)

    if is_update:
        # For updates, keep the same graph ID but increment version
        # and reassign node/link IDs to avoid conflicts
        if graph.id:
            existing_versions = await get_graph_all_versions(graph.id, user_id)
            if existing_versions:
                latest_version = max(v.version for v in existing_versions)
                graph.version = latest_version + 1
                # Reassign node IDs (but keep graph ID the same)
                _reassign_node_ids(graph)
                logger.info(f"Updating agent {graph.id} to version {graph.version}")
    else:
        # For new agents, always generate a fresh UUID to avoid collisions
        graph.id = str(uuid.uuid4())
        graph.version = 1
        # Reassign all node IDs as well
        _reassign_node_ids(graph)
        logger.info(f"Creating new agent with ID {graph.id}")

    # Save to database
    created_graph = await create_graph(graph, user_id)

    # Add to user's library (or update existing library agent)
    library_agents = await library_db.create_library_agent(
        graph=created_graph,
        user_id=user_id,
        sensitive_action_safe_mode=True,
        create_library_agents_for_sub_graphs=False,
    )

    return created_graph, library_agents[0]


async def get_agent_as_json(
    graph_id: str, user_id: str | None
) -> dict[str, Any] | None:
    """Fetch an agent and convert to JSON format for editing.

    Args:
        graph_id: Graph ID or library agent ID
        user_id: User ID

    Returns:
        Agent as JSON dict or None if not found
    """
    from backend.data.graph import get_graph

    # Try to get the graph (version=None gets the active version)
    graph = await get_graph(graph_id, version=None, user_id=user_id)
    if not graph:
        return None

    # Convert to JSON format
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


async def generate_agent_patch(
    update_request: str,
    current_agent: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
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
        error dict {"type": "error", ...}, or None on unexpected error

    Raises:
        AgentGeneratorNotConfiguredError: If the external service is not configured.
    """
    _check_service_configured()
    logger.info("Calling external Agent Generator service for generate_agent_patch")
    return await generate_agent_patch_external(
        update_request, current_agent, library_agents
    )
