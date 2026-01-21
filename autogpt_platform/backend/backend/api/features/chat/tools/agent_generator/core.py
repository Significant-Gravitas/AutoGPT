"""Core agent generation functions."""

import copy
import json
import logging
import uuid
from typing import Any

from backend.api.features.library import db as library_db
from backend.data.graph import Graph, Link, Node, create_graph

from .client import AGENT_GENERATOR_MODEL, get_client
from .prompts import DECOMPOSITION_PROMPT, GENERATION_PROMPT, PATCH_PROMPT
from .utils import get_block_summaries, parse_json_from_llm

logger = logging.getLogger(__name__)


async def decompose_goal(description: str, context: str = "") -> dict[str, Any] | None:
    """Break down a goal into steps or return clarifying questions.

    Args:
        description: Natural language goal description
        context: Additional context (e.g., answers to previous questions)

    Returns:
        Dict with either:
        - {"type": "clarifying_questions", "questions": [...]}
        - {"type": "instructions", "steps": [...]}
        Or None on error
    """
    client = get_client()
    prompt = DECOMPOSITION_PROMPT.format(block_summaries=get_block_summaries())

    full_description = description
    if context:
        full_description = f"{description}\n\nAdditional context:\n{context}"

    try:
        response = await client.chat.completions.create(
            model=AGENT_GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": full_description},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            logger.error("LLM returned empty content for decomposition")
            return None

        result = parse_json_from_llm(content)

        if result is None:
            logger.error(f"Failed to parse decomposition response: {content[:200]}")
            return None

        return result

    except Exception as e:
        logger.error(f"Error decomposing goal: {e}")
        return None


async def generate_agent(instructions: dict[str, Any]) -> dict[str, Any] | None:
    """Generate agent JSON from instructions.

    Args:
        instructions: Structured instructions from decompose_goal

    Returns:
        Agent JSON dict or None on error
    """
    client = get_client()
    prompt = GENERATION_PROMPT.format(block_summaries=get_block_summaries())

    try:
        response = await client.chat.completions.create(
            model=AGENT_GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(instructions, indent=2)},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            logger.error("LLM returned empty content for agent generation")
            return None

        result = parse_json_from_llm(content)

        if result is None:
            logger.error(f"Failed to parse agent JSON: {content[:200]}")
            return None

        # Ensure required fields
        if "id" not in result:
            result["id"] = str(uuid.uuid4())
        if "version" not in result:
            result["version"] = 1
        if "is_active" not in result:
            result["is_active"] = True

        return result

    except Exception as e:
        logger.error(f"Error generating agent: {e}")
        return None


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
    update_request: str, current_agent: dict[str, Any]
) -> dict[str, Any] | None:
    """Generate a patch to update an existing agent.

    Args:
        update_request: Natural language description of changes
        current_agent: Current agent JSON

    Returns:
        Patch dict or clarifying questions, or None on error
    """
    client = get_client()
    prompt = PATCH_PROMPT.format(
        current_agent=json.dumps(current_agent, indent=2),
        block_summaries=get_block_summaries(),
    )

    try:
        response = await client.chat.completions.create(
            model=AGENT_GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": update_request},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            logger.error("LLM returned empty content for patch generation")
            return None

        return parse_json_from_llm(content)

    except Exception as e:
        logger.error(f"Error generating patch: {e}")
        return None


def apply_agent_patch(
    current_agent: dict[str, Any], patch: dict[str, Any]
) -> dict[str, Any]:
    """Apply a patch to an existing agent.

    Args:
        current_agent: Current agent JSON
        patch: Patch dict with operations

    Returns:
        Updated agent JSON
    """
    agent = copy.deepcopy(current_agent)
    patches = patch.get("patches", [])

    for p in patches:
        patch_type = p.get("type")

        if patch_type == "modify":
            node_id = p.get("node_id")
            changes = p.get("changes", {})

            for node in agent.get("nodes", []):
                if node["id"] == node_id:
                    _deep_update(node, changes)
                    logger.debug(f"Modified node {node_id}")
                    break

        elif patch_type == "add":
            new_nodes = p.get("new_nodes", [])
            new_links = p.get("new_links", [])

            agent["nodes"] = agent.get("nodes", []) + new_nodes
            agent["links"] = agent.get("links", []) + new_links
            logger.debug(f"Added {len(new_nodes)} nodes, {len(new_links)} links")

        elif patch_type == "remove":
            node_ids_to_remove = set(p.get("node_ids", []))
            link_ids_to_remove = set(p.get("link_ids", []))

            # Remove nodes
            agent["nodes"] = [
                n for n in agent.get("nodes", []) if n["id"] not in node_ids_to_remove
            ]

            # Remove links (both explicit and those referencing removed nodes)
            agent["links"] = [
                link
                for link in agent.get("links", [])
                if link["id"] not in link_ids_to_remove
                and link["source_id"] not in node_ids_to_remove
                and link["sink_id"] not in node_ids_to_remove
            ]

            logger.debug(
                f"Removed {len(node_ids_to_remove)} nodes, {len(link_ids_to_remove)} links"
            )

    return agent


def _deep_update(target: dict, source: dict) -> None:
    """Recursively update a dict with another dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
