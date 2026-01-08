"""Agent fixer - Fixes common LLM generation errors."""

import logging
import re
import uuid
from typing import Any

from .utils import (
    ADDTODICTIONARY_BLOCK_ID,
    ADDTOLIST_BLOCK_ID,
    CODE_EXECUTION_BLOCK_ID,
    CONDITION_BLOCK_ID,
    CREATEDICT_BLOCK_ID,
    CREATELIST_BLOCK_ID,
    DATA_SAMPLING_BLOCK_ID,
    DOUBLE_CURLY_BRACES_BLOCK_IDS,
    GET_CURRENT_DATE_BLOCK_ID,
    STORE_VALUE_BLOCK_ID,
    UNIVERSAL_TYPE_CONVERTER_BLOCK_ID,
    get_blocks_info,
    is_valid_uuid,
)

logger = logging.getLogger(__name__)


def fix_agent_ids(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix invalid UUIDs in agent and link IDs."""
    # Fix agent ID
    if not is_valid_uuid(agent.get("id", "")):
        agent["id"] = str(uuid.uuid4())
        logger.debug(f"Fixed agent ID: {agent['id']}")

    # Fix node IDs
    id_mapping = {}  # Old ID -> New ID
    for node in agent.get("nodes", []):
        if not is_valid_uuid(node.get("id", "")):
            old_id = node.get("id", "")
            new_id = str(uuid.uuid4())
            id_mapping[old_id] = new_id
            node["id"] = new_id
            logger.debug(f"Fixed node ID: {old_id} -> {new_id}")

    # Fix link IDs and update references
    for link in agent.get("links", []):
        if not is_valid_uuid(link.get("id", "")):
            link["id"] = str(uuid.uuid4())
            logger.debug(f"Fixed link ID: {link['id']}")

        # Update source/sink IDs if they were remapped
        if link.get("source_id") in id_mapping:
            link["source_id"] = id_mapping[link["source_id"]]
        if link.get("sink_id") in id_mapping:
            link["sink_id"] = id_mapping[link["sink_id"]]

    return agent


def fix_double_curly_braces(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix single curly braces to double in template blocks."""
    for node in agent.get("nodes", []):
        if node.get("block_id") not in DOUBLE_CURLY_BRACES_BLOCK_IDS:
            continue

        input_data = node.get("input_default", {})
        for key in ("prompt", "format"):
            if key in input_data and isinstance(input_data[key], str):
                original = input_data[key]
                # Fix simple variable references: {var} -> {{var}}
                fixed = re.sub(
                    r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})",
                    r"{{\1}}",
                    original,
                )
                if fixed != original:
                    input_data[key] = fixed
                    logger.debug(f"Fixed curly braces in {key}")

    return agent


def fix_storevalue_before_condition(agent: dict[str, Any]) -> dict[str, Any]:
    """Add StoreValueBlock before ConditionBlock if needed for value2."""
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])

    # Find all ConditionBlock nodes
    condition_node_ids = {
        node["id"] for node in nodes if node.get("block_id") == CONDITION_BLOCK_ID
    }

    if not condition_node_ids:
        return agent

    new_nodes = []
    new_links = []
    processed_conditions = set()

    for link in links:
        sink_id = link.get("sink_id")
        sink_name = link.get("sink_name")

        # Check if this link goes to a ConditionBlock's value2
        if sink_id in condition_node_ids and sink_name == "value2":
            source_node = next(
                (n for n in nodes if n["id"] == link.get("source_id")), None
            )

            # Skip if source is already a StoreValueBlock
            if source_node and source_node.get("block_id") == STORE_VALUE_BLOCK_ID:
                continue

            # Skip if we already processed this condition
            if sink_id in processed_conditions:
                continue

            processed_conditions.add(sink_id)

            # Create StoreValueBlock
            store_node_id = str(uuid.uuid4())
            store_node = {
                "id": store_node_id,
                "block_id": STORE_VALUE_BLOCK_ID,
                "input_default": {"data": None},
                "metadata": {"position": {"x": 0, "y": -100}},
            }
            new_nodes.append(store_node)

            # Create link: original source -> StoreValueBlock
            new_links.append(
                {
                    "id": str(uuid.uuid4()),
                    "source_id": link["source_id"],
                    "source_name": link["source_name"],
                    "sink_id": store_node_id,
                    "sink_name": "input",
                    "is_static": False,
                }
            )

            # Update original link: StoreValueBlock -> ConditionBlock
            link["source_id"] = store_node_id
            link["source_name"] = "output"

            logger.debug(f"Added StoreValueBlock before ConditionBlock {sink_id}")

    if new_nodes:
        agent["nodes"] = nodes + new_nodes

    return agent


def fix_addtolist_blocks(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix AddToList blocks by adding prerequisite empty AddToList block.

    When an AddToList block is found:
    1. Checks if there's a CreateListBlock before it
    2. Removes CreateListBlock if linked directly to AddToList
    3. Adds an empty AddToList block before the original
    4. Ensures the original has a self-referencing link
    """
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])
    new_nodes = []
    original_addtolist_ids = set()
    nodes_to_remove = set()
    links_to_remove = []

    # First pass: identify CreateListBlock nodes to remove
    for link in links:
        source_node = next(
            (n for n in nodes if n.get("id") == link.get("source_id")), None
        )
        sink_node = next((n for n in nodes if n.get("id") == link.get("sink_id")), None)

        if (
            source_node
            and sink_node
            and source_node.get("block_id") == CREATELIST_BLOCK_ID
            and sink_node.get("block_id") == ADDTOLIST_BLOCK_ID
        ):
            nodes_to_remove.add(source_node.get("id"))
            links_to_remove.append(link)
            logger.debug(f"Removing CreateListBlock {source_node.get('id')}")

    # Second pass: process AddToList blocks
    filtered_nodes = []
    for node in nodes:
        if node.get("id") in nodes_to_remove:
            continue

        if node.get("block_id") == ADDTOLIST_BLOCK_ID:
            original_addtolist_ids.add(node.get("id"))
            node_id = node.get("id")
            pos = node.get("metadata", {}).get("position", {"x": 0, "y": 0})

            # Check if already has prerequisite
            has_prereq = any(
                link.get("sink_id") == node_id
                and link.get("sink_name") == "list"
                and link.get("source_name") == "updated_list"
                for link in links
            )

            if not has_prereq:
                # Remove links to "list" input (except self-reference)
                for link in links:
                    if (
                        link.get("sink_id") == node_id
                        and link.get("sink_name") == "list"
                        and link.get("source_id") != node_id
                        and link not in links_to_remove
                    ):
                        links_to_remove.append(link)

                # Create prerequisite AddToList block
                prereq_id = str(uuid.uuid4())
                prereq_node = {
                    "id": prereq_id,
                    "block_id": ADDTOLIST_BLOCK_ID,
                    "input_default": {"list": [], "entry": None, "entries": []},
                    "metadata": {
                        "position": {"x": pos.get("x", 0) - 800, "y": pos.get("y", 0)}
                    },
                }
                new_nodes.append(prereq_node)

                # Link prerequisite to original
                links.append(
                    {
                        "id": str(uuid.uuid4()),
                        "source_id": prereq_id,
                        "source_name": "updated_list",
                        "sink_id": node_id,
                        "sink_name": "list",
                        "is_static": False,
                    }
                )
                logger.debug(f"Added prerequisite AddToList block for {node_id}")

        filtered_nodes.append(node)

    # Remove marked links
    filtered_links = [link for link in links if link not in links_to_remove]

    # Add self-referencing links for original AddToList blocks
    for node in filtered_nodes + new_nodes:
        if (
            node.get("block_id") == ADDTOLIST_BLOCK_ID
            and node.get("id") in original_addtolist_ids
        ):
            node_id = node.get("id")
            has_self_ref = any(
                link["source_id"] == node_id
                and link["sink_id"] == node_id
                and link["source_name"] == "updated_list"
                and link["sink_name"] == "list"
                for link in filtered_links
            )
            if not has_self_ref:
                filtered_links.append(
                    {
                        "id": str(uuid.uuid4()),
                        "source_id": node_id,
                        "source_name": "updated_list",
                        "sink_id": node_id,
                        "sink_name": "list",
                        "is_static": False,
                    }
                )
                logger.debug(f"Added self-reference for AddToList {node_id}")

    agent["nodes"] = filtered_nodes + new_nodes
    agent["links"] = filtered_links
    return agent


def fix_addtodictionary_blocks(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix AddToDictionary blocks by removing empty CreateDictionary nodes."""
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])
    nodes_to_remove = set()
    links_to_remove = []

    for link in links:
        source_node = next(
            (n for n in nodes if n.get("id") == link.get("source_id")), None
        )
        sink_node = next((n for n in nodes if n.get("id") == link.get("sink_id")), None)

        if (
            source_node
            and sink_node
            and source_node.get("block_id") == CREATEDICT_BLOCK_ID
            and sink_node.get("block_id") == ADDTODICTIONARY_BLOCK_ID
        ):
            nodes_to_remove.add(source_node.get("id"))
            links_to_remove.append(link)
            logger.debug(f"Removing CreateDictionary {source_node.get('id')}")

    agent["nodes"] = [n for n in nodes if n.get("id") not in nodes_to_remove]
    agent["links"] = [link for link in links if link not in links_to_remove]
    return agent


def fix_code_execution_output(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix CodeExecutionBlock output: change 'response' to 'stdout_logs'."""
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])

    for link in links:
        source_node = next(
            (n for n in nodes if n.get("id") == link.get("source_id")), None
        )
        if (
            source_node
            and source_node.get("block_id") == CODE_EXECUTION_BLOCK_ID
            and link.get("source_name") == "response"
        ):
            link["source_name"] = "stdout_logs"
            logger.debug("Fixed CodeExecutionBlock output: response -> stdout_logs")

    return agent


def fix_data_sampling_sample_size(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix DataSamplingBlock by setting sample_size to 1 as default."""
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])
    links_to_remove = []

    for node in nodes:
        if node.get("block_id") == DATA_SAMPLING_BLOCK_ID:
            node_id = node.get("id")
            input_default = node.get("input_default", {})

            # Remove links to sample_size
            for link in links:
                if (
                    link.get("sink_id") == node_id
                    and link.get("sink_name") == "sample_size"
                ):
                    links_to_remove.append(link)

            # Set default
            input_default["sample_size"] = 1
            node["input_default"] = input_default
            logger.debug(f"Fixed DataSamplingBlock {node_id} sample_size to 1")

    if links_to_remove:
        agent["links"] = [link for link in links if link not in links_to_remove]

    return agent


def fix_node_x_coordinates(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix node x-coordinates to ensure 800+ unit spacing between linked nodes."""
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])
    node_lookup = {n.get("id"): n for n in nodes}

    for link in links:
        source_id = link.get("source_id")
        sink_id = link.get("sink_id")

        source_node = node_lookup.get(source_id)
        sink_node = node_lookup.get(sink_id)

        if not source_node or not sink_node:
            continue

        source_pos = source_node.get("metadata", {}).get("position", {})
        sink_pos = sink_node.get("metadata", {}).get("position", {})

        source_x = source_pos.get("x", 0)
        sink_x = sink_pos.get("x", 0)

        if abs(sink_x - source_x) < 800:
            new_x = source_x + 800
            if "metadata" not in sink_node:
                sink_node["metadata"] = {}
            if "position" not in sink_node["metadata"]:
                sink_node["metadata"]["position"] = {}
            sink_node["metadata"]["position"]["x"] = new_x
            logger.debug(f"Fixed node {sink_id} x: {sink_x} -> {new_x}")

    return agent


def fix_getcurrentdate_offset(agent: dict[str, Any]) -> dict[str, Any]:
    """Fix GetCurrentDateBlock offset to ensure it's positive."""
    for node in agent.get("nodes", []):
        if node.get("block_id") == GET_CURRENT_DATE_BLOCK_ID:
            input_default = node.get("input_default", {})
            if "offset" in input_default:
                offset = input_default["offset"]
                if isinstance(offset, (int, float)) and offset < 0:
                    input_default["offset"] = abs(offset)
                    logger.debug(f"Fixed offset: {offset} -> {abs(offset)}")

    return agent


def fix_ai_model_parameter(
    agent: dict[str, Any],
    blocks_info: list[dict[str, Any]],
    default_model: str = "gpt-4o",
) -> dict[str, Any]:
    """Add default model parameter to AI blocks if missing."""
    block_map = {b.get("id"): b for b in blocks_info}

    for node in agent.get("nodes", []):
        block_id = node.get("block_id")
        block = block_map.get(block_id)

        if not block:
            continue

        # Check if block has AI category
        categories = block.get("categories", [])
        is_ai_block = any(
            cat.get("category") == "AI" for cat in categories if isinstance(cat, dict)
        )

        if is_ai_block:
            input_default = node.get("input_default", {})
            if "model" not in input_default:
                input_default["model"] = default_model
                node["input_default"] = input_default
                logger.debug(
                    f"Added model '{default_model}' to AI block {node.get('id')}"
                )

    return agent


def fix_link_static_properties(
    agent: dict[str, Any], blocks_info: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fix is_static property based on source block's staticOutput."""
    block_map = {b.get("id"): b for b in blocks_info}
    node_lookup = {n.get("id"): n for n in agent.get("nodes", [])}

    for link in agent.get("links", []):
        source_node = node_lookup.get(link.get("source_id"))
        if not source_node:
            continue

        source_block = block_map.get(source_node.get("block_id"))
        if not source_block:
            continue

        static_output = source_block.get("staticOutput", False)
        if link.get("is_static") != static_output:
            link["is_static"] = static_output
            logger.debug(f"Fixed link {link.get('id')} is_static to {static_output}")

    return agent


def fix_data_type_mismatch(
    agent: dict[str, Any], blocks_info: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fix data type mismatches by inserting UniversalTypeConverterBlock."""
    nodes = agent.get("nodes", [])
    links = agent.get("links", [])
    block_map = {b.get("id"): b for b in blocks_info}
    node_lookup = {n.get("id"): n for n in nodes}

    def get_property_type(schema: dict, name: str) -> str | None:
        if "_#_" in name:
            parent, child = name.split("_#_", 1)
            parent_schema = schema.get(parent, {})
            if "properties" in parent_schema:
                return parent_schema["properties"].get(child, {}).get("type")
            return None
        return schema.get(name, {}).get("type")

    def are_types_compatible(src: str, sink: str) -> bool:
        if {src, sink} <= {"integer", "number"}:
            return True
        return src == sink

    type_mapping = {
        "string": "string",
        "text": "string",
        "integer": "number",
        "number": "number",
        "float": "number",
        "boolean": "boolean",
        "bool": "boolean",
        "array": "list",
        "list": "list",
        "object": "dictionary",
        "dict": "dictionary",
        "dictionary": "dictionary",
    }

    new_links = []
    nodes_to_add = []

    for link in links:
        source_node = node_lookup.get(link.get("source_id"))
        sink_node = node_lookup.get(link.get("sink_id"))

        if not source_node or not sink_node:
            new_links.append(link)
            continue

        source_block = block_map.get(source_node.get("block_id"))
        sink_block = block_map.get(sink_node.get("block_id"))

        if not source_block or not sink_block:
            new_links.append(link)
            continue

        source_outputs = source_block.get("outputSchema", {}).get("properties", {})
        sink_inputs = sink_block.get("inputSchema", {}).get("properties", {})

        source_type = get_property_type(source_outputs, link.get("source_name", ""))
        sink_type = get_property_type(sink_inputs, link.get("sink_name", ""))

        if (
            source_type
            and sink_type
            and not are_types_compatible(source_type, sink_type)
        ):
            # Insert type converter
            converter_id = str(uuid.uuid4())
            target_type = type_mapping.get(sink_type, sink_type)

            converter_node = {
                "id": converter_id,
                "block_id": UNIVERSAL_TYPE_CONVERTER_BLOCK_ID,
                "input_default": {"type": target_type},
                "metadata": {"position": {"x": 0, "y": 100}},
            }
            nodes_to_add.append(converter_node)

            # source -> converter
            new_links.append(
                {
                    "id": str(uuid.uuid4()),
                    "source_id": link["source_id"],
                    "source_name": link["source_name"],
                    "sink_id": converter_id,
                    "sink_name": "value",
                    "is_static": False,
                }
            )

            # converter -> sink
            new_links.append(
                {
                    "id": str(uuid.uuid4()),
                    "source_id": converter_id,
                    "source_name": "value",
                    "sink_id": link["sink_id"],
                    "sink_name": link["sink_name"],
                    "is_static": False,
                }
            )

            logger.debug(f"Inserted type converter: {source_type} -> {target_type}")
        else:
            new_links.append(link)

    if nodes_to_add:
        agent["nodes"] = nodes + nodes_to_add
        agent["links"] = new_links

    return agent


def apply_all_fixes(
    agent: dict[str, Any], blocks_info: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Apply all fixes to an agent JSON.

    Args:
        agent: Agent JSON dict
        blocks_info: Optional list of block info dicts for advanced fixes

    Returns:
        Fixed agent JSON
    """
    # Basic fixes (no block info needed)
    agent = fix_agent_ids(agent)
    agent = fix_double_curly_braces(agent)
    agent = fix_storevalue_before_condition(agent)
    agent = fix_addtolist_blocks(agent)
    agent = fix_addtodictionary_blocks(agent)
    agent = fix_code_execution_output(agent)
    agent = fix_data_sampling_sample_size(agent)
    agent = fix_node_x_coordinates(agent)
    agent = fix_getcurrentdate_offset(agent)

    # Advanced fixes (require block info)
    if blocks_info is None:
        blocks_info = get_blocks_info()

    agent = fix_ai_model_parameter(agent, blocks_info)
    agent = fix_link_static_properties(agent, blocks_info)
    agent = fix_data_type_mismatch(agent, blocks_info)

    return agent
