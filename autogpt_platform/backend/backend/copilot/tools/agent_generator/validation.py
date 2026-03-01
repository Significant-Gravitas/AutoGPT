import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Module-level cache for blocks
_blocks_cache: list[dict[str, Any]] | None = None


def get_blocks_as_dicts() -> list[dict[str, Any]]:
    """Get all available blocks as a list of dicts.

    Each dict has keys: id, name, description, inputSchema, outputSchema,
    categories, staticOutput, costs, contributors, uiType.

    The result is cached at module level since blocks don't change at runtime.
    """
    global _blocks_cache
    if _blocks_cache is not None:
        return _blocks_cache

    from backend.blocks import get_blocks

    block_classes = get_blocks()
    blocks_list: list[dict[str, Any]] = []
    for block_cls in block_classes.values():
        try:
            block_instance = block_cls()
            info = block_instance.get_info()
            blocks_list.append(info.model_dump())
        except Exception:
            logger.warning(
                f"Failed to get info for block class {block_cls}", exc_info=True
            )
    _blocks_cache = blocks_list
    return _blocks_cache


class AgentFixer:
    """
    A comprehensive fixer for AutoGPT agents that applies various fixes to ensure
    agents are valid and functional.
    """

    def __init__(self):
        self.UUID_REGEX = re.compile(
            r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}$"
        )
        self.FIX_VALUE2_EMPTY_STRING_BLOCK_IDS = [
            "715696a0-e1da-45c8-b209-c2fa9c3b0be6"
        ]
        self.ADDTOLIST_BLOCK_ID = "aeb08fc1-2fc1-4141-bc8e-f758f183a822"
        self.ADDTODICTIONARY_BLOCK_ID = "31d1064e-7446-4693-a7d4-65e5ca1180d1"
        self.CODE_EXECUTION_BLOCK_ID = "0b02b072-abe7-11ef-8372-fb5d162dd712"
        self.DATA_SAMPLING_BLOCK_ID = "4a448883-71fa-49cf-91cf-70d793bd7d87"
        self.STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"
        self.UNIVERSAL_TYPE_CONVERTER_BLOCK_ID = "95d1b990-ce13-4d88-9737-ba5c2070c97b"
        self.GET_CURRENT_DATE_BLOCK_ID = "b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0b1"
        self.GMAIL_SEND_BLOCK_ID = "6c27abc2-e51d-499e-a85f-5a0041ba94f0"
        self.TEXT_REPLACE_BLOCK_ID = "7e7c87ab-3469-4bcc-9abe-67705091b713"
        self.AGENT_EXECUTOR_BLOCK_ID = "e189baac-8c20-45a1-94a7-55177ea42565"
        self.fixes_applied: list[str] = []

    def is_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        return isinstance(value, str) and self.UUID_REGEX.match(value) is not None

    def generate_uuid(self) -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())

    def add_fix_log(self, fix_description: str):
        """Add a fix description to the applied fixes list."""
        self.fixes_applied.append(fix_description)

    async def fix_agent_ids(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Fix agent and link IDs to ensure they are valid UUIDs.

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """

        def get_new_id() -> str:
            new_id = self.generate_uuid()
            return new_id

        # Fix link IDs
        links = []
        for link in agent.get("links", []):
            if not self.is_uuid(link["id"]):
                link["id"] = get_new_id()
                self.add_fix_log(f"Fixed link ID: {link['id']}")
            links.append(link)
        agent["links"] = links

        # Fix agent ID
        if not self.is_uuid(agent.get("id", "")):
            old_id = agent.get("id", "missing")
            agent["id"] = self.generate_uuid()
            self.add_fix_log(f"Fixed agent ID: {old_id} -> {agent['id']}")

        return agent

    async def fix_storevalue_before_condition(
        self, agent: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Add a StoreValueBlock before each ConditionBlock to provide a value for 'value2'.

        - Creates a StoreValueBlock node with default input and data False
        - Adds a link from the StoreValueBlock 'output' to the ConditionBlock 'value2'
        - Skips if a link to 'value2' already exists for the ConditionBlock
        - Prevents duplicate StoreValueBlocks by checking if one already exists for the
          same condition

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """

        nodes = agent.get("nodes", [])
        links = agent.get("links", []) or []

        # Collect ConditionBlock node ids
        condition_block_id = self.FIX_VALUE2_EMPTY_STRING_BLOCK_IDS[0]
        condition_node_ids = {
            node.get("id")
            for node in nodes
            if node.get("block_id") == condition_block_id
        }

        if not condition_node_ids:
            return agent

        new_links = []
        nodes_to_add = []
        store_node_counter = 0  # Counter to ensure unique positions

        for link in links:
            # Identify links going into ConditionBlock.value2
            if (
                link.get("sink_id") in condition_node_ids
                and link.get("sink_name") == "value2"
            ):
                condition_node_id = link.get("sink_id")

                # If the upstream source is already a StoreValueBlock.output, keep as-is
                source_node = next(
                    (n for n in nodes if n.get("id") == link.get("source_id")),
                    None,
                )
                if (
                    source_node
                    and source_node.get("block_id") == self.STORE_VALUE_BLOCK_ID
                    and link.get("source_name") == "output"
                ):
                    new_links.append(link)
                    continue

                # Check if there's already a StoreValueBlock connected to this
                # condition's value2. This prevents duplicates when the fix runs
                # multiple times.
                existing_storevalue_for_condition = False
                for existing_link in links:
                    if (
                        existing_link.get("sink_id") == condition_node_id
                        and existing_link.get("sink_name") == "value2"
                    ):
                        existing_source_node = next(
                            (
                                n
                                for n in nodes
                                if n.get("id") == existing_link.get("source_id")
                            ),
                            None,
                        )
                        if (
                            existing_source_node
                            and existing_source_node.get("block_id")
                            == self.STORE_VALUE_BLOCK_ID
                            and existing_link.get("source_name") == "output"
                        ):
                            existing_storevalue_for_condition = True
                            break

                if existing_storevalue_for_condition:
                    self.add_fix_log(
                        f"Skipped adding StoreValueBlock for ConditionBlock "
                        f"{condition_node_id} - already has one connected"
                    )
                    new_links.append(link)
                    continue

                # Create StoreValueBlock node (input will be linked; data left
                # default None)
                store_node_id = self.generate_uuid()
                store_node = {
                    "id": store_node_id,
                    "block_id": self.STORE_VALUE_BLOCK_ID,
                    "input_default": {"data": None},
                    "metadata": {
                        "position": {
                            "x": store_node_counter * 200,
                            "y": -100,
                        }
                    },
                    "graph_id": agent.get("id"),
                    "graph_version": 1,
                }
                nodes_to_add.append(store_node)
                store_node_counter += 1

                # Rewire: old source -> StoreValueBlock.input
                upstream_to_store_link = {
                    "id": self.generate_uuid(),
                    "source_id": link.get("source_id"),
                    "source_name": link.get("source_name"),
                    "sink_id": store_node_id,
                    "sink_name": "input",
                }

                # Then StoreValueBlock.output -> ConditionBlock.value2
                store_to_condition_link = {
                    "id": self.generate_uuid(),
                    "source_id": store_node_id,
                    "source_name": "output",
                    "sink_id": condition_node_id,
                    "sink_name": "value2",
                }

                new_links.append(upstream_to_store_link)
                new_links.append(store_to_condition_link)

                self.add_fix_log(
                    f"Inserted StoreValueBlock {store_node_id} between "
                    f"{link.get('source_id')}:{link.get('source_name')} and "
                    f"ConditionBlock {condition_node_id} value2"
                )
            else:
                new_links.append(link)

        if nodes_to_add:
            nodes.extend(nodes_to_add)
            agent["nodes"] = nodes
            agent["links"] = new_links

        return agent

    async def fix_double_curly_braces(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Fix single curly braces to double curly braces in nodes with prompt or
        format fields. Also ensures that prompt_values (passed via links) are
        referenced in the prompt/format. Skips fixing if the block's output will
        be passed to a CodeExecutionBlock.

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """

        nodes = agent.get("nodes", [])
        links = agent.get("links", [])

        # Build a map of node_id -> list of prompt_value names that are linked
        node_prompt_values: dict[str, list[str]] = {}
        for link in links:
            sink_id = link.get("sink_id")
            sink_name = link.get("sink_name", "")

            # Check if this link is passing a prompt_value
            if sink_name == "prompt_values":
                # Direct prompt_values link
                if sink_id not in node_prompt_values:
                    node_prompt_values[sink_id] = []
                # We don't have a specific name for this, skip for now
            elif sink_name.startswith("prompt_values_"):
                # Extract value name from pattern: prompt_values_#_[value_name]
                after_prefix = sink_name[len("prompt_values_") :]
                first_underscore_idx = after_prefix.find("_")
                if first_underscore_idx != -1:
                    value_name = after_prefix[first_underscore_idx + 1 :]
                    if sink_id not in node_prompt_values:
                        node_prompt_values[sink_id] = []
                    node_prompt_values[sink_id].append(value_name)

        # Process nodes that have prompt or format fields
        for node in nodes:
            node_id = node.get("id")
            input_data = node.get("input_default", {})

            # Check if this node has prompt or format fields
            has_prompt_or_format = "prompt" in input_data or "format" in input_data

            if not has_prompt_or_format:
                continue

            # Check if this block's output is linked to a CodeExecutionBlock
            is_linked_to_code_execution = False
            for link in links:
                if link.get("source_id") == node_id:
                    sink_node = next(
                        (n for n in nodes if n.get("id") == link.get("sink_id")),
                        None,
                    )
                    if (
                        sink_node
                        and sink_node.get("block_id") == self.CODE_EXECUTION_BLOCK_ID
                    ):
                        is_linked_to_code_execution = True
                        break

            # Skip fixing if this block's output goes to a CodeExecutionBlock
            if is_linked_to_code_execution:
                continue

            # Fix single curly braces to double curly braces
            for key in ("prompt", "format"):
                if key in input_data:
                    original_text = input_data[key]

                    # Avoid fixing already double-braced values
                    fixed_text = re.sub(
                        r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})",
                        r"{{\1}}",
                        original_text,
                    )

                    if fixed_text != original_text:
                        input_data[key] = fixed_text
                        self.add_fix_log(
                            f"Fixed {key} in node {node_id}: "
                            f"{original_text} -> {fixed_text}"
                        )

            # Check if this node has prompt_values linked to it
            if node_id in node_prompt_values:
                prompt_values = node_prompt_values[node_id]

                # Determine which field to add missing values to
                target_field = "prompt" if "prompt" in input_data else "format"

                if target_field in input_data:
                    current_text = input_data[target_field]
                    missing_values = []

                    for value_name in prompt_values:
                        pattern = r"\{\{" + re.escape(value_name) + r"\}\}"
                        if not re.search(pattern, current_text):
                            missing_values.append(value_name)

                    # Add missing values to the text
                    if missing_values:
                        additions = "\n".join(
                            [f"{{{{{value_name}}}}}" for value_name in missing_values]
                        )
                        updated_text = current_text + "\n" + additions
                        input_data[target_field] = updated_text
                        self.add_fix_log(
                            f"Added missing prompt_values to {target_field} "
                            f"in node {node_id}: {missing_values}"
                        )

        return agent

    async def fix_addtolist_blocks(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Fix AddToList blocks by adding a prerequisite empty AddToList block.

        When an AddToList block is found, this fixer:
        1. Checks if there's a CreateListBlock before it (directly or through
           StoreValueBlock)
        2. If CreateListBlock exists (direct link), removes it and its link to
           AddToList block
        3. If CreateListBlock + StoreValueBlock exists, only removes the link from
           StoreValueBlock to AddToList block
        4. Adds an empty AddToList block before the original AddToList block
        5. The first block is standalone (not connected to other blocks)
        6. The second block receives input from previous blocks and can
           self-reference
        7. Ensures the original AddToList block has a self-referencing link
        8. Prevents duplicate prerequisite blocks by checking existing connections

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        new_nodes: list[dict[str, Any]] = []
        new_links: list[dict[str, Any]] = []
        original_addtolist_node_ids: set[str] = set()
        createlist_block_id = "a912d5c7-6e00-4542-b2a9-8034136930e4"

        # First pass: identify CreateListBlock nodes and links to remove
        createlist_nodes_to_remove: set[str] = set()
        links_to_remove: list[dict[str, Any]] = []

        for link in links:
            source_node = next(
                (node for node in nodes if node.get("id") == link.get("source_id")),
                None,
            )
            sink_node = next(
                (node for node in nodes if node.get("id") == link.get("sink_id")),
                None,
            )

            # Case 1: CreateListBlock directly linked to AddToList block
            if (
                source_node
                and sink_node
                and source_node.get("block_id") == createlist_block_id
                and sink_node.get("block_id") == self.ADDTOLIST_BLOCK_ID
            ):
                createlist_nodes_to_remove.add(source_node.get("id"))
                links_to_remove.append(link)
                self.add_fix_log(
                    f"Identified CreateListBlock {source_node.get('id')} linked "
                    f"to AddToList block {sink_node.get('id')} for removal"
                )

            # Case 2: StoreValueBlock linked to AddToList block
            if (
                source_node
                and sink_node
                and source_node.get("block_id") == self.STORE_VALUE_BLOCK_ID
                and sink_node.get("block_id") == self.ADDTOLIST_BLOCK_ID
            ):
                storevalue_id = source_node.get("id")
                has_createlist_before = False
                for prev_link in links:
                    if prev_link.get("sink_id") == storevalue_id:
                        prev_source_node = next(
                            (
                                node
                                for node in nodes
                                if node.get("id") == prev_link.get("source_id")
                            ),
                            None,
                        )
                        if (
                            prev_source_node
                            and prev_source_node.get("block_id") == createlist_block_id
                        ):
                            has_createlist_before = True
                            break

                if has_createlist_before:
                    links_to_remove.append(link)
                    self.add_fix_log(
                        f"Identified StoreValueBlock {storevalue_id} (with "
                        f"CreateListBlock before it) linked to AddToList block "
                        f"{sink_node.get('id')} - removing only the link"
                    )

        # Second pass: process nodes, skipping CreateListBlock nodes to remove
        prerequisite_counter = 0
        for node in nodes:
            if node.get("id") in createlist_nodes_to_remove:
                continue

            if node.get("block_id") == self.ADDTOLIST_BLOCK_ID:
                original_addtolist_node_ids.add(node.get("id"))
                original_node_id = node.get("id")
                original_node_position = node.get("metadata", {}).get("position", {})
                if original_node_position:
                    original_node_position_x = original_node_position.get("x", 0)
                    original_node_position_y = original_node_position.get("y", 0)
                else:
                    original_node_position_x = 0
                    original_node_position_y = 0

                # Check if there's already a prerequisite AddToList block
                has_prerequisite_block = False
                for link in links:
                    if (
                        link.get("sink_id") == original_node_id
                        and link.get("sink_name") == "list"
                        and link.get("source_name") == "updated_list"
                    ):
                        source_node = next(
                            (n for n in nodes if n.get("id") == link.get("source_id")),
                            None,
                        )
                        if (
                            source_node
                            and source_node.get("block_id") == self.ADDTOLIST_BLOCK_ID
                            and source_node.get("id") != original_node_id
                        ):
                            has_prerequisite_block = True
                            break

                # Check if this node is already a prerequisite block
                is_prerequisite_block = (
                    node.get("input_default", {}).get("list") == []
                    and node.get("input_default", {}).get("entry") is None
                    and node.get("input_default", {}).get("entries") == []
                    and not any(
                        link.get("sink_id") == original_node_id
                        and link.get("sink_name") == "list"
                        for link in links
                    )
                )

                if is_prerequisite_block:
                    self.add_fix_log(
                        f"Skipped adding prerequisite AddToList block for "
                        f"{original_node_id} - this is already a prerequisite "
                        f"block"
                    )
                elif has_prerequisite_block:
                    self.add_fix_log(
                        f"Skipped adding prerequisite AddToList block for "
                        f"{original_node_id} - already has prerequisite block "
                        f"exists"
                    )
                else:
                    # Before adding prerequisite block, remove all links to
                    # the "list" input (except self-referencing)
                    links_to_list_input = []
                    for link in links:
                        if (
                            link.get("sink_id") == original_node_id
                            and link.get("sink_name") == "list"
                            and link.get("source_id") != original_node_id
                        ):
                            links_to_list_input.append(link)

                    for link in links_to_list_input:
                        if link not in links_to_remove:
                            links_to_remove.append(link)
                            self.add_fix_log(
                                f"Removed link from "
                                f"{link.get('source_id')}:"
                                f"{link.get('source_name')} to AddToList "
                                f"block {original_node_id} 'list' input "
                                f"(will be replaced by prerequisite block)"
                            )

                    prerequisite_node_id = self.generate_uuid()

                    prerequisite_node = {
                        "id": prerequisite_node_id,
                        "block_id": self.ADDTOLIST_BLOCK_ID,
                        "input_default": {
                            "list": [],
                            "entry": None,
                            "entries": [],
                            "position": None,
                        },
                        "metadata": {
                            "position": {
                                "x": original_node_position_x - 800,
                                "y": original_node_position_y + 800,
                            }
                        },
                        "graph_id": agent.get("id"),
                        "graph_version": 1,
                    }
                    prerequisite_counter += 1

                    prerequisite_link = {
                        "id": self.generate_uuid(),
                        "source_id": prerequisite_node_id,
                        "source_name": "updated_list",
                        "sink_id": original_node_id,
                        "sink_name": "list",
                    }

                    new_nodes.append(prerequisite_node)
                    new_links.append(prerequisite_link)

                    self.add_fix_log(
                        f"Added prerequisite AddToList block "
                        f"{prerequisite_node_id} before {original_node_id}"
                    )

            # Add the original node
            new_nodes.append(node)

        # Add all existing links except those marked for removal
        new_links.extend([link for link in links if link not in links_to_remove])

        # Check for original AddToList blocks and ensure they have
        # self-referencing links
        for node in new_nodes:
            if (
                node.get("block_id") == self.ADDTOLIST_BLOCK_ID
                and node.get("id") in original_addtolist_node_ids
            ):
                node_id = node.get("id")

                is_prerequisite_block = (
                    node.get("input_default", {}).get("list") == []
                    and node.get("input_default", {}).get("entry") is None
                    and node.get("input_default", {}).get("entries") == []
                    and not any(
                        link.get("sink_id") == node_id
                        and link.get("sink_name") == "list"
                        for link in new_links
                    )
                )

                if is_prerequisite_block:
                    self.add_fix_log(
                        f"Skipped adding self-referencing link for "
                        f"prerequisite AddToList block {node_id}"
                    )
                    continue

                has_self_reference = any(
                    link["source_id"] == node_id
                    and link["sink_id"] == node_id
                    and link["source_name"] == "updated_list"
                    and link["sink_name"] == "list"
                    for link in new_links
                )

                if not has_self_reference:
                    self_reference_link = {
                        "id": self.generate_uuid(),
                        "source_id": node_id,
                        "source_name": "updated_list",
                        "sink_id": node_id,
                        "sink_name": "list",
                    }
                    new_links.append(self_reference_link)
                    self.add_fix_log(
                        f"Added self-referencing link for original "
                        f"AddToList block {node_id}"
                    )

        # Update the agent with new nodes and links
        agent["nodes"] = new_nodes
        agent["links"] = new_links

        return agent

    async def fix_addtodictionary_blocks(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Fix AddToDictionary blocks by removing empty CreateDictionaryBlock nodes
        that are linked to them.

        When an AddToDictionary block is found, this fixer:
        1. Checks if there's a CreateDictionaryBlock before it
        2. If CreateDictionaryBlock exists and is linked to AddToDictionary block,
           removes it and its link
        3. The AddToDictionary block will work with an empty dictionary as default

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        createlist_block_id = (
            "b924ddf4-de4f-4b56-9a85-358930dcbc91"  # CreateDictionaryBlock ID
        )

        # First pass: identify CreateDictionaryBlock nodes linked to
        # AddToDictionary blocks
        createlist_nodes_to_remove: set[str] = set()
        links_to_remove: list[dict[str, Any]] = []

        for link in links:
            source_node = next(
                (node for node in nodes if node.get("id") == link.get("source_id")),
                None,
            )
            sink_node = next(
                (node for node in nodes if node.get("id") == link.get("sink_id")),
                None,
            )

            if (
                source_node
                and sink_node
                and source_node.get("block_id") == createlist_block_id
                and sink_node.get("block_id") == self.ADDTODICTIONARY_BLOCK_ID
            ):
                createlist_nodes_to_remove.add(source_node.get("id"))
                links_to_remove.append(link)
                self.add_fix_log(
                    f"Identified CreateDictionaryBlock "
                    f"{source_node.get('id')} linked to AddToDictionary "
                    f"block {sink_node.get('id')} for removal"
                )

        # Second pass: process nodes, skipping CreateDictionaryBlock nodes
        new_nodes = []
        for node in nodes:
            if node.get("id") in createlist_nodes_to_remove:
                continue
            new_nodes.append(node)

        # Remove the links that were marked for removal
        new_links = [link for link in links if link not in links_to_remove]

        # Update the agent with new nodes and links
        agent["nodes"] = new_nodes
        agent["links"] = new_links

        return agent

    async def fix_link_static_properties(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Fix the is_static property of links based on the source block's
        staticOutput property.

        If source block's staticOutput is true, link's is_static should be true.
        If source block's staticOutput is false, link's is_static should be false.

        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks with their schemas

        Returns:
            The fixed agent dictionary
        """
        # Create a mapping of block_id to block for quick lookup
        block_map = {block.get("id"): block for block in blocks}

        for link in agent.get("links", []):
            # Find the source node
            source_node = next(
                (
                    node
                    for node in agent.get("nodes", [])
                    if node["id"] == link["source_id"]
                ),
                None,
            )
            if not source_node:
                continue

            # Get the source block
            source_block = block_map.get(source_node.get("block_id"))
            if not source_block:
                continue

            # Check if the source block has staticOutput property
            static_output = source_block.get("staticOutput", False)

            # Update the link's is_static property
            old_is_static = link.get("is_static", False)
            link["is_static"] = static_output

            if old_is_static != static_output:
                self.add_fix_log(
                    f"Fixed link {link.get('id')} is_static: "
                    f"{old_is_static} -> {static_output} (based on source "
                    f"block {source_node.get('block_id')} staticOutput: "
                    f"{static_output})"
                )

        return agent

    async def fix_code_execution_output(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Fix CodeExecutionBlock output by changing source_name from "response"
        to "stdout_logs" in links.

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """

        links = agent.get("links", [])

        for link in links:
            source_node = next(
                (
                    node
                    for node in agent.get("nodes", [])
                    if node.get("id") == link.get("source_id")
                ),
                None,
            )

            if (
                source_node
                and source_node.get("block_id") == self.CODE_EXECUTION_BLOCK_ID
                and link.get("source_name") == "response"
            ):
                old_source_name = link.get("source_name")
                link["source_name"] = "stdout_logs"
                self.add_fix_log(
                    f"Fixed CodeExecutionBlock link {link.get('id')}: "
                    f"source_name {old_source_name} -> stdout_logs"
                )

        return agent

    async def fix_data_sampling_sample_size(
        self, agent: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fix DataSamplingBlock by setting sample_size to 1 as default.
        If old value is set as default, just reset to 1.
        If old value is from another block, delete that link and set 1 as default.

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """

        nodes = agent.get("nodes", [])
        links = agent.get("links", [])
        links_to_remove: list[dict[str, Any]] = []

        for node in nodes:
            if node.get("block_id") == self.DATA_SAMPLING_BLOCK_ID:
                node_id = node.get("id")
                input_default = node.get("input_default", {})

                # Check if there's a link to the sample_size field
                has_sample_size_link = False
                for link in links:
                    if (
                        link.get("sink_id") == node_id
                        and link.get("sink_name") == "sample_size"
                    ):
                        has_sample_size_link = True
                        links_to_remove.append(link)
                        self.add_fix_log(
                            f"Removed link {link.get('id')} to "
                            f"DataSamplingBlock {node_id} sample_size "
                            f"field (will set default to 1)"
                        )

                # Set sample_size to 1 as default
                old_value = input_default.get("sample_size", None)
                input_default["sample_size"] = 1

                if has_sample_size_link:
                    self.add_fix_log(
                        f"Fixed DataSamplingBlock {node_id} sample_size: "
                        f"removed link and set default to 1"
                    )
                elif old_value != 1:
                    self.add_fix_log(
                        f"Fixed DataSamplingBlock {node_id} sample_size: "
                        f"{old_value} -> 1"
                    )

        # Remove the links that were marked for removal
        if links_to_remove:
            agent["links"] = [link for link in links if link not in links_to_remove]

        return agent

    async def fix_ai_model_parameter(
        self,
        agent: dict[str, Any],
        blocks: list[dict[str, Any]],
        default_model: str = "gpt-5.2-2025-12-11",
    ) -> dict[str, Any]:
        """
        Add or fix the model parameter on AI blocks.

        For nodes whose block has category "AI", this function ensures that the
        input_default has a "model" parameter set to one of the allowed models.
        If missing or set to an unsupported value, it is replaced with
        default_model.

        Allowed models: "gpt-5.2-2025-12-11", "claude-opus-4-6"

        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks with their schemas
            default_model: The fallback model to use (default "gpt-5.2-2025-12-11")

        Returns:
            The fixed agent dictionary
        """
        allowed_models = {"gpt-5.2-2025-12-11", "claude-opus-4-6"}

        # Create a mapping of block_id to block for quick lookup
        block_map = {block.get("id"): block for block in blocks}

        nodes = agent.get("nodes", [])
        fixed_count = 0

        for node in nodes:
            block_id = node.get("block_id")
            block = block_map.get(block_id)

            if not block:
                continue

            # Check if the block has category "AI" in its categories array
            categories = block.get("categories", [])
            is_ai_block = any(
                cat.get("category") == "AI"
                for cat in categories
                if isinstance(cat, dict)
            )

            if is_ai_block:
                node_id = node.get("id")
                input_default = node.get("input_default", {})
                current_model = input_default.get("model")

                if current_model not in allowed_models:
                    block_name = block.get("name", "Unknown AI Block")
                    if current_model is None:
                        self.add_fix_log(
                            f"Added model parameter '{default_model}' to AI "
                            f"block node {node_id} ({block_name})"
                        )
                    else:
                        self.add_fix_log(
                            f"Replaced unsupported model '{current_model}' "
                            f"with '{default_model}' on AI block node "
                            f"{node_id} ({block_name})"
                        )
                    input_default["model"] = default_model
                    node["input_default"] = input_default
                    fixed_count += 1

        if fixed_count > 0:
            logger.info(f"Fixed model parameter on {fixed_count} AI block nodes")

        return agent

    async def fix_data_type_mismatch(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Fix data type mismatches by inserting UniversalTypeConverterBlock between
        incompatible connections.

        This function:
        1. Identifies links with type mismatches using the same logic as
           validate_data_type_compatibility
        2. Inserts UniversalTypeConverterBlock nodes to convert data types
        3. Rewires the connections to go through the converter block

        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks for reference

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])

        # Create lookup dictionaries for efficiency
        block_lookup = {block["id"]: block for block in blocks}
        node_lookup = {node["id"]: node for node in nodes}

        def get_defined_property_type(schema: dict[str, Any], name: str) -> str | None:
            """Helper function to get property type from schema, handling
            nested properties."""
            if "_#_" in name:
                parent, child = name.split("_#_", 1)
                parent_schema = schema.get(parent, {})
                if "properties" in parent_schema and isinstance(
                    parent_schema["properties"], dict
                ):
                    return parent_schema["properties"].get(child, {}).get("type")
                else:
                    return None
            else:
                return schema.get(name, {}).get("type")

        def are_types_compatible(src: str, sink: str) -> bool:
            """Check if two types are compatible."""
            if {src, sink} <= {"integer", "number"}:
                return True
            return src == sink

        def get_target_type_for_conversion(
            source_type: str, sink_type: str  # noqa: ARG001
        ) -> str:
            """Determine the target type for conversion based on sink
            requirements."""
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
            return type_mapping.get(sink_type, sink_type)

        new_links: list[dict[str, Any]] = []
        nodes_to_add: list[dict[str, Any]] = []
        converter_counter = 0

        for link in links:
            source_node = node_lookup.get(link["source_id"])
            sink_node = node_lookup.get(link["sink_id"])

            if not source_node or not sink_node:
                new_links.append(link)
                continue

            source_block = block_lookup.get(source_node.get("block_id"))
            sink_block = block_lookup.get(sink_node.get("block_id"))

            if not source_block or not sink_block:
                new_links.append(link)
                continue

            source_outputs = source_block.get("outputSchema", {}).get("properties", {})
            sink_inputs = sink_block.get("inputSchema", {}).get("properties", {})

            source_type = get_defined_property_type(source_outputs, link["source_name"])
            sink_type = get_defined_property_type(sink_inputs, link["sink_name"])

            # Check if types are incompatible
            if (
                source_type
                and sink_type
                and not are_types_compatible(source_type, sink_type)
            ):
                # Create UniversalTypeConverterBlock node
                converter_node_id = self.generate_uuid()
                target_type = get_target_type_for_conversion(source_type, sink_type)

                converter_node = {
                    "id": converter_node_id,
                    "block_id": self.UNIVERSAL_TYPE_CONVERTER_BLOCK_ID,
                    "input_default": {"type": target_type},
                    "metadata": {
                        "position": {
                            "x": converter_counter * 250,
                            "y": 100,
                        }
                    },
                    "graph_id": agent.get("id"),
                    "graph_version": 1,
                }
                nodes_to_add.append(converter_node)
                converter_counter += 1

                # Create new links: source -> converter -> sink
                source_to_converter_link = {
                    "id": self.generate_uuid(),
                    "source_id": link["source_id"],
                    "source_name": link["source_name"],
                    "sink_id": converter_node_id,
                    "sink_name": "value",
                }

                converter_to_sink_link = {
                    "id": self.generate_uuid(),
                    "source_id": converter_node_id,
                    "source_name": "value",
                    "sink_id": link["sink_id"],
                    "sink_name": link["sink_name"],
                }

                new_links.append(source_to_converter_link)
                new_links.append(converter_to_sink_link)

                source_block_name = source_block.get("name", "Unknown Block")
                sink_block_name = sink_block.get("name", "Unknown Block")
                self.add_fix_log(
                    f"Fixed data type mismatch: Inserted "
                    f"UniversalTypeConverterBlock {converter_node_id} "
                    f"between {source_block_name} ({source_type}) and "
                    f"{sink_block_name} ({sink_type}) converting to "
                    f"{target_type}"
                )
            else:
                # Keep the original link if types are compatible
                new_links.append(link)

        # Update the agent with new nodes and links
        if nodes_to_add:
            nodes.extend(nodes_to_add)
            agent["nodes"] = nodes
            agent["links"] = new_links

        return agent

    async def fix_node_x_coordinates(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Fix node x-coordinates to ensure adjacent nodes (connected via links)
        have at least 800 units difference in their x-coordinates.

        For each link connecting two nodes, if the x-coordinate difference is
        less than or equal to 800, the sink node's x-coordinate will be adjusted
        to be at least 800 units to the right of the source node.

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])

        # Create a lookup dictionary for nodes by ID
        node_lookup = {node.get("id"): node for node in nodes}

        # Iterate through all links and adjust positions as needed
        for link in links:
            source_id = link.get("source_id")
            sink_id = link.get("sink_id")

            if not source_id or not sink_id:
                continue

            source_node = node_lookup.get(source_id)
            sink_node = node_lookup.get(sink_id)

            if not source_node or not sink_node:
                continue

            source_x = source_node["metadata"]["position"].get("x", 0)
            sink_x = sink_node["metadata"]["position"].get("x", 0)

            difference = abs(sink_x - source_x)
            if difference < 800:
                required_x = source_x + 800
                sink_node["metadata"]["position"]["x"] = required_x
                self.add_fix_log(
                    f"Adjusted x-coordinate for node {sink_id}: "
                    f"{sink_x} -> {required_x} (source node {source_id} "
                    f"at x={source_x}, ensuring minimum 800 unit spacing)"
                )
            else:
                continue

        return agent

    async def fix_getcurrentdate_offset(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Fix GetCurrentDateBlock offset to ensure it's a positive value.

        If the offset in input_default is negative, it will be changed to its
        absolute value (positive).

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])

        for node in nodes:
            if node.get("block_id") == self.GET_CURRENT_DATE_BLOCK_ID:
                node_id = node.get("id")
                input_default = node.get("input_default", {})

                # Check if offset exists and is negative
                if "offset" in input_default:
                    offset_value = input_default["offset"]

                    if isinstance(offset_value, (int, float)) and offset_value < 0:
                        old_offset = offset_value
                        input_default["offset"] = abs(offset_value)
                        self.add_fix_log(
                            f"Fixed GetCurrentDateBlock {node_id} offset: "
                            f"{old_offset} -> {input_default['offset']}"
                        )

        return agent

    async def fix_addtolist_gmail_self_reference(
        self, agent: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Remove self-referencing links from AddToList blocks that are connected
        to GmailSendBlock.

        When an AddToList block has a link to a GmailSendBlock, this fixer:
        1. Identifies the AddToList block that is linked to a GmailSendBlock
        2. Removes the self-referencing link (updated_list -> list) from that
           AddToList block

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])

        # Find AddToList blocks that are connected to GmailSendBlock
        addtolist_nodes_linked_to_gmail: set[str] = set()

        for link in links:
            source_node = next(
                (node for node in nodes if node.get("id") == link.get("source_id")),
                None,
            )
            sink_node = next(
                (node for node in nodes if node.get("id") == link.get("sink_id")),
                None,
            )

            if (
                source_node
                and sink_node
                and source_node.get("block_id") == self.ADDTOLIST_BLOCK_ID
                and sink_node.get("block_id") == self.GMAIL_SEND_BLOCK_ID
            ):
                addtolist_nodes_linked_to_gmail.add(source_node.get("id"))
                self.add_fix_log(
                    f"Identified AddToList block {source_node.get('id')} "
                    f"linked to GmailSendBlock {sink_node.get('id')}"
                )

        # Remove self-referencing links from identified AddToList blocks
        if addtolist_nodes_linked_to_gmail:
            links_to_remove = []

            for link in links:
                if (
                    link.get("source_id") in addtolist_nodes_linked_to_gmail
                    and link.get("sink_id") in addtolist_nodes_linked_to_gmail
                    and link.get("source_id") == link.get("sink_id")
                    and link.get("source_name") == "updated_list"
                    and link.get("sink_name") == "list"
                ):
                    links_to_remove.append(link)
                    self.add_fix_log(
                        f"Removed self-referencing link {link.get('id')} "
                        f"from AddToList block {link.get('source_id')} "
                        f"(linked to GmailSendBlock)"
                    )

            if links_to_remove:
                agent["links"] = [link for link in links if link not in links_to_remove]

        return agent

    async def fix_text_replace_new_parameter(
        self, agent: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fix TextReplaceBlock's 'new' parameter by changing empty string to space.

        When a TextReplaceBlock node has an empty string ("") in its 'new'
        parameter within input_default, this fixer changes it to a space (" ").

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])

        for node in nodes:
            if node.get("block_id") == self.TEXT_REPLACE_BLOCK_ID:
                node_id = node.get("id")
                input_default = node.get("input_default", {})

                if "new" in input_default and input_default["new"] == "":
                    input_default["new"] = " "
                    self.add_fix_log(
                        f"Fixed TextReplaceBlock {node_id} 'new' parameter: "
                        f'empty string ("") -> space (" ")'
                    )

        return agent

    async def fix_credentials(self, agent: dict[str, Any]) -> dict[str, Any]:
        """
        Delete credentials from input_default if it exists.

        When a node's input_default contains a 'credentials' object,
        this fixer removes the entire 'credentials' field.

        Args:
            agent: The agent dictionary to fix

        Returns:
            The fixed agent dictionary
        """
        nodes = agent.get("nodes", [])

        for node in nodes:
            node_id = node.get("id")
            input_default = node.get("input_default", {})

            if "credentials" in input_default:
                deleted_credentials = input_default.pop("credentials")
                self.add_fix_log(
                    f"Deleted credentials in node {node_id}: " f"{deleted_credentials}"
                )

        return agent

    async def fix_agent_executor_blocks(
        self,
        agent: dict[str, Any],
        library_agents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Fix AgentExecutorBlock nodes to ensure they have valid graph_id
        references.

        This function:
        1. Validates that AgentExecutorBlock nodes reference valid library agents
        2. Fills in missing graph_version if graph_id is valid
        3. Ensures input_default has required fields
        4. Clears hardcoded inputs (they should be connected via links)
        5. Logs missing required input links for the LLM to fix

        Args:
            agent: The agent dictionary to fix
            library_agents: List of library agents available for composition

        Returns:
            The fixed agent dictionary
        """
        if not library_agents:
            logger.debug(
                "fix_agent_executor_blocks: No library_agents provided, " "skipping"
            )
            return agent

        nodes = agent.get("nodes", [])
        links = agent.get("links", [])

        # Create lookup for library agents
        library_agent_lookup = {la["graph_id"]: la for la in library_agents}
        logger.info(
            f"fix_agent_executor_blocks: library_agent_lookup keys = "
            f"{list(library_agent_lookup.keys())}"
        )

        for node in nodes:
            if node.get("block_id") != self.AGENT_EXECUTOR_BLOCK_ID:
                continue

            node_id = node.get("id")
            input_default = node.get("input_default", {})

            # Check if graph_id references a library agent
            graph_id = input_default.get("graph_id")
            logger.info(
                f"fix_agent_executor_blocks: Found AgentExecutorBlock "
                f"{node_id}, graph_id={graph_id}"
            )

            if not graph_id:
                logger.warning(
                    f"fix_agent_executor_blocks: Node {node_id} has no "
                    f"graph_id, skipping"
                )
                continue

            library_agent = library_agent_lookup.get(graph_id)
            if not library_agent:
                logger.warning(
                    f"fix_agent_executor_blocks: graph_id {graph_id} not "
                    f"found in library_agents lookup"
                )
                continue

            logger.info(
                f"fix_agent_executor_blocks: Found matching library agent "
                f"'{library_agent.get('name')}', input_schema keys: "
                f"{list(library_agent.get('input_schema', {}).get('properties', {}).keys())}"
            )

            # Fill in graph_version if missing or mismatched
            expected_version = library_agent.get("graph_version")
            current_version = input_default.get("graph_version")

            if current_version != expected_version:
                input_default["graph_version"] = expected_version
                self.add_fix_log(
                    f"Fixed AgentExecutorBlock {node_id}: "
                    f"graph_version {current_version} -> {expected_version} "
                    f"(for library agent '{library_agent.get('name')}')"
                )

            # Ensure user_id is present (can be empty string, filled at runtime)
            if "user_id" not in input_default:
                input_default["user_id"] = ""
                self.add_fix_log(
                    f"Fixed AgentExecutorBlock {node_id}: Added missing " f"user_id"
                )

            # Ensure inputs is present
            if "inputs" not in input_default:
                input_default["inputs"] = {}

            # Ensure input_schema is present and valid (copy from library agent)
            current_input_schema = input_default.get("input_schema")
            lib_input_schema = library_agent.get("input_schema", {})
            if not isinstance(lib_input_schema, dict):
                lib_input_schema = {}
            if not isinstance(current_input_schema, dict):
                current_input_schema = {}
            logger.info(
                f"fix_agent_executor_blocks: current_input_schema="
                f"{current_input_schema}"
            )
            logger.info(
                f"fix_agent_executor_blocks: lib_input_schema keys="
                f"{list(lib_input_schema.get('properties', {}).keys()) if lib_input_schema else 'None'}"
            )
            if not current_input_schema or not current_input_schema.get("properties"):
                input_default["input_schema"] = lib_input_schema
                logger.info(
                    "fix_agent_executor_blocks: Replaced input_schema "
                    "with library agent's schema"
                )
                if not current_input_schema:
                    self.add_fix_log(
                        f"Fixed AgentExecutorBlock {node_id}: Added "
                        f"missing input_schema"
                    )
                else:
                    self.add_fix_log(
                        f"Fixed AgentExecutorBlock {node_id}: Replaced "
                        f"empty input_schema with library agent's schema"
                    )

            # Populate inputs object with default values from input_schema
            # properties. This matches how the frontend creates
            # AgentExecutorBlock nodes.
            final_input_schema = input_default.get("input_schema", {})
            schema_properties = final_input_schema.get("properties", {})
            inputs_obj = input_default.get("inputs", {})
            if isinstance(schema_properties, dict) and isinstance(inputs_obj, dict):
                for prop_name, prop_schema in schema_properties.items():
                    if prop_name not in inputs_obj and isinstance(prop_schema, dict):
                        default_value = prop_schema.get("default")
                        if default_value is not None:
                            inputs_obj[prop_name] = default_value
                input_default["inputs"] = inputs_obj

            # Ensure output_schema is present and valid (copy from library
            # agent)
            current_output_schema = input_default.get("output_schema")
            lib_output_schema = library_agent.get("output_schema", {})
            if not isinstance(lib_output_schema, dict):
                lib_output_schema = {}
            if not isinstance(current_output_schema, dict):
                current_output_schema = {}
            if not current_output_schema or not current_output_schema.get("properties"):
                input_default["output_schema"] = lib_output_schema
                if not current_output_schema:
                    self.add_fix_log(
                        f"Fixed AgentExecutorBlock {node_id}: Added "
                        f"missing output_schema"
                    )
                else:
                    self.add_fix_log(
                        f"Fixed AgentExecutorBlock {node_id}: Replaced "
                        f"empty output_schema with library agent's schema"
                    )

            # Check for missing required input links and fix sink_name format
            sub_agent_input_schema = library_agent.get("input_schema", {})
            if not isinstance(sub_agent_input_schema, dict):
                sub_agent_input_schema = {}
            sub_agent_required_inputs = sub_agent_input_schema.get("required", [])

            # Get all linked inputs to this node
            sub_agent_properties = sub_agent_input_schema.get("properties", {})
            linked_sub_agent_inputs: set[str] = set()
            for link in links:
                if link.get("sink_id") == node_id:
                    sink_name = link.get("sink_name", "")
                    # Fix: Convert "inputs_#_<field>" to direct property
                    # name "<field>"
                    if sink_name.startswith("inputs_#_"):
                        prop_name = sink_name[9:]  # Remove "inputs_#_"
                        if prop_name in sub_agent_properties:
                            link["sink_name"] = prop_name
                            self.add_fix_log(
                                f"Fixed AgentExecutorBlock link: "
                                f"sink_name '{sink_name}' -> "
                                f"'{prop_name}' (removed inputs_#_ "
                                f"prefix)"
                            )
                            linked_sub_agent_inputs.add(prop_name)
                    elif sink_name in sub_agent_properties:
                        linked_sub_agent_inputs.add(sink_name)

            missing_inputs = [
                inp
                for inp in sub_agent_required_inputs
                if inp not in linked_sub_agent_inputs
            ]
            if missing_inputs:
                self.add_fix_log(
                    f"AgentExecutorBlock {node_id} (sub-agent "
                    f"'{library_agent.get('name')}') needs links for "
                    f"required inputs: {missing_inputs}."
                )

            node["input_default"] = input_default

        return agent

    async def fix_invalid_nested_sink_links(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Fix invalid nested sink links (links with _#_ notation pointing to
        array indices).

        The LLM sometimes generates links like 'values_#_0' for CreateListBlock,
        which is invalid because:
        1. 'values' is an array type, not an object with named properties
        2. The _#_ notation is for accessing nested object properties, not array
           indices

        This fix removes such invalid links to prevent validation errors.

        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks with their schemas

        Returns:
            The fixed agent dictionary
        """
        if not blocks:
            return agent

        block_input_schemas = {
            block["id"]: block.get("inputSchema", {}).get("properties", {})
            for block in blocks
        }
        block_names = {
            block["id"]: block.get("name", "Unknown Block") for block in blocks
        }

        links = agent.get("links", [])
        links_to_remove: list[str] = []

        for link in links:
            sink_name = link.get("sink_name", "")

            if "_#_" in sink_name:
                parent, child = sink_name.split("_#_", 1)

                # Check if child is a numeric index (invalid for _#_ notation)
                if child.isdigit():
                    sink_node = next(
                        (
                            node
                            for node in agent.get("nodes", [])
                            if node["id"] == link["sink_id"]
                        ),
                        None,
                    )
                    if sink_node:
                        block_id = sink_node.get("block_id")
                        block_name = block_names.get(block_id, "Unknown Block")
                        self.add_fix_log(
                            f"Removing invalid nested sink link "
                            f"'{sink_name}' for block '{block_name}': "
                            f"Array indices (like '{child}') are not "
                            f"valid with _#_ notation"
                        )
                        links_to_remove.append(link["id"])
                    continue

                # Check if parent property exists and child is valid
                sink_node = next(
                    (
                        node
                        for node in agent.get("nodes", [])
                        if node["id"] == link["sink_id"]
                    ),
                    None,
                )
                if sink_node:
                    block_id = sink_node.get("block_id")
                    input_props = block_input_schemas.get(block_id, {})
                    parent_schema = input_props.get(parent)

                    # If parent doesn't exist or is an array type, remove
                    if parent_schema:
                        parent_type = parent_schema.get("type")
                        if parent_type == "array":
                            block_name = block_names.get(block_id, "Unknown Block")
                            self.add_fix_log(
                                f"Removing invalid nested sink link "
                                f"'{sink_name}' for block "
                                f"'{block_name}': '{parent}' is an "
                                f"array type, _#_ notation not "
                                f"applicable"
                            )
                            links_to_remove.append(link["id"])

        # Remove invalid links
        if links_to_remove:
            agent["links"] = [
                link for link in links if link["id"] not in links_to_remove
            ]

        return agent

    async def apply_all_fixes(
        self,
        agent: dict[str, Any],
        blocks: list[dict[str, Any]] | None = None,
        library_agents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Apply all available fixes to the agent.

        Args:
            agent: The agent dictionary to fix
            blocks: List of available blocks with their schemas (optional)
            library_agents: List of library agents for AgentExecutorBlock
                            validation

        Returns:
            The fixed agent dictionary
        """
        self.fixes_applied = []

        # Apply fixes in order
        agent = await self.fix_agent_ids(agent)
        agent = await self.fix_double_curly_braces(agent)
        agent = await self.fix_storevalue_before_condition(agent)
        agent = await self.fix_addtolist_blocks(agent)
        agent = await self.fix_addtolist_gmail_self_reference(agent)
        agent = await self.fix_addtodictionary_blocks(agent)
        agent = await self.fix_code_execution_output(agent)
        agent = await self.fix_data_sampling_sample_size(agent)
        agent = await self.fix_text_replace_new_parameter(agent)
        agent = await self.fix_credentials(agent)
        agent = await self.fix_node_x_coordinates(agent)
        agent = await self.fix_getcurrentdate_offset(agent)

        # Apply fixes that require blocks information
        if blocks:
            agent = await self.fix_invalid_nested_sink_links(agent, blocks)
            agent = await self.fix_ai_model_parameter(agent, blocks)
            agent = await self.fix_link_static_properties(agent, blocks)
            agent = await self.fix_data_type_mismatch(agent, blocks)

        # Apply fixes for AgentExecutorBlock nodes (sub-agents)
        if library_agents:
            agent = await self.fix_agent_executor_blocks(agent, library_agents)

        logger.info(f"Applied {len(self.fixes_applied)} fixes to agent")
        for fix in self.fixes_applied:
            logger.warning(f"  - {fix}")

        return agent

    def get_fixes_applied(self) -> list[str]:
        """Get a list of all fixes that were applied."""
        return self.fixes_applied.copy()

    def clear_fixes_log(self):
        """Clear the list of applied fixes."""
        self.fixes_applied = []


class AgentValidator:
    """
    A comprehensive validator for AutoGPT agents that provides detailed error
    reporting for LLM-based fixes.
    """

    def __init__(self):
        self.UUID_REGEX = re.compile(
            r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}$"
        )
        self.AGENT_EXECUTOR_BLOCK_ID = "e189baac-8c20-45a1-94a7-55177ea42565"
        self.errors: list[str] = []

    def is_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        return isinstance(value, str) and self.UUID_REGEX.match(value) is not None

    def generate_uuid(self) -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())

    def add_error(self, error_message: str):
        """Add an error message to the validation errors list."""
        self.errors.append(error_message)

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Compare two values, handling complex types like dicts and lists."""
        if type(val1) is not type(val2):
            return False
        if isinstance(val1, dict):
            return json.dumps(val1, sort_keys=True) == json.dumps(val2, sort_keys=True)
        if isinstance(val1, list):
            return json.dumps(val1, sort_keys=True) == json.dumps(val2, sort_keys=True)
        return val1 == val2

    def validate_block_existence(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> bool:
        """
        Validate that all block IDs used in the agent actually exist in the
        blocks list. Returns True if all block IDs exist, False otherwise.
        """
        valid = True

        # Create a set of all valid block IDs for fast lookup
        valid_block_ids = {block.get("id") for block in blocks if block.get("id")}

        # Check each node's block_id
        for node in agent.get("nodes", []):
            block_id = node.get("block_id")
            node_id = node.get("id")

            if not block_id:
                self.add_error(
                    f"Node '{node_id}' is missing a 'block_id' field. "
                    f"Every node must reference a valid block."
                )
                valid = False
                continue

            if block_id not in valid_block_ids:
                self.add_error(
                    f"Node '{node_id}' references block_id '{block_id}' "
                    f"which does not exist in the available blocks. "
                    f"This block may have been deprecated, removed, or "
                    f"the ID is incorrect. Please use a valid block from "
                    f"the blocks library."
                )
                valid = False

        return valid

    def validate_link_node_references(self, agent: dict[str, Any]) -> bool:
        """
        Validate that all node IDs referenced in links actually exist in the
        agent's nodes. Returns True if all link references are valid, False
        otherwise.
        """
        valid = True

        # Create a set of all valid node IDs for fast lookup
        valid_node_ids = {
            node.get("id") for node in agent.get("nodes", []) if node.get("id")
        }

        # Check each link's source_id and sink_id
        for link in agent.get("links", []):
            link_id = link.get("id", "Unknown")
            source_id = link.get("source_id")
            sink_id = link.get("sink_id")
            source_name = link.get("source_name", "")
            sink_name = link.get("sink_name", "")

            # Check source_id
            if not source_id:
                self.add_error(
                    f"Link '{link_id}' is missing a 'source_id' field. "
                    f"Every link must reference a valid source node."
                )
                valid = False
            elif source_id not in valid_node_ids:
                self.add_error(
                    f"Link '{link_id}' references source_id '{source_id}' "
                    f"which does not exist in the agent's nodes. The link "
                    f"from '{source_name}' cannot be established because "
                    f"the source node is missing."
                )
                valid = False

            # Check sink_id
            if not sink_id:
                self.add_error(
                    f"Link '{link_id}' is missing a 'sink_id' field. "
                    f"Every link must reference a valid sink (destination) "
                    f"node."
                )
                valid = False
            elif sink_id not in valid_node_ids:
                self.add_error(
                    f"Link '{link_id}' references sink_id '{sink_id}' "
                    f"which does not exist in the agent's nodes. The link "
                    f"to '{sink_name}' cannot be established because the "
                    f"destination node is missing."
                )
                valid = False

        return valid

    def validate_required_inputs(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> bool:
        """
        Validate that all required inputs are provided for each node.
        Returns True if all required inputs are satisfied, False otherwise.
        """
        valid = True

        for node in agent.get("nodes", []):
            block_id = node.get("block_id")
            block = next((b for b in blocks if b.get("id") == block_id), None)

            if not block:
                continue

            required_inputs = block.get("inputSchema", {}).get("required", [])
            input_defaults = node.get("input_default", {})
            node_id = node.get("id")

            linked_inputs = set(
                link["sink_name"]
                for link in agent.get("links", [])
                if link.get("sink_id") == node_id
            )

            for req_input in required_inputs:
                if (
                    req_input not in input_defaults
                    and req_input not in linked_inputs
                    and req_input != "credentials"
                ):
                    block_name = block.get("name", "Unknown Block")
                    self.add_error(
                        f"Node '{node_id}' (block '{block_name}' - "
                        f"{block_id}) is missing required input "
                        f"'{req_input}'. This input must be either "
                        f"provided as a default value in the node's "
                        f"'input_default' field or connected via a link "
                        f"from another node's output."
                    )
                    valid = False

        return valid

    def validate_data_type_compatibility(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> bool:
        """
        Validate that linked data types are compatible between source and sink.
        Returns True if all data types are compatible, False otherwise.
        """
        valid = True

        for link in agent.get("links", []):
            source_node = next(
                (
                    node
                    for node in agent.get("nodes", [])
                    if node["id"] == link["source_id"]
                ),
                None,
            )
            sink_node = next(
                (
                    node
                    for node in agent.get("nodes", [])
                    if node["id"] == link["sink_id"]
                ),
                None,
            )

            if not source_node or not sink_node:
                continue

            source_block = next(
                (b for b in blocks if b.get("id") == source_node.get("block_id")),
                None,
            )
            sink_block = next(
                (b for b in blocks if b.get("id") == sink_node.get("block_id")),
                None,
            )

            if not source_block or not sink_block:
                continue

            source_outputs = source_block.get("outputSchema", {}).get("properties", {})
            sink_inputs = sink_block.get("inputSchema", {}).get("properties", {})

            def get_defined_property_type(
                schema: dict[str, Any], name: str
            ) -> str | None:
                if "_#_" in name:
                    parent, child = name.split("_#_", 1)
                    parent_schema = schema.get(parent, {})
                    if "properties" in parent_schema and isinstance(
                        parent_schema["properties"], dict
                    ):
                        return parent_schema["properties"].get(child, {}).get("type")
                    else:
                        return None
                else:
                    return schema.get(name, {}).get("type")

            source_type = get_defined_property_type(source_outputs, link["source_name"])
            sink_type = get_defined_property_type(sink_inputs, link["sink_name"])

            def are_types_compatible(src: str, sink: str) -> bool:
                if {src, sink} <= {"integer", "number"}:
                    return True
                return src == sink

            if (
                source_type
                and sink_type
                and not are_types_compatible(source_type, sink_type)
            ):
                source_block_name = source_block.get("name", "Unknown Block")
                sink_block_name = sink_block.get("name", "Unknown Block")
                self.add_error(
                    f"Data type mismatch in link '{link.get('id')}': "
                    f"Source '{source_block_name}' output "
                    f"'{link['source_name']}' outputs '{source_type}' "
                    f"type, but sink '{sink_block_name}' input "
                    f"'{link['sink_name']}' expects '{sink_type}' type. "
                    f"These types must match for the connection to work "
                    f"properly."
                )
                valid = False

        return valid

    def validate_nested_sink_links(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> bool:
        """
        Validate nested sink links (links with _#_ notation).
        Returns True if all nested links are valid, False otherwise.
        """
        valid = True
        block_input_schemas = {
            block["id"]: block.get("inputSchema", {}).get("properties", {})
            for block in blocks
        }
        block_names = {
            block["id"]: block.get("name", "Unknown Block") for block in blocks
        }

        for link in agent.get("links", []):
            sink_name = link["sink_name"]

            if "_#_" in sink_name:
                parent, child = sink_name.split("_#_", 1)

                sink_node = next(
                    (
                        node
                        for node in agent.get("nodes", [])
                        if node["id"] == link["sink_id"]
                    ),
                    None,
                )
                if not sink_node:
                    continue

                block_id = sink_node.get("block_id")
                input_props = block_input_schemas.get(block_id, {})

                parent_schema = input_props.get(parent)
                if not parent_schema:
                    block_name = block_names.get(block_id, "Unknown Block")
                    self.add_error(
                        f"Invalid nested sink link '{sink_name}' for "
                        f"node '{link['sink_id']}' (block "
                        f"'{block_name}' - {block_id}): Parent property "
                        f"'{parent}' does not exist in the block's "
                        f"input schema."
                    )
                    valid = False
                    continue

                # Check if additionalProperties is allowed either directly
                # or via anyOf
                allows_additional_properties = parent_schema.get(
                    "additionalProperties", False
                )

                # Check anyOf for additionalProperties
                if not allows_additional_properties and "anyOf" in parent_schema:
                    any_of_schemas = parent_schema.get("anyOf", [])
                    if isinstance(any_of_schemas, list):
                        for schema_option in any_of_schemas:
                            if isinstance(schema_option, dict) and schema_option.get(
                                "additionalProperties"
                            ):
                                allows_additional_properties = True
                                break

                if not allows_additional_properties:
                    if not (
                        isinstance(parent_schema, dict)
                        and "properties" in parent_schema
                        and isinstance(parent_schema["properties"], dict)
                        and child in parent_schema["properties"]
                    ):
                        block_name = block_names.get(block_id, "Unknown Block")
                        self.add_error(
                            f"Invalid nested sink link '{sink_name}' "
                            f"for node '{link['sink_id']}' (block "
                            f"'{block_name}' - {block_id}): Child "
                            f"property '{child}' does not exist in "
                            f"parent '{parent}' schema. Available "
                            f"properties: "
                            f"{list(parent_schema.get('properties', {}).keys())}"
                        )
                        valid = False

        return valid

    def validate_prompt_double_curly_braces_spaces(self, agent: dict[str, Any]) -> bool:
        """
        Validate that prompt parameters do not contain spaces in double curly
        braces.

        Checks the 'prompt' parameter in input_default of each node and reports
        errors if values within double curly braces ({{...}}) contain spaces.
        For example, {{user name}} should be {{user_name}}.

        Args:
            agent: The agent dictionary to validate

        Returns:
            True if all prompts are valid (no spaces in double curly braces),
            False otherwise
        """
        valid = True
        nodes = agent.get("nodes", [])

        for node in nodes:
            node_id = node.get("id")
            input_default = node.get("input_default", {})

            # Check if 'prompt' parameter exists
            if "prompt" not in input_default:
                continue

            prompt_text = input_default["prompt"]

            # Only process if it's a string
            if not isinstance(prompt_text, str):
                continue

            # Find all double curly brace patterns with spaces
            matches = re.finditer(r"\{\{([^}]+)\}\}", prompt_text)

            for match in matches:
                content = match.group(1)
                if " " in content:
                    start_pos = match.start()
                    snippet_start = max(0, start_pos - 30)
                    snippet_end = min(len(prompt_text), match.end() + 30)
                    snippet = prompt_text[snippet_start:snippet_end]

                    self.add_error(
                        f"Node '{node_id}' has spaces in double curly "
                        f"braces in prompt parameter: "
                        f"'{{{{{content}}}}}' should be "
                        f"'{{{{{content.replace(' ', '_')}}}}}'. "
                        f"Context: ...{snippet}..."
                    )
                    valid = False

        return valid

    def validate_source_output_existence(
        self, agent: dict[str, Any], blocks: list[dict[str, Any]]
    ) -> bool:
        """
        Validate that all source_names in links exist in the corresponding
        block's output schema.

        Checks that for each link, the source_name field references a valid
        output property in the source block's outputSchema. Also handles nested
        outputs with _#_ notation.

        Args:
            agent: The agent dictionary to validate
            blocks: List of available blocks with their schemas

        Returns:
            True if all source output fields exist, False otherwise
        """
        valid = True

        # Create lookup dictionaries for efficiency
        block_output_schemas = {
            block["id"]: block.get("outputSchema", {}).get("properties", {})
            for block in blocks
        }
        block_names = {
            block["id"]: block.get("name", "Unknown Block") for block in blocks
        }

        for link in agent.get("links", []):
            source_id = link.get("source_id")
            source_name = link.get("source_name")
            link_id = link.get("id", "Unknown")

            # Find the source node
            source_node = next(
                (
                    node
                    for node in agent.get("nodes", [])
                    if node.get("id") == source_id
                ),
                None,
            )
            if not source_node:
                # This error is already caught by
                # validate_link_node_references
                continue

            block_id = source_node.get("block_id")
            block_name = block_names.get(block_id, "Unknown Block")

            # Special handling for AgentExecutorBlock - use dynamic
            # output_schema from input_default
            if block_id == self.AGENT_EXECUTOR_BLOCK_ID:
                input_default = source_node.get("input_default", {})
                dynamic_output_schema = input_default.get("output_schema", {})
                if not isinstance(dynamic_output_schema, dict):
                    dynamic_output_schema = {}
                output_props = dynamic_output_schema.get("properties", {})
                if not isinstance(output_props, dict):
                    output_props = {}
            else:
                output_props = block_output_schemas.get(block_id, {})

            # Handle nested source names (with _#_ notation)
            if "_#_" in source_name:
                parent, child = source_name.split("_#_", 1)

                parent_schema = output_props.get(parent)
                if not parent_schema:
                    self.add_error(
                        f"Invalid source output field '{source_name}' "
                        f"in link '{link_id}' from node '{source_id}' "
                        f"(block '{block_name}' - {block_id}): Parent "
                        f"property '{parent}' does not exist in the "
                        f"block's output schema."
                    )
                    valid = False
                    continue

                # Check if additionalProperties is allowed either directly
                # or via anyOf
                allows_additional_properties = parent_schema.get(
                    "additionalProperties", False
                )
                if not allows_additional_properties and "anyOf" in parent_schema:
                    any_of_schemas = parent_schema.get("anyOf", [])
                    if isinstance(any_of_schemas, list):
                        for schema_option in any_of_schemas:
                            if isinstance(schema_option, dict) and schema_option.get(
                                "additionalProperties"
                            ):
                                allows_additional_properties = True
                                break
                            # Also allow when items have
                            # additionalProperties (array of objects)
                            if (
                                isinstance(schema_option, dict)
                                and "items" in schema_option
                            ):
                                items_schema = schema_option.get("items")
                                if isinstance(items_schema, dict) and items_schema.get(
                                    "additionalProperties"
                                ):
                                    allows_additional_properties = True
                                    break

                # Only require child in properties when
                # additionalProperties is not allowed
                if not allows_additional_properties:
                    if not (
                        isinstance(parent_schema, dict)
                        and "properties" in parent_schema
                        and isinstance(parent_schema["properties"], dict)
                        and child in parent_schema["properties"]
                    ):
                        available_props = (
                            list(parent_schema.get("properties", {}).keys())
                            if isinstance(parent_schema, dict)
                            else []
                        )
                        self.add_error(
                            f"Invalid nested source output field "
                            f"'{source_name}' in link '{link_id}' from "
                            f"node '{source_id}' (block "
                            f"'{block_name}' - {block_id}): Child "
                            f"property '{child}' does not exist in "
                            f"parent '{parent}' output schema. "
                            f"Available properties: {available_props}"
                        )
                        valid = False
            else:
                # Check simple (non-nested) source name
                if source_name not in output_props:
                    available_outputs = list(output_props.keys())
                    self.add_error(
                        f"Invalid source output field '{source_name}' "
                        f"in link '{link_id}' from node '{source_id}' "
                        f"(block '{block_name}' - {block_id}): Output "
                        f"property '{source_name}' does not exist in "
                        f"the block's output schema. Available outputs: "
                        f"{available_outputs}"
                    )
                    valid = False

        return valid

    def validate_agent_executor_blocks(
        self,
        agent: dict[str, Any],
        library_agents: list[dict[str, Any]] | None = None,
    ) -> bool:
        """
        Validate AgentExecutorBlock nodes have required fields and valid
        references.

        Checks that AgentExecutorBlock nodes:
        1. Have a valid graph_id in input_default (required)
        2. If graph_id matches a known library agent, validates version
           consistency
        3. Sub-agent required inputs are connected via links (not hardcoded)

        Note: Unknown graph_ids are not treated as errors - they could be valid
        direct references to agents by their actual ID (not via library_agents).
        This is consistent with fix_agent_executor_blocks() behavior.

        Args:
            agent: The agent dictionary to validate
            library_agents: List of available library agents (for version
                            validation)

        Returns:
            True if all AgentExecutorBlock nodes are valid, False otherwise
        """
        valid = True
        nodes = agent.get("nodes", [])
        links = agent.get("links", [])

        # Create lookup for library agents
        library_agent_lookup: dict[str, dict[str, Any]] = {}
        if library_agents:
            library_agent_lookup = {la["graph_id"]: la for la in library_agents}

        for node in nodes:
            if node.get("block_id") != self.AGENT_EXECUTOR_BLOCK_ID:
                continue

            node_id = node.get("id")
            input_default = node.get("input_default", {})

            # Check for required graph_id
            graph_id = input_default.get("graph_id")
            if not graph_id:
                self.add_error(
                    f"AgentExecutorBlock node '{node_id}' is missing "
                    f"required 'graph_id' in input_default. This field "
                    f"must reference the ID of the sub-agent to execute."
                )
                valid = False
                continue

            # If graph_id is not in library_agent_lookup, skip validation
            if graph_id not in library_agent_lookup:
                continue

            # Validate version consistency for known library agents
            library_agent = library_agent_lookup[graph_id]
            expected_version = library_agent.get("graph_version")
            current_version = input_default.get("graph_version")
            if (
                current_version
                and expected_version
                and current_version != expected_version
            ):
                self.add_error(
                    f"AgentExecutorBlock node '{node_id}' has mismatched "
                    f"graph_version: got {current_version}, expected "
                    f"{expected_version} for library agent "
                    f"'{library_agent.get('name')}'"
                )
                valid = False

            # Validate sub-agent inputs are properly linked (not hardcoded)
            sub_agent_input_schema = library_agent.get("input_schema", {})
            if not isinstance(sub_agent_input_schema, dict):
                sub_agent_input_schema = {}
            sub_agent_required_inputs = sub_agent_input_schema.get("required", [])
            sub_agent_properties = sub_agent_input_schema.get("properties", {})

            # Get all linked inputs to this node
            linked_sub_agent_inputs: set[str] = set()
            for link in links:
                if link.get("sink_id") == node_id:
                    sink_name = link.get("sink_name", "")
                    if sink_name in sub_agent_properties:
                        linked_sub_agent_inputs.add(sink_name)

            # Check for hardcoded inputs that should be linked
            hardcoded_inputs = input_default.get("inputs", {})
            input_schema = input_default.get("input_schema", {})
            schema_properties = (
                input_schema.get("properties", {})
                if isinstance(input_schema, dict)
                else {}
            )
            if isinstance(hardcoded_inputs, dict) and hardcoded_inputs:
                for input_name, value in hardcoded_inputs.items():
                    if input_name not in sub_agent_properties:
                        continue
                    if value is None:
                        continue
                    # Skip if this input is already linked
                    if input_name in linked_sub_agent_inputs:
                        continue
                    prop_schema = schema_properties.get(input_name, {})
                    schema_default = (
                        prop_schema.get("default")
                        if isinstance(prop_schema, dict)
                        else None
                    )
                    if schema_default is not None and self._values_equal(
                        value, schema_default
                    ):
                        continue
                    # This is a non-default hardcoded value without a link
                    self.add_error(
                        f"AgentExecutorBlock node '{node_id}' has "
                        f"hardcoded input '{input_name}' = "
                        f"{repr(value)[:50]}. Sub-agent inputs should "
                        f"be connected via links using '{input_name}' "
                        f"as sink_name, not hardcoded in "
                        f"input_default.inputs. Create a link from the "
                        f"appropriate source node."
                    )
                    valid = False

            # Check for missing required sub-agent inputs
            for req_input in sub_agent_required_inputs:
                if req_input not in linked_sub_agent_inputs:
                    self.add_error(
                        f"AgentExecutorBlock node '{node_id}' is "
                        f"missing required sub-agent input "
                        f"'{req_input}'. Create a link to this node "
                        f"using sink_name '{req_input}' to connect "
                        f"the input."
                    )
                    valid = False

        return valid

    def validate_agent_executor_block_schemas(
        self,
        agent: dict[str, Any],
    ) -> bool:
        """
        Validate that AgentExecutorBlock nodes have valid input_schema and
        output_schema.

        This validation runs regardless of library_agents availability and
        ensures that the schemas are properly populated to prevent frontend
        crashes.

        Args:
            agent: The agent dictionary to validate

        Returns:
            True if all AgentExecutorBlock nodes have valid schemas, False
            otherwise
        """
        valid = True
        nodes = agent.get("nodes", [])

        for node in nodes:
            if node.get("block_id") != self.AGENT_EXECUTOR_BLOCK_ID:
                continue

            node_id = node.get("id")
            input_default = node.get("input_default", {})
            customized_name = node.get("metadata", {}).get("customized_name", "Unknown")

            # Check input_schema
            input_schema = input_default.get("input_schema")
            if input_schema is None or not isinstance(input_schema, dict):
                self.add_error(
                    f"AgentExecutorBlock node '{node_id}' "
                    f"({customized_name}) has missing or invalid "
                    f"input_schema. The input_schema must be a valid "
                    f"JSON Schema object with 'properties' and "
                    f"'required' fields."
                )
                valid = False
            elif not input_schema.get("properties") and not input_schema.get("type"):
                # Empty schema like {} is invalid
                self.add_error(
                    f"AgentExecutorBlock node '{node_id}' "
                    f"({customized_name}) has empty input_schema. The "
                    f"input_schema must define the sub-agent's expected "
                    f"inputs. This usually indicates the sub-agent "
                    f"reference is incomplete or the library agent was "
                    f"not properly passed."
                )
                valid = False

            # Check output_schema
            output_schema = input_default.get("output_schema")
            if output_schema is None or not isinstance(output_schema, dict):
                self.add_error(
                    f"AgentExecutorBlock node '{node_id}' "
                    f"({customized_name}) has missing or invalid "
                    f"output_schema. The output_schema must be a valid "
                    f"JSON Schema object defining the sub-agent's "
                    f"outputs."
                )
                valid = False
            elif not output_schema.get("properties") and not output_schema.get("type"):
                # Empty schema like {} is invalid
                self.add_error(
                    f"AgentExecutorBlock node '{node_id}' "
                    f"({customized_name}) has empty output_schema. "
                    f"The output_schema must define the sub-agent's "
                    f"expected outputs. This usually indicates the "
                    f"sub-agent reference is incomplete or the library "
                    f"agent was not properly passed."
                )
                valid = False

        return valid

    def validate(
        self,
        agent: dict[str, Any],
        blocks: list[dict[str, Any]],
        library_agents: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str | None]:
        """
        Comprehensive validation of an agent against available blocks.

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
            - is_valid: True if agent passes all validations, False otherwise
            - error_message: Detailed error message if validation fails, None
              if successful
        """
        logger.info("Validating agent...")
        self.errors = []

        checks = [
            (
                "Block existence",
                self.validate_block_existence(agent, blocks),
            ),
            (
                "Link node references",
                self.validate_link_node_references(agent),
            ),
            (
                "Required inputs",
                self.validate_required_inputs(agent, blocks),
            ),
            (
                "Data type compatibility",
                self.validate_data_type_compatibility(agent, blocks),
            ),
            (
                "Nested sink links",
                self.validate_nested_sink_links(agent, blocks),
            ),
            (
                "Source output existence",
                self.validate_source_output_existence(agent, blocks),
            ),
            (
                "Prompt double curly braces spaces",
                self.validate_prompt_double_curly_braces_spaces(agent),
            ),
            # Always validate AgentExecutorBlock schemas to prevent
            # frontend crashes
            (
                "AgentExecutorBlock schemas",
                self.validate_agent_executor_block_schemas(agent),
            ),
        ]

        # Add AgentExecutorBlock detailed validation if library_agents
        # provided
        if library_agents:
            checks.append(
                (
                    "AgentExecutorBlock references",
                    self.validate_agent_executor_blocks(agent, library_agents),
                )
            )

        all_passed = all(check[1] for check in checks)

        if all_passed:
            logger.info("Agent validation successful.")
            return True, None
        else:
            error_message = "Agent validation failed with the following errors:\n\n"
            for i, error in enumerate(self.errors, 1):
                error_message += f"{i}. {error}\n"

            logger.error(f"Agent validation failed: {error_message}")
            return False, error_message


async def fix_and_validate(
    agent_json: dict[str, Any],
    library_agents: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], bool, str | None, list[str]]:
    """Fix and validate an agent JSON.

    Returns:
        Tuple of (fixed_agent_json, is_valid, error_message, fixes_applied)
    """
    blocks = get_blocks_as_dicts()

    fixer = AgentFixer()
    fixed_agent = await fixer.apply_all_fixes(agent_json, blocks, library_agents)
    fixes_applied = fixer.get_fixes_applied()

    validator = AgentValidator()
    is_valid, error_message = validator.validate(fixed_agent, blocks, library_agents)

    return fixed_agent, is_valid, error_message, fixes_applied
