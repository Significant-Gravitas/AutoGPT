"""Agent validator - Validates agent structure and connections."""

import logging
import re
from typing import Any

from .utils import get_blocks_info

logger = logging.getLogger(__name__)


class AgentValidator:
    """Validator for AutoGPT agents with detailed error reporting."""

    def __init__(self):
        self.errors: list[str] = []

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def validate_block_existence(
        self, agent: dict[str, Any], blocks_info: list[dict[str, Any]]
    ) -> bool:
        """Validate all block IDs exist in the blocks library."""
        valid = True
        valid_block_ids = {b.get("id") for b in blocks_info if b.get("id")}

        for node in agent.get("nodes", []):
            block_id = node.get("block_id")
            node_id = node.get("id")

            if not block_id:
                self.add_error(f"Node '{node_id}' is missing 'block_id' field.")
                valid = False
                continue

            if block_id not in valid_block_ids:
                self.add_error(
                    f"Node '{node_id}' references block_id '{block_id}' which does not exist."
                )
                valid = False

        return valid

    def validate_link_node_references(self, agent: dict[str, Any]) -> bool:
        """Validate all node IDs referenced in links exist."""
        valid = True
        valid_node_ids = {n.get("id") for n in agent.get("nodes", []) if n.get("id")}

        for link in agent.get("links", []):
            link_id = link.get("id", "Unknown")
            source_id = link.get("source_id")
            sink_id = link.get("sink_id")

            if not source_id:
                self.add_error(f"Link '{link_id}' is missing 'source_id'.")
                valid = False
            elif source_id not in valid_node_ids:
                self.add_error(
                    f"Link '{link_id}' references non-existent source_id '{source_id}'."
                )
                valid = False

            if not sink_id:
                self.add_error(f"Link '{link_id}' is missing 'sink_id'.")
                valid = False
            elif sink_id not in valid_node_ids:
                self.add_error(
                    f"Link '{link_id}' references non-existent sink_id '{sink_id}'."
                )
                valid = False

        return valid

    def validate_required_inputs(
        self, agent: dict[str, Any], blocks_info: list[dict[str, Any]]
    ) -> bool:
        """Validate required inputs are provided."""
        valid = True
        block_map = {b.get("id"): b for b in blocks_info}

        for node in agent.get("nodes", []):
            block_id = node.get("block_id")
            block = block_map.get(block_id)

            if not block:
                continue

            required_inputs = block.get("inputSchema", {}).get("required", [])
            input_defaults = node.get("input_default", {})
            node_id = node.get("id")

            # Get linked inputs
            linked_inputs = {
                link["sink_name"]
                for link in agent.get("links", [])
                if link.get("sink_id") == node_id
            }

            for req_input in required_inputs:
                if (
                    req_input not in input_defaults
                    and req_input not in linked_inputs
                    and req_input != "credentials"
                ):
                    block_name = block.get("name", "Unknown Block")
                    self.add_error(
                        f"Node '{node_id}' ({block_name}) is missing required input '{req_input}'."
                    )
                    valid = False

        return valid

    def validate_data_type_compatibility(
        self, agent: dict[str, Any], blocks_info: list[dict[str, Any]]
    ) -> bool:
        """Validate linked data types are compatible."""
        valid = True
        block_map = {b.get("id"): b for b in blocks_info}
        node_lookup = {n.get("id"): n for n in agent.get("nodes", [])}

        def get_type(schema: dict, name: str) -> str | None:
            if "_#_" in name:
                parent, child = name.split("_#_", 1)
                parent_schema = schema.get(parent, {})
                if "properties" in parent_schema:
                    return parent_schema["properties"].get(child, {}).get("type")
                return None
            return schema.get(name, {}).get("type")

        def are_compatible(src: str, sink: str) -> bool:
            if {src, sink} <= {"integer", "number"}:
                return True
            return src == sink

        for link in agent.get("links", []):
            source_node = node_lookup.get(link.get("source_id"))
            sink_node = node_lookup.get(link.get("sink_id"))

            if not source_node or not sink_node:
                continue

            source_block = block_map.get(source_node.get("block_id"))
            sink_block = block_map.get(sink_node.get("block_id"))

            if not source_block or not sink_block:
                continue

            source_outputs = source_block.get("outputSchema", {}).get("properties", {})
            sink_inputs = sink_block.get("inputSchema", {}).get("properties", {})

            source_type = get_type(source_outputs, link.get("source_name", ""))
            sink_type = get_type(sink_inputs, link.get("sink_name", ""))

            if source_type and sink_type and not are_compatible(source_type, sink_type):
                self.add_error(
                    f"Type mismatch: {source_block.get('name')} output '{link['source_name']}' "
                    f"({source_type}) -> {sink_block.get('name')} input '{link['sink_name']}' ({sink_type})."
                )
                valid = False

        return valid

    def validate_nested_sink_links(
        self, agent: dict[str, Any], blocks_info: list[dict[str, Any]]
    ) -> bool:
        """Validate nested sink links (with _#_ notation)."""
        valid = True
        block_map = {b.get("id"): b for b in blocks_info}
        node_lookup = {n.get("id"): n for n in agent.get("nodes", [])}

        for link in agent.get("links", []):
            sink_name = link.get("sink_name", "")

            if "_#_" in sink_name:
                parent, child = sink_name.split("_#_", 1)

                sink_node = node_lookup.get(link.get("sink_id"))
                if not sink_node:
                    continue

                block = block_map.get(sink_node.get("block_id"))
                if not block:
                    continue

                input_props = block.get("inputSchema", {}).get("properties", {})
                parent_schema = input_props.get(parent)

                if not parent_schema:
                    self.add_error(
                        f"Invalid nested link '{sink_name}': parent '{parent}' not found."
                    )
                    valid = False
                    continue

                if not parent_schema.get("additionalProperties"):
                    if not (
                        isinstance(parent_schema, dict)
                        and "properties" in parent_schema
                        and child in parent_schema.get("properties", {})
                    ):
                        self.add_error(
                            f"Invalid nested link '{sink_name}': child '{child}' not found in '{parent}'."
                        )
                        valid = False

        return valid

    def validate_prompt_spaces(self, agent: dict[str, Any]) -> bool:
        """Validate prompts don't have spaces in template variables."""
        valid = True

        for node in agent.get("nodes", []):
            input_default = node.get("input_default", {})
            prompt = input_default.get("prompt", "")

            if not isinstance(prompt, str):
                continue

            # Find {{...}} with spaces
            matches = re.finditer(r"\{\{([^}]+)\}\}", prompt)
            for match in matches:
                content = match.group(1)
                if " " in content:
                    self.add_error(
                        f"Node '{node.get('id')}' has spaces in template variable: "
                        f"'{{{{{content}}}}}' should be '{{{{{content.replace(' ', '_')}}}}}'."
                    )
                    valid = False

        return valid

    def validate(
        self, agent: dict[str, Any], blocks_info: list[dict[str, Any]] | None = None
    ) -> tuple[bool, str | None]:
        """Run all validations.

        Returns:
            Tuple of (is_valid, error_message)
        """
        self.errors = []

        if blocks_info is None:
            blocks_info = get_blocks_info()

        checks = [
            self.validate_block_existence(agent, blocks_info),
            self.validate_link_node_references(agent),
            self.validate_required_inputs(agent, blocks_info),
            self.validate_data_type_compatibility(agent, blocks_info),
            self.validate_nested_sink_links(agent, blocks_info),
            self.validate_prompt_spaces(agent),
        ]

        all_passed = all(checks)

        if all_passed:
            logger.info("Agent validation successful")
            return True, None

        error_message = "Agent validation failed:\n"
        for i, error in enumerate(self.errors, 1):
            error_message += f"{i}. {error}\n"

        logger.warning(f"Agent validation failed with {len(self.errors)} errors")
        return False, error_message


def validate_agent(
    agent: dict[str, Any], blocks_info: list[dict[str, Any]] | None = None
) -> tuple[bool, str | None]:
    """Convenience function to validate an agent.

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = AgentValidator()
    return validator.validate(agent, blocks_info)
