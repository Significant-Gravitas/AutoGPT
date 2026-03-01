"""AgentValidator — validates agent JSON graphs for correctness."""

import json
import logging
import re
import uuid
from typing import Any

from .fixer import AgentFixer
from .helpers import get_blocks_as_dicts

logger = logging.getLogger(__name__)


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
                link.get("sink_name")
                for link in agent.get("links", [])
                if link.get("sink_id") == node_id and link.get("sink_name")
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
            source_id = link.get("source_id")
            sink_id = link.get("sink_id")
            source_name = link.get("source_name")
            sink_name = link.get("sink_name")

            if not all(
                isinstance(v, str) and v
                for v in (source_id, sink_id, source_name, sink_name)
            ):
                self.add_error(
                    f"Link '{link.get('id', 'Unknown')}' is missing required "
                    f"fields (source_id/sink_id/source_name/sink_name)."
                )
                valid = False
                continue

            source_node = next(
                (
                    node
                    for node in agent.get("nodes", [])
                    if node.get("id") == source_id
                ),
                None,
            )
            sink_node = next(
                (node for node in agent.get("nodes", []) if node.get("id") == sink_id),
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

            source_type = get_defined_property_type(source_outputs, source_name)
            sink_type = get_defined_property_type(sink_inputs, sink_name)

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
                    f"'{link.get('source_name', '')}' outputs '{source_type}' "
                    f"type, but sink '{sink_block_name}' input "
                    f"'{link.get('sink_name', '')}' expects '{sink_type}' type. "
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
            block.get("id", ""): block.get("inputSchema", {}).get("properties", {})
            for block in blocks
        }
        block_names = {
            block.get("id", ""): block.get("name", "Unknown Block") for block in blocks
        }

        for link in agent.get("links", []):
            sink_name = link.get("sink_name", "")
            sink_id = link.get("sink_id")

            if not sink_name or not sink_id:
                continue

            if "_#_" in sink_name:
                parent, child = sink_name.split("_#_", 1)

                sink_node = next(
                    (
                        node
                        for node in agent.get("nodes", [])
                        if node.get("id") == sink_id
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
                        f"node '{sink_id}' (block "
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
                            f"for node '{link.get('sink_id', '')}' (block "
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
            block.get("id", ""): block.get("outputSchema", {}).get("properties", {})
            for block in blocks
        }
        block_names = {
            block.get("id", ""): block.get("name", "Unknown Block") for block in blocks
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
            library_agent_lookup = {la.get("graph_id", ""): la for la in library_agents}

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
                        f"hardcoded input '{input_name}'. Sub-agent "
                        f"inputs should be connected via links using "
                        f"'{input_name}' as sink_name, not hardcoded "
                        f"in input_default.inputs. Create a link from "
                        f"the appropriate source node."
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
