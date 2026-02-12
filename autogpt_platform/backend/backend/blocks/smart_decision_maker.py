import logging
import re
from collections import Counter
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

import backend.blocks.llm as llm
from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockInput,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
)
from backend.blocks.agent import AgentExecutorBlock
from backend.data.dynamic_fields import (
    extract_base_field_name,
    get_dynamic_field_description,
    is_dynamic_field,
    is_tool_pin,
)
from backend.data.execution import ExecutionContext
from backend.data.model import NodeExecutionStats, SchemaField
from backend.util import json
from backend.util.clients import get_database_manager_async_client
from backend.util.prompt import MAIN_OBJECTIVE_PREFIX

if TYPE_CHECKING:
    from backend.data.graph import Link, Node
    from backend.executor.manager import ExecutionProcessor

logger = logging.getLogger(__name__)


class ToolInfo(BaseModel):
    """Processed tool call information."""

    tool_call: Any  # The original tool call object from LLM response
    tool_name: str  # The function name
    tool_def: dict[str, Any]  # The tool definition from tool_functions
    input_data: dict[str, Any]  # Processed input data ready for tool execution
    field_mapping: dict[str, str]  # Field name mapping for the tool


class ExecutionParams(BaseModel):
    """Tool execution parameters."""

    user_id: str
    graph_id: str
    node_id: str
    graph_version: int
    graph_exec_id: str
    node_exec_id: str
    execution_context: "ExecutionContext"


def _get_tool_requests(entry: dict[str, Any]) -> list[str]:
    """
    Return a list of tool_call_ids if the entry is a tool request.
    Supports both OpenAI and Anthropics formats.
    """
    tool_call_ids = []
    if entry.get("role") != "assistant":
        return tool_call_ids

    # OpenAI: check for tool_calls in the entry.
    calls = entry.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if tool_id := call.get("id"):
                tool_call_ids.append(tool_id)

    # Anthropics: check content items for tool_use type.
    content = entry.get("content")
    if isinstance(content, list):
        for item in content:
            if item.get("type") != "tool_use":
                continue
            if tool_id := item.get("id"):
                tool_call_ids.append(tool_id)

    return tool_call_ids


def _get_tool_responses(entry: dict[str, Any]) -> list[str]:
    """
    Return a list of tool_call_ids if the entry is a tool response.
    Supports both OpenAI and Anthropics formats.
    """
    tool_call_ids: list[str] = []

    # OpenAI: a tool response message with role "tool" and key "tool_call_id".
    if entry.get("role") == "tool":
        if tool_call_id := entry.get("tool_call_id"):
            tool_call_ids.append(str(tool_call_id))

    # Anthropics: check content items for tool_result type.
    if entry.get("role") == "user":
        content = entry.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") != "tool_result":
                    continue
                if tool_call_id := item.get("tool_use_id"):
                    tool_call_ids.append(tool_call_id)

    return tool_call_ids


def _create_tool_response(call_id: str, output: Any) -> dict[str, Any]:
    """
    Create a tool response message for either OpenAI or Anthropics,
    based on the tool_id format.
    """
    content = output if isinstance(output, str) else json.dumps(output)

    # Anthropics format: tool IDs typically start with "toolu_"
    if call_id.startswith("toolu_"):
        return {
            "role": "user",
            "type": "message",
            "content": [
                {"tool_use_id": call_id, "type": "tool_result", "content": content}
            ],
        }

    # OpenAI format: tool IDs typically start with "call_".
    # Or default fallback (if the tool_id doesn't match any known prefix)
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _combine_tool_responses(tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Combine multiple Anthropic tool responses into a single user message.
    For non-Anthropic formats, returns the original list unchanged.
    """
    if len(tool_outputs) <= 1:
        return tool_outputs

    # Anthropic responses have role="user", type="message", and content is a list with tool_result items
    anthropic_responses = [
        output
        for output in tool_outputs
        if (
            output.get("role") == "user"
            and output.get("type") == "message"
            and isinstance(output.get("content"), list)
            and any(
                item.get("type") == "tool_result"
                for item in output.get("content", [])
                if isinstance(item, dict)
            )
        )
    ]

    if len(anthropic_responses) > 1:
        combined_content = [
            item for response in anthropic_responses for item in response["content"]
        ]

        combined_response = {
            "role": "user",
            "type": "message",
            "content": combined_content,
        }

        non_anthropic_responses = [
            output for output in tool_outputs if output not in anthropic_responses
        ]

        return [combined_response] + non_anthropic_responses

    return tool_outputs


def _convert_raw_response_to_dict(raw_response: Any) -> dict[str, Any]:
    """
    Safely convert raw_response to dictionary format for conversation history.
    Handles different response types from different LLM providers.
    """
    if isinstance(raw_response, str):
        # Ollama returns a string, convert to dict format
        return {"role": "assistant", "content": raw_response}
    elif isinstance(raw_response, dict):
        # Already a dict (from tests or some providers)
        return raw_response
    else:
        # OpenAI/Anthropic return objects, convert with json.to_dict
        return json.to_dict(raw_response)


def get_pending_tool_calls(conversation_history: list[Any] | None) -> dict[str, int]:
    """
    All the tool calls entry in the conversation history requires a response.
    This function returns the pending tool calls that has not generated an output yet.

    Return: dict[str, int] - A dictionary of pending tool call IDs with their count.
    """
    if not conversation_history:
        return {}

    pending_calls = Counter()
    for history in conversation_history:
        for call_id in _get_tool_requests(history):
            pending_calls[call_id] += 1

        for call_id in _get_tool_responses(history):
            pending_calls[call_id] -= 1

    return {call_id: count for call_id, count in pending_calls.items() if count > 0}


class SmartDecisionMakerBlock(Block):
    """
    A block that uses a language model to make smart decisions based on a given prompt.
    """

    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description="The prompt to send to the language model.",
            placeholder="Enter your prompt here...",
        )
        model: llm.LlmModel = SchemaField(
            title="LLM Model",
            default=llm.DEFAULT_LLM_MODEL,
            description="The language model to use for answering the prompt.",
            advanced=False,
        )
        credentials: llm.AICredentials = llm.AICredentialsField()
        multiple_tool_calls: bool = SchemaField(
            title="Multiple Tool Calls",
            default=False,
            description="Whether to allow multiple tool calls in a single response.",
            advanced=True,
        )
        sys_prompt: str = SchemaField(
            title="System Prompt",
            default="Thinking carefully step by step decide which function to call. "
            "Always choose a function call from the list of function signatures, "
            "and always provide the complete argument provided with the type "
            "matching the required jsonschema signature, no missing argument is allowed. "
            "If you have already completed the task objective, you can end the task "
            "by providing the end result of your work as a finish message. "
            "Function parameters that has no default value and not optional typed has to be provided. ",
            description="The system prompt to provide additional context to the model.",
        )
        conversation_history: list[dict] | None = SchemaField(
            default_factory=list,
            description="The conversation history to provide context for the prompt.",
        )
        last_tool_output: Any = SchemaField(
            default=None,
            description="The output of the last tool that was called.",
        )
        retry: int = SchemaField(
            title="Retry Count",
            default=3,
            description="Number of times to retry the LLM call if the response does not match the expected format.",
        )
        prompt_values: dict[str, str] = SchemaField(
            advanced=False,
            default_factory=dict,
            description="Values used to fill in the prompt. The values can be used in the prompt by putting them in a double curly braces, e.g. {{variable_name}}.",
        )
        max_tokens: int | None = SchemaField(
            advanced=True,
            default=None,
            description="The maximum number of tokens to generate in the chat completion.",
        )
        ollama_host: str = SchemaField(
            advanced=True,
            default="localhost:11434",
            description="Ollama host for local  models",
        )
        agent_mode_max_iterations: int = SchemaField(
            title="Agent Mode Max Iterations",
            description="Maximum iterations for agent mode. 0 = traditional mode (single LLM call, yield tool calls for external execution), -1 = infinite agent mode (loop until finished), 1+ = agent mode with max iterations limit.",
            advanced=True,
            default=0,
        )
        conversation_compaction: bool = SchemaField(
            default=True,
            title="Context window auto-compaction",
            description="Automatically compact the context window once it hits the limit",
        )

        @classmethod
        def get_missing_links(cls, data: BlockInput, links: list["Link"]) -> set[str]:
            # conversation_history & last_tool_output validation is handled differently
            missing_links = super().get_missing_links(
                data,
                [
                    link
                    for link in links
                    if link.sink_name
                    not in ["conversation_history", "last_tool_output"]
                ],
            )

            # Avoid executing the block if the last_tool_output is connected to a static
            # link, like StoreValueBlock or AgentInputBlock.
            if any(link.sink_name == "conversation_history" for link in links) and any(
                link.sink_name == "last_tool_output" and link.is_static
                for link in links
            ):
                raise ValueError(
                    "Last Tool Output can't be connected to a static (dashed line) "
                    "link like the output of `StoreValue` or `AgentInput` block"
                )

            # Check that both conversation_history and last_tool_output are connected together
            if any(link.sink_name == "conversation_history" for link in links) != any(
                link.sink_name == "last_tool_output" for link in links
            ):
                raise ValueError(
                    "Last Tool Output is needed when Conversation History is used, "
                    "and vice versa. Please connect both inputs together."
                )

            return missing_links

        @classmethod
        def get_missing_input(cls, data: BlockInput) -> set[str]:
            if missing_input := super().get_missing_input(data):
                return missing_input

            conversation_history = data.get("conversation_history", [])
            pending_tool_calls = get_pending_tool_calls(conversation_history)
            last_tool_output = data.get("last_tool_output")

            # Tool call is pending, wait for the tool output to be provided.
            if last_tool_output is None and pending_tool_calls:
                return {"last_tool_output"}

            # No tool call is pending, wait for the conversation history to be updated.
            if last_tool_output is not None and not pending_tool_calls:
                return {"conversation_history"}

            return set()

    class Output(BlockSchemaOutput):
        tools: Any = SchemaField(description="The tools that are available to use.")
        finished: str = SchemaField(
            description="The finished message to display to the user."
        )
        conversations: list[Any] = SchemaField(
            description="The conversation history to provide context for the prompt."
        )

    def __init__(self):
        super().__init__(
            id="3b191d9f-356f-482d-8238-ba04b6d18381",
            description="Uses AI to intelligently decide what tool to use.",
            categories={BlockCategory.AI},
            block_type=BlockType.AI,
            input_schema=SmartDecisionMakerBlock.Input,
            output_schema=SmartDecisionMakerBlock.Output,
            test_input={
                "prompt": "Hello, World!",
                "credentials": llm.TEST_CREDENTIALS_INPUT,
            },
            test_output=[],
            test_credentials=llm.TEST_CREDENTIALS,
        )

    @staticmethod
    def cleanup(s: str):
        """Clean up block names for use as tool function names."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", s).lower()

    @staticmethod
    async def _create_block_function_signature(
        sink_node: "Node", links: list["Link"]
    ) -> dict[str, Any]:
        """
        Creates a function signature for a block node.

        Args:
            sink_node: The node for which to create a function signature.
            links: The list of links connected to the sink node.

        Returns:
            A dictionary representing the function signature in the format expected by LLM tools.

        Raises:
            ValueError: If the block specified by sink_node.block_id is not found.
        """
        block = sink_node.block

        # Use custom name from node metadata if set, otherwise fall back to block.name
        custom_name = sink_node.metadata.get("customized_name")
        tool_name = custom_name if custom_name else block.name

        tool_function: dict[str, Any] = {
            "name": SmartDecisionMakerBlock.cleanup(tool_name),
            "description": block.description,
        }
        sink_block_input_schema = block.input_schema
        properties = {}
        field_mapping = {}  # clean_name -> original_name

        for link in links:
            field_name = link.sink_name
            is_dynamic = is_dynamic_field(field_name)
            # Clean property key to ensure Anthropic API compatibility for ALL fields
            clean_field_name = SmartDecisionMakerBlock.cleanup(field_name)
            field_mapping[clean_field_name] = field_name

            if is_dynamic:
                # For dynamic fields, use cleaned name but preserve original in description
                properties[clean_field_name] = {
                    "type": "string",
                    "description": get_dynamic_field_description(field_name),
                }
            else:
                # For regular fields, use the block's schema directly
                try:
                    properties[clean_field_name] = (
                        sink_block_input_schema.get_field_schema(field_name)
                    )
                except (KeyError, AttributeError):
                    # If field doesn't exist in schema, provide a generic one
                    properties[clean_field_name] = {
                        "type": "string",
                        "description": f"Value for {field_name}",
                    }

        # Build the parameters schema using a single unified path
        base_schema = block.input_schema.jsonschema()
        base_required = set(base_schema.get("required", []))

        # Compute required fields at the leaf level:
        # - If a linked field is dynamic and its base is required in the block schema, require the leaf
        # - If a linked field is regular and is required in the block schema, require the leaf
        required_fields: set[str] = set()
        for link in links:
            field_name = link.sink_name
            is_dynamic = is_dynamic_field(field_name)
            # Always use cleaned field name for property key (Anthropic API compliance)
            clean_field_name = SmartDecisionMakerBlock.cleanup(field_name)

            if is_dynamic:
                base_name = extract_base_field_name(field_name)
                if base_name in base_required:
                    required_fields.add(clean_field_name)
            else:
                if field_name in base_required:
                    required_fields.add(clean_field_name)

        tool_function["parameters"] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "required": sorted(required_fields),
        }

        # Store field mapping and node info for later use in output processing
        tool_function["_field_mapping"] = field_mapping
        tool_function["_sink_node_id"] = sink_node.id

        return {"type": "function", "function": tool_function}

    @staticmethod
    async def _create_agent_function_signature(
        sink_node: "Node", links: list["Link"]
    ) -> dict[str, Any]:
        """
        Creates a function signature for an agent node.

        Args:
            sink_node: The agent node for which to create a function signature.
            links: The list of links connected to the sink node.

        Returns:
            A dictionary representing the function signature in the format expected by LLM tools.

        Raises:
            ValueError: If the graph metadata for the specified graph_id and graph_version is not found.
        """
        graph_id = sink_node.input_default.get("graph_id")
        graph_version = sink_node.input_default.get("graph_version")
        if not graph_id or not graph_version:
            raise ValueError("Graph ID or Graph Version not found in sink node.")

        db_client = get_database_manager_async_client()
        sink_graph_meta = await db_client.get_graph_metadata(graph_id, graph_version)
        if not sink_graph_meta:
            raise ValueError(
                f"Sink graph metadata not found: {graph_id} {graph_version}"
            )

        # Use custom name from node metadata if set, otherwise fall back to graph name
        custom_name = sink_node.metadata.get("customized_name")
        tool_name = custom_name if custom_name else sink_graph_meta.name

        tool_function: dict[str, Any] = {
            "name": SmartDecisionMakerBlock.cleanup(tool_name),
            "description": sink_graph_meta.description,
        }

        properties = {}
        field_mapping = {}

        for link in links:
            field_name = link.sink_name

            clean_field_name = SmartDecisionMakerBlock.cleanup(field_name)
            field_mapping[clean_field_name] = field_name

            sink_block_input_schema = sink_node.input_default["input_schema"]
            sink_block_properties = sink_block_input_schema.get("properties", {}).get(
                link.sink_name, {}
            )
            description = (
                sink_block_properties["description"]
                if "description" in sink_block_properties
                else f"The {link.sink_name} of the tool"
            )
            properties[clean_field_name] = {
                "type": "string",
                "description": description,
                "default": json.dumps(sink_block_properties.get("default", None)),
            }

        tool_function["parameters"] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "strict": True,
        }

        tool_function["_field_mapping"] = field_mapping
        tool_function["_sink_node_id"] = sink_node.id

        return {"type": "function", "function": tool_function}

    @staticmethod
    async def _create_tool_node_signatures(
        node_id: str,
    ) -> list[dict[str, Any]]:
        """
        Creates function signatures for connected tools.

        Args:
            node_id: The node_id for which to create function signatures.

        Returns:
            List of function signatures for tools
        """
        db_client = get_database_manager_async_client()
        tools = [
            (link, node)
            for link, node in await db_client.get_connected_output_nodes(node_id)
            if is_tool_pin(link.source_name) and link.source_id == node_id
        ]
        if not tools:
            raise ValueError("There is no next node to execute.")

        return_tool_functions: list[dict[str, Any]] = []

        grouped_tool_links: dict[str, tuple["Node", list["Link"]]] = {}
        for link, node in tools:
            if link.sink_id not in grouped_tool_links:
                grouped_tool_links[link.sink_id] = (node, [link])
            else:
                grouped_tool_links[link.sink_id][1].append(link)

        for sink_node, links in grouped_tool_links.values():
            if not sink_node:
                raise ValueError(f"Sink node not found: {links[0].sink_id}")

            if sink_node.block_id == AgentExecutorBlock().id:
                tool_func = (
                    await SmartDecisionMakerBlock._create_agent_function_signature(
                        sink_node, links
                    )
                )
                return_tool_functions.append(tool_func)
            else:
                tool_func = (
                    await SmartDecisionMakerBlock._create_block_function_signature(
                        sink_node, links
                    )
                )
                return_tool_functions.append(tool_func)

        return return_tool_functions

    async def _attempt_llm_call_with_validation(
        self,
        credentials: llm.APIKeyCredentials,
        input_data: Input,
        current_prompt: list[dict],
        tool_functions: list[dict[str, Any]],
    ):
        """
        Attempt a single LLM call with tool validation.

        Returns the response if successful, raises ValueError if validation fails.
        """
        resp = await llm.llm_call(
            compress_prompt_to_fit=input_data.conversation_compaction,
            credentials=credentials,
            llm_model=input_data.model,
            prompt=current_prompt,
            max_tokens=input_data.max_tokens,
            tools=tool_functions,
            ollama_host=input_data.ollama_host,
            parallel_tool_calls=input_data.multiple_tool_calls,
        )

        # Track LLM usage stats per call
        self.merge_stats(
            NodeExecutionStats(
                input_token_count=resp.prompt_tokens,
                output_token_count=resp.completion_tokens,
                llm_call_count=1,
            )
        )

        if not resp.tool_calls:
            return resp
        validation_errors_list: list[str] = []
        for tool_call in resp.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except Exception as e:
                validation_errors_list.append(
                    f"Tool call '{tool_name}' has invalid JSON arguments: {e}"
                )
                continue

            # Find the tool definition to get the expected arguments
            tool_def = next(
                (
                    tool
                    for tool in tool_functions
                    if tool["function"]["name"] == tool_name
                ),
                None,
            )
            if tool_def is None:
                if len(tool_functions) == 1:
                    tool_def = tool_functions[0]
                else:
                    validation_errors_list.append(
                        f"Tool call for '{tool_name}' does not match any known "
                        "tool definition."
                    )

            # Get parameters schema from tool definition
            if (
                tool_def
                and "function" in tool_def
                and "parameters" in tool_def["function"]
            ):
                parameters = tool_def["function"]["parameters"]
                expected_args = parameters.get("properties", {})
                required_params = set(parameters.get("required", []))
            else:
                expected_args = {arg: {} for arg in tool_args.keys()}
                required_params = set()

            # Validate tool call arguments
            provided_args = set(tool_args.keys())
            expected_args_set = set(expected_args.keys())

            # Check for unexpected arguments (typos)
            unexpected_args = provided_args - expected_args_set
            # Only check for missing REQUIRED parameters
            missing_required_args = required_params - provided_args

            if unexpected_args or missing_required_args:
                error_msg = f"Tool call '{tool_name}' has parameter errors:"
                if unexpected_args:
                    error_msg += f" Unknown parameters: {sorted(unexpected_args)}."
                if missing_required_args:
                    error_msg += f" Missing required parameters: {sorted(missing_required_args)}."
                error_msg += f" Expected parameters: {sorted(expected_args_set)}."
                if required_params:
                    error_msg += f" Required parameters: {sorted(required_params)}."
                validation_errors_list.append(error_msg)

        if validation_errors_list:
            raise ValueError("; ".join(validation_errors_list))

        return resp

    def _process_tool_calls(
        self, response, tool_functions: list[dict[str, Any]]
    ) -> list[ToolInfo]:
        """Process tool calls and extract tool definitions, arguments, and input data.

        Returns a list of tool info dicts with:
        - tool_call: The original tool call object
        - tool_name: The function name
        - tool_def: The tool definition from tool_functions
        - input_data: Processed input data dict (includes None values)
        - field_mapping: Field name mapping for the tool
        """
        if not response.tool_calls:
            return []

        processed_tools = []
        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_def = next(
                (
                    tool
                    for tool in tool_functions
                    if tool["function"]["name"] == tool_name
                ),
                None,
            )
            if not tool_def:
                if len(tool_functions) == 1:
                    tool_def = tool_functions[0]
                else:
                    continue

            # Build input data for the tool
            input_data = {}
            field_mapping = tool_def["function"].get("_field_mapping", {})
            if "function" in tool_def and "parameters" in tool_def["function"]:
                expected_args = tool_def["function"]["parameters"].get("properties", {})
                for clean_arg_name in expected_args:
                    original_field_name = field_mapping.get(
                        clean_arg_name, clean_arg_name
                    )
                    arg_value = tool_args.get(clean_arg_name)
                    # Include all expected parameters, even if None (for backward compatibility with tests)
                    input_data[original_field_name] = arg_value

            processed_tools.append(
                ToolInfo(
                    tool_call=tool_call,
                    tool_name=tool_name,
                    tool_def=tool_def,
                    input_data=input_data,
                    field_mapping=field_mapping,
                )
            )

        return processed_tools

    def _update_conversation(
        self, prompt: list[dict], response, tool_outputs: list | None = None
    ):
        """Update conversation history with response and tool outputs."""
        # Don't add separate reasoning message with tool calls (breaks Anthropic's tool_use->tool_result pairing)
        assistant_message = _convert_raw_response_to_dict(response.raw_response)
        has_tool_calls = isinstance(assistant_message.get("content"), list) and any(
            item.get("type") == "tool_use"
            for item in assistant_message.get("content", [])
        )

        if response.reasoning and not has_tool_calls:
            prompt.append(
                {"role": "assistant", "content": f"[Reasoning]: {response.reasoning}"}
            )

        prompt.append(assistant_message)

        if tool_outputs:
            prompt.extend(tool_outputs)

    async def _execute_single_tool_with_manager(
        self,
        tool_info: ToolInfo,
        execution_params: ExecutionParams,
        execution_processor: "ExecutionProcessor",
    ) -> dict:
        """Execute a single tool using the execution manager for proper integration."""
        # Lazy imports to avoid circular dependencies
        from backend.data.execution import NodeExecutionEntry

        tool_call = tool_info.tool_call
        tool_def = tool_info.tool_def
        raw_input_data = tool_info.input_data

        # Get sink node and field mapping
        sink_node_id = tool_def["function"]["_sink_node_id"]

        # Use proper database operations for tool execution
        db_client = get_database_manager_async_client()

        # Get target node
        target_node = await db_client.get_node(sink_node_id)
        if not target_node:
            raise ValueError(f"Target node {sink_node_id} not found")

        # Create proper node execution using upsert_execution_input
        node_exec_result = None
        final_input_data = None

        # Add all inputs to the execution
        if not raw_input_data:
            raise ValueError(f"Tool call has no input data: {tool_call}")

        for input_name, input_value in raw_input_data.items():
            node_exec_result, final_input_data = await db_client.upsert_execution_input(
                node_id=sink_node_id,
                graph_exec_id=execution_params.graph_exec_id,
                input_name=input_name,
                input_data=input_value,
            )

        assert node_exec_result is not None, "node_exec_result should not be None"

        # Create NodeExecutionEntry for execution manager
        node_exec_entry = NodeExecutionEntry(
            user_id=execution_params.user_id,
            graph_exec_id=execution_params.graph_exec_id,
            graph_id=execution_params.graph_id,
            graph_version=execution_params.graph_version,
            node_exec_id=node_exec_result.node_exec_id,
            node_id=sink_node_id,
            block_id=target_node.block_id,
            inputs=final_input_data or {},
            execution_context=execution_params.execution_context,
        )

        # Use the execution manager to execute the tool node
        try:
            # Get NodeExecutionProgress from the execution manager's running nodes
            node_exec_progress = execution_processor.running_node_execution[
                sink_node_id
            ]

            # Use the execution manager's own graph stats
            graph_stats_pair = (
                execution_processor.execution_stats,
                execution_processor.execution_stats_lock,
            )

            # Create a completed future for the task tracking system
            node_exec_future = Future()
            node_exec_progress.add_task(
                node_exec_id=node_exec_result.node_exec_id,
                task=node_exec_future,
            )

            # Execute the node directly since we're in the SmartDecisionMaker context
            node_exec_future.set_result(
                await execution_processor.on_node_execution(
                    node_exec=node_exec_entry,
                    node_exec_progress=node_exec_progress,
                    nodes_input_masks=None,
                    graph_stats_pair=graph_stats_pair,
                )
            )

            # Get outputs from database after execution completes using database manager client
            node_outputs = await db_client.get_execution_outputs_by_node_exec_id(
                node_exec_result.node_exec_id
            )

            # Create tool response
            tool_response_content = (
                json.dumps(node_outputs)
                if node_outputs
                else "Tool executed successfully"
            )
            return _create_tool_response(tool_call.id, tool_response_content)

        except Exception as e:
            logger.error(f"Tool execution with manager failed: {e}")
            # Return error response
            return _create_tool_response(
                tool_call.id, f"Tool execution failed: {str(e)}"
            )

    async def _execute_tools_agent_mode(
        self,
        input_data,
        credentials,
        tool_functions: list[dict[str, Any]],
        prompt: list[dict],
        graph_exec_id: str,
        node_id: str,
        node_exec_id: str,
        user_id: str,
        graph_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        execution_processor: "ExecutionProcessor",
    ):
        """Execute tools in agent mode with a loop until finished."""
        max_iterations = input_data.agent_mode_max_iterations
        iteration = 0

        # Execution parameters for tool execution
        execution_params = ExecutionParams(
            user_id=user_id,
            graph_id=graph_id,
            node_id=node_id,
            graph_version=graph_version,
            graph_exec_id=graph_exec_id,
            node_exec_id=node_exec_id,
            execution_context=execution_context,
        )

        current_prompt = list(prompt)

        while max_iterations < 0 or iteration < max_iterations:
            iteration += 1
            logger.debug(f"Agent mode iteration {iteration}")

            # Prepare prompt for this iteration
            iteration_prompt = list(current_prompt)

            # On the last iteration, add a special system message to encourage completion
            if max_iterations > 0 and iteration == max_iterations:
                last_iteration_message = {
                    "role": "system",
                    "content": f"{MAIN_OBJECTIVE_PREFIX}This is your last iteration ({iteration}/{max_iterations}). "
                    "Try to complete the task with the information you have. If you cannot fully complete it, "
                    "provide a summary of what you've accomplished and what remains to be done. "
                    "Prefer finishing with a clear response rather than making additional tool calls.",
                }
                iteration_prompt.append(last_iteration_message)

            # Get LLM response
            try:
                response = await self._attempt_llm_call_with_validation(
                    credentials, input_data, iteration_prompt, tool_functions
                )
            except Exception as e:
                yield "error", f"LLM call failed in agent mode iteration {iteration}: {str(e)}"
                return

            # Process tool calls
            processed_tools = self._process_tool_calls(response, tool_functions)

            # If no tool calls, we're done
            if not processed_tools:
                yield "finished", response.response
                self._update_conversation(current_prompt, response)
                yield "conversations", current_prompt
                return

            # Execute tools and collect responses
            tool_outputs = []
            for tool_info in processed_tools:
                try:
                    tool_response = await self._execute_single_tool_with_manager(
                        tool_info, execution_params, execution_processor
                    )
                    tool_outputs.append(tool_response)
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    # Create error response for the tool
                    error_response = _create_tool_response(
                        tool_info.tool_call.id, f"Error: {str(e)}"
                    )
                    tool_outputs.append(error_response)

            tool_outputs = _combine_tool_responses(tool_outputs)

            self._update_conversation(current_prompt, response, tool_outputs)

            # Yield intermediate conversation state
            yield "conversations", current_prompt

        # If we reach max iterations, yield the current state
        if max_iterations < 0:
            yield "finished", f"Agent mode completed after {iteration} iterations"
        else:
            yield "finished", f"Agent mode completed after {max_iterations} iterations (limit reached)"
        yield "conversations", current_prompt

    async def run(
        self,
        input_data: Input,
        *,
        credentials: llm.APIKeyCredentials,
        graph_id: str,
        node_id: str,
        graph_exec_id: str,
        node_exec_id: str,
        user_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        execution_processor: "ExecutionProcessor",
        nodes_to_skip: set[str] | None = None,
        **kwargs,
    ) -> BlockOutput:

        tool_functions = await self._create_tool_node_signatures(node_id)
        original_tool_count = len(tool_functions)

        # Filter out tools for nodes that should be skipped (e.g., missing optional credentials)
        if nodes_to_skip:
            tool_functions = [
                tf
                for tf in tool_functions
                if tf.get("function", {}).get("_sink_node_id") not in nodes_to_skip
            ]

            # Only raise error if we had tools but they were all filtered out
            if original_tool_count > 0 and not tool_functions:
                raise ValueError(
                    "No available tools to execute - all downstream nodes are unavailable "
                    "(possibly due to missing optional credentials)"
                )

        yield "tool_functions", json.dumps(tool_functions)

        conversation_history = input_data.conversation_history or []
        prompt = [json.to_dict(p) for p in conversation_history if p]

        pending_tool_calls = get_pending_tool_calls(conversation_history)
        if pending_tool_calls and input_data.last_tool_output is None:
            raise ValueError(f"Tool call requires an output for {pending_tool_calls}")

        tool_output = []
        if pending_tool_calls and input_data.last_tool_output is not None:
            first_call_id = next(iter(pending_tool_calls.keys()))
            tool_output.append(
                _create_tool_response(first_call_id, input_data.last_tool_output)
            )

            prompt.extend(tool_output)
            remaining_pending_calls = get_pending_tool_calls(prompt)

            if remaining_pending_calls:
                yield "conversations", prompt
                return
        elif input_data.last_tool_output:
            logger.error(
                f"[SmartDecisionMakerBlock-node_exec_id={node_exec_id}] "
                f"No pending tool calls found. This may indicate an issue with the "
                f"conversation history, or the tool giving response more than once."
                f"This should not happen! Please check the conversation history for any inconsistencies."
            )
            tool_output.append(
                {
                    "role": "user",
                    "content": f"Last tool output: {json.dumps(input_data.last_tool_output)}",
                }
            )
            prompt.extend(tool_output)

        values = input_data.prompt_values
        if values:
            input_data.prompt = llm.fmt.format_string(input_data.prompt, values)
            input_data.sys_prompt = llm.fmt.format_string(input_data.sys_prompt, values)

        if input_data.sys_prompt and not any(
            p["role"] == "system" and p["content"].startswith(MAIN_OBJECTIVE_PREFIX)
            for p in prompt
        ):
            prompt.append(
                {
                    "role": "system",
                    "content": MAIN_OBJECTIVE_PREFIX + input_data.sys_prompt,
                }
            )

        if input_data.prompt and not any(
            p["role"] == "user" and p["content"].startswith(MAIN_OBJECTIVE_PREFIX)
            for p in prompt
        ):
            prompt.append(
                {"role": "user", "content": MAIN_OBJECTIVE_PREFIX + input_data.prompt}
            )

        # Execute tools based on the selected mode
        if input_data.agent_mode_max_iterations != 0:
            # In agent mode, execute tools directly in a loop until finished
            async for result in self._execute_tools_agent_mode(
                input_data=input_data,
                credentials=credentials,
                tool_functions=tool_functions,
                prompt=prompt,
                graph_exec_id=graph_exec_id,
                node_id=node_id,
                node_exec_id=node_exec_id,
                user_id=user_id,
                graph_id=graph_id,
                graph_version=graph_version,
                execution_context=execution_context,
                execution_processor=execution_processor,
            ):
                yield result
            return

        # One-off mode: single LLM call and yield tool calls for external execution
        current_prompt = list(prompt)
        max_attempts = max(1, int(input_data.retry))
        response = None

        last_error = None
        for _ in range(max_attempts):
            try:
                response = await self._attempt_llm_call_with_validation(
                    credentials, input_data, current_prompt, tool_functions
                )
                break

            except ValueError as e:
                last_error = e
                error_feedback = (
                    "Your tool call had errors. Please fix the following issues and try again:\n"
                    + f"- {str(e)}\n"
                    + "\nPlease make sure to use the exact tool and parameter names as specified in the function schema."
                )
                current_prompt = list(current_prompt) + [
                    {"role": "user", "content": error_feedback}
                ]

        if response is None:
            raise last_error or ValueError(
                "Failed to get valid response after all retry attempts"
            )

        if not response.tool_calls:
            yield "finished", response.response
            return

        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_def = next(
                (
                    tool
                    for tool in tool_functions
                    if tool["function"]["name"] == tool_name
                ),
                None,
            )
            if not tool_def:
                # NOTE: This matches the logic in _attempt_llm_call_with_validation and
                # relies on its validation for the assumption that this is valid to use.
                if len(tool_functions) == 1:
                    tool_def = tool_functions[0]
                else:
                    # This should not happen due to prior validation
                    continue

            if "function" in tool_def and "parameters" in tool_def["function"]:
                expected_args = tool_def["function"]["parameters"].get("properties", {})
            else:
                expected_args = {arg: {} for arg in tool_args.keys()}

            # Get the sink node ID and field mapping from tool definition
            field_mapping = tool_def["function"].get("_field_mapping", {})
            sink_node_id = tool_def["function"]["_sink_node_id"]

            for clean_arg_name in expected_args:
                # arg_name is now always the cleaned field name (for Anthropic API compliance)
                # Get the original field name from field mapping for proper emit key generation
                original_field_name = field_mapping.get(clean_arg_name, clean_arg_name)
                arg_value = tool_args.get(clean_arg_name)

                # Use original_field_name directly (not sanitized) to match link sink_name
                # The field_mapping already translates from LLM's cleaned names to original names
                emit_key = f"tools_^_{sink_node_id}_~_{original_field_name}"

                logger.debug(
                    "[SmartDecisionMakerBlock|geid:%s|neid:%s] emit %s",
                    graph_exec_id,
                    node_exec_id,
                    emit_key,
                )
                yield emit_key, arg_value

        if response.reasoning:
            prompt.append(
                {"role": "assistant", "content": f"[Reasoning]: {response.reasoning}"}
            )

        prompt.append(_convert_raw_response_to_dict(response.raw_response))

        yield "conversations", prompt
