import asyncio
import logging
import re
import shutil
import tempfile
import types
import uuid as uuid_mod
from collections import Counter
from collections.abc import AsyncIterable, Sequence
from concurrent.futures import Future
from enum import Enum
from functools import partial
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
from backend.copilot.sdk.env import config as copilot_config
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
from backend.util.security import SENSITIVE_FIELD_NAMES
from backend.util.tool_call_loop import (
    LLMLoopResponse,
    LLMToolCall,
    ToolCallLoopResult,
    ToolCallResult,
    tool_call_loop,
)

if TYPE_CHECKING:
    from backend.data.graph import Link, Node
    from backend.executor.manager import ExecutionProcessor

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """How the OrchestratorBlock executes tool calls.

    Designed to be provider-agnostic: new SDK integrations (e.g. Codex,
    Copilot) can be added as additional enum members without renaming
    existing ones.
    """

    BUILT_IN = "built_in"
    """Default built-in tool-call loop (supports all LLM providers)."""

    EXTENDED_THINKING = "extended_thinking"
    """Delegate the tool-calling loop to an external Agent SDK for richer
    reasoning (e.g. extended thinking, multi-step planning).
    Currently supports Anthropic-compatible providers (anthropic / open_router)
    via the Claude Agent SDK; additional SDK backends may be added later."""


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
    Supports OpenAI Chat Completions, Responses API, and Anthropic formats.
    """
    tool_call_ids = []

    # OpenAI Responses API: function_call items have type="function_call"
    if entry.get("type") == "function_call":
        if call_id := entry.get("call_id"):
            tool_call_ids.append(call_id)
        return tool_call_ids

    if entry.get("role") != "assistant":
        return tool_call_ids

    # OpenAI Chat Completions: check for tool_calls in the entry.
    calls = entry.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if tool_id := call.get("id"):
                tool_call_ids.append(tool_id)

    # Anthropic: check content items for tool_use type.
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
    Supports OpenAI Chat Completions, Responses API, and Anthropic formats.
    """
    tool_call_ids: list[str] = []

    # OpenAI Responses API: function_call_output items
    if entry.get("type") == "function_call_output":
        if call_id := entry.get("call_id"):
            tool_call_ids.append(str(call_id))
        return tool_call_ids

    # OpenAI Chat Completions: a tool response message with role "tool".
    if entry.get("role") == "tool":
        if tool_call_id := entry.get("tool_call_id"):
            tool_call_ids.append(str(tool_call_id))

    # Anthropic: check content items for tool_result type.
    if entry.get("role") == "user":
        content = entry.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") != "tool_result":
                    continue
                if tool_call_id := item.get("tool_use_id"):
                    tool_call_ids.append(tool_call_id)

    return tool_call_ids


def _create_tool_response(
    call_id: str, output: Any, *, responses_api: bool = False
) -> dict[str, Any]:
    """
    Create a tool response message for OpenAI, Anthropic, or OpenAI Responses API,
    based on the tool_id format and the responses_api flag.
    """
    content = output if isinstance(output, str) else json.dumps(output)

    # Anthropic format: tool IDs typically start with "toolu_"
    if call_id.startswith("toolu_"):
        return {
            "role": "user",
            "type": "message",
            "content": [
                {"tool_use_id": call_id, "type": "tool_result", "content": content}
            ],
        }

    # OpenAI Responses API format
    if responses_api:
        return {"type": "function_call_output", "call_id": call_id, "output": content}

    # OpenAI Chat Completions format (default fallback)
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


def _convert_raw_response_to_dict(
    raw_response: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Safely convert raw_response to dictionary format for conversation history.
    Handles different response types from different LLM providers.

    For the OpenAI Responses API, the raw_response is the entire Response
    object.  Its ``output`` items (messages, function_calls) are extracted
    individually so they can be used as valid input items on the next call.
    Returns a **list** of dicts in that case.

    For Chat Completions / Anthropic / Ollama, returns a single dict.
    """
    if isinstance(raw_response, str):
        # Ollama returns a string, convert to dict format
        return {"role": "assistant", "content": raw_response}
    elif isinstance(raw_response, dict):
        # Already a dict (from tests or some providers)
        return raw_response
    elif _is_responses_api_object(raw_response):
        # OpenAI Responses API: extract individual output items.
        # Strip 'status' — it's a response-only field that OpenAI rejects
        # when the item is sent back as input on the next API call.
        items = [
            {k: v for k, v in json.to_dict(item).items() if k != "status"}
            for item in raw_response.output
        ]
        return items if items else [{"role": "assistant", "content": ""}]
    else:
        # Chat Completions / Anthropic return message objects
        return json.to_dict(raw_response)


def _is_responses_api_object(obj: Any) -> bool:
    """Detect an OpenAI Responses API Response object.

    These have ``object == "response"`` and an ``output`` list, but no
    ``role`` attribute (unlike ChatCompletionMessage).
    """
    return (
        getattr(obj, "object", None) == "response"
        and hasattr(obj, "output")
        and not hasattr(obj, "role")
    )


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


def _disambiguate_tool_names(tools: list[dict[str, Any]]) -> None:
    """Ensure all tool names are unique (Anthropic API requires this).

    When multiple nodes use the same block type, they get the same tool name.
    This appends _1, _2, etc. and enriches descriptions with hardcoded defaults
    so the LLM can distinguish them. Mutates the list in place.

    Malformed tools (missing ``function`` or ``function.name``) are silently
    skipped so the caller never crashes on unexpected input.
    """
    # Collect tools that have the required structure, skipping malformed ones.
    valid_tools: list[dict[str, Any]] = []
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(func, dict) or not isinstance(func.get("name"), str):
            # Strip internal metadata even from malformed entries.
            if isinstance(func, dict):
                func.pop("_hardcoded_defaults", None)
            continue
        valid_tools.append(tool)

    names = [t.get("function", {}).get("name", "") for t in valid_tools]
    name_counts = Counter(names)
    duplicates = {n for n, c in name_counts.items() if c > 1}

    if not duplicates:
        for t in valid_tools:
            t.get("function", {}).pop("_hardcoded_defaults", None)
        return

    taken: set[str] = set(names)
    counters: dict[str, int] = {}

    for tool in valid_tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        defaults = func.pop("_hardcoded_defaults", {})

        if name not in duplicates:
            continue

        counters[name] = counters.get(name, 0) + 1
        # Skip suffixes that collide with existing (e.g. user-named) tools
        while True:
            suffix = f"_{counters[name]}"
            candidate = f"{name[: 64 - len(suffix)]}{suffix}"
            if candidate not in taken:
                break
            counters[name] += 1

        func["name"] = candidate
        taken.add(candidate)

        if defaults and isinstance(defaults, dict):
            parts: list[str] = []
            for k, v in defaults.items():
                rendered = json.dumps(v)
                if len(rendered) > 100:
                    rendered = rendered[:80] + "...<truncated>"
                parts.append(f"{k}={rendered}")
            summary = ", ".join(parts)
            original_desc = func.get("description", "") or ""
            func["description"] = f"{original_desc} [Pre-configured: {summary}]"


class OrchestratorBlock(Block):
    """
    A block that uses a language model to orchestrate tool calls, supporting both
    single-shot and iterative agent mode execution.
    """

    # MCP server name used by the Claude Code SDK execution mode.  Keep in sync
    # with _create_graph_mcp_server and the MCP_PREFIX derivation in _execute_tools_sdk_mode.
    _SDK_MCP_SERVER_NAME = "graph_tools"

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
        execution_mode: ExecutionMode = SchemaField(
            title="Execution Mode",
            default=ExecutionMode.BUILT_IN,
            description="How tool calls are executed. "
            "'built_in' uses the default tool-call loop (all providers). "
            "'extended_thinking' delegates to an external Agent SDK for richer reasoning "
            "(currently Anthropic / OpenRouter only, requires API credentials, "
            "ignores 'Agent Mode Max Iterations').",
            advanced=True,
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
            input_schema=OrchestratorBlock.Input,
            output_schema=OrchestratorBlock.Output,
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
    def _build_tool_info_from_args(
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_def: dict[str, Any],
    ) -> ToolInfo:
        """Build a ToolInfo from parsed tool call arguments and a tool definition.

        Shared between the agent mode tool executor and the SDK MCP handler
        to avoid duplicating the field-mapping + ToolInfo construction logic.
        """
        field_mapping = tool_def["function"].get("_field_mapping", {})
        input_data: dict[str, Any] = {}
        if "function" in tool_def and "parameters" in tool_def["function"]:
            expected_args = tool_def["function"]["parameters"].get("properties", {})
            for clean_name in expected_args:
                original = field_mapping.get(clean_name, clean_name)
                input_data[original] = tool_args.get(clean_name)

        mock_tc = types.SimpleNamespace(
            id=tool_call_id,
            function=types.SimpleNamespace(
                name=tool_name,
                arguments=json.dumps(tool_args),
            ),
        )
        return ToolInfo(
            tool_call=mock_tc,
            tool_name=tool_name,
            tool_def=tool_def,
            input_data=input_data,
            field_mapping=field_mapping,
        )

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
            "name": OrchestratorBlock.cleanup(tool_name),
            "description": block.description,
        }
        sink_block_input_schema = block.input_schema
        properties = {}
        field_mapping = {}  # clean_name -> original_name

        for link in links:
            field_name = link.sink_name
            is_dynamic = is_dynamic_field(field_name)
            # Clean property key to ensure Anthropic API compatibility for ALL fields
            clean_field_name = OrchestratorBlock.cleanup(field_name)
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
            clean_field_name = OrchestratorBlock.cleanup(field_name)

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

        # Store hardcoded defaults (non-linked inputs) for disambiguation.
        # Exclude linked fields, private fields, and credential/auth fields
        # to avoid leaking sensitive data into tool descriptions.
        linked_fields = {link.sink_name for link in links}
        defaults = sink_node.input_default
        tool_function["_hardcoded_defaults"] = (
            {
                k: v
                for k, v in defaults.items()
                if k not in linked_fields
                and not k.startswith("_")
                and k.lower() not in SENSITIVE_FIELD_NAMES
            }
            if isinstance(defaults, dict)
            else {}
        )

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
            "name": OrchestratorBlock.cleanup(tool_name),
            "description": sink_graph_meta.description,
        }

        properties = {}
        field_mapping = {}

        for link in links:
            field_name = link.sink_name

            clean_field_name = OrchestratorBlock.cleanup(field_name)
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

        # Store hardcoded defaults (non-linked inputs) for disambiguation.
        # Exclude linked fields, private fields, agent meta fields, and
        # credential/auth fields to avoid leaking sensitive data.
        linked_fields = {link.sink_name for link in links}
        defaults = sink_node.input_default
        tool_function["_hardcoded_defaults"] = (
            {
                k: v
                for k, v in defaults.items()
                if k not in linked_fields
                and k not in ("graph_id", "graph_version", "input_schema")
                and not k.startswith("_")
                and k.lower() not in SENSITIVE_FIELD_NAMES
            }
            if isinstance(defaults, dict)
            else {}
        )

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
                tool_func = await OrchestratorBlock._create_agent_function_signature(
                    sink_node, links
                )
                return_tool_functions.append(tool_func)
            else:
                tool_func = await OrchestratorBlock._create_block_function_signature(
                    sink_node, links
                )
                return_tool_functions.append(tool_func)

        _disambiguate_tool_names(return_tool_functions)
        return return_tool_functions

    async def _attempt_llm_call_with_validation(
        self,
        credentials: llm.APIKeyCredentials,
        input_data: Input,
        current_prompt: list[dict[str, Any]],
        tool_functions: list[dict[str, Any]],
    ) -> Any:
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
        self, response: Any, tool_functions: list[dict[str, Any]]
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
        self,
        prompt: list[dict[str, Any]],
        response: Any,
        tool_outputs: list[dict[str, Any]] | None = None,
    ):
        """Update conversation history with response and tool outputs.

        ``response`` must be an ``LLMResponse`` (from ``backend.blocks.llm``),
        **not** an ``LLMLoopResponse``. The method accesses ``.raw_response``
        and ``.reasoning`` attributes on the passed object.
        """
        converted = _convert_raw_response_to_dict(response.raw_response)

        if isinstance(converted, list):
            # Responses API: output items are already individual dicts
            has_tool_calls = any(
                item.get("type") == "function_call" for item in converted
            )
            if response.reasoning and not has_tool_calls:
                prompt.append(
                    {
                        "role": "assistant",
                        "content": f"[Reasoning]: {response.reasoning}",
                    }
                )
            prompt.extend(converted)
        else:
            # Chat Completions / Anthropic: single assistant message dict
            has_tool_calls = isinstance(converted.get("content"), list) and any(
                item.get("type") == "tool_use" for item in converted.get("content", [])
            )
            if response.reasoning and not has_tool_calls:
                prompt.append(
                    {
                        "role": "assistant",
                        "content": f"[Reasoning]: {response.reasoning}",
                    }
                )
            prompt.append(converted)

        if tool_outputs:
            prompt.extend(tool_outputs)

    async def _execute_single_tool_with_manager(
        self,
        tool_info: ToolInfo,
        execution_params: ExecutionParams,
        execution_processor: "ExecutionProcessor",
        *,
        responses_api: bool = False,
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

        # Merge static defaults from the target node with LLM-provided inputs.
        # The LLM only passes values it decides to fill (e.g., "value"), but
        # static defaults like "name" on Agent Output Blocks must be included
        # so the execution record is complete for from_db() reconstruction.
        merged_input_data = {**target_node.input_default, **raw_input_data}

        # Add all inputs to the execution
        if not merged_input_data:
            raise ValueError(f"Tool call has no input data: {tool_call}")

        for input_name, input_value in merged_input_data.items():
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

            # Execute the node directly since we're in the Orchestrator context
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
            return _create_tool_response(
                tool_call.id, tool_response_content, responses_api=responses_api
            )

        except Exception as e:
            logger.warning("Tool execution with manager failed: %s", e)
            # Return error response
            return _create_tool_response(
                tool_call.id,
                f"Tool execution failed: {e}",
                responses_api=responses_api,
            )

    async def _agent_mode_llm_caller(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Any],
        *,
        credentials: llm.APIKeyCredentials,
        input_data: "OrchestratorBlock.Input",
    ) -> LLMLoopResponse:
        """LLM caller callback for agent mode: wraps _attempt_llm_call_with_validation."""
        resp = await self._attempt_llm_call_with_validation(
            credentials, input_data, messages, list(tools)
        )
        tool_calls = [
            LLMToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in (resp.tool_calls or [])
        ]
        return LLMLoopResponse(
            response_text=resp.response,
            tool_calls=tool_calls,
            raw_response=resp,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            reasoning=resp.reasoning,
        )

    async def _agent_mode_tool_executor(
        self,
        tool_call: LLMToolCall,
        tools: Sequence[Any],
        *,
        execution_params: ExecutionParams,
        execution_processor: "ExecutionProcessor",
        use_responses_api: bool,
    ) -> ToolCallResult:
        """Tool executor callback for agent mode: wraps _execute_single_tool_with_manager."""
        # Find tool definition
        tool_def = next(
            (t for t in tools if t["function"]["name"] == tool_call.name),
            None,
        )
        if not tool_def and len(tools) == 1:
            tool_def = tools[0]
        if not tool_def:
            return ToolCallResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=f"Unknown tool: {tool_call.name}",
                is_error=True,
            )

        try:
            tool_args = json.loads(tool_call.arguments)
        except (ValueError, TypeError) as e:
            return ToolCallResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=f"Invalid JSON arguments: {e}",
                is_error=True,
            )
        tool_info = OrchestratorBlock._build_tool_info_from_args(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            tool_args=tool_args,
            tool_def=tool_def,
        )

        try:
            result = await self._execute_single_tool_with_manager(
                tool_info,
                execution_params,
                execution_processor,
                responses_api=use_responses_api,
            )
            # Unwrap the tool content from the provider-specific envelope.
            # _execute_single_tool_with_manager returns a full message dict
            # (e.g. {"role":"tool","content":"..."} for Chat API,
            #  or {"type":"function_call_output","output":"..."} for Responses API).
            raw_content = result.get("content") or result.get("output")
            if isinstance(raw_content, list):
                # Anthropic format: [{"type":"tool_result","content":"..."}]
                parts = [
                    item.get("content", "")
                    for item in raw_content
                    if isinstance(item, dict)
                ]
                content = (
                    "\n".join(str(p) for p in parts)
                    if parts
                    else "Tool executed successfully"
                )
            elif raw_content is not None:
                content = str(raw_content)
            else:
                content = "Tool executed successfully"
            tool_failed = content.startswith("Tool execution failed:")
            return ToolCallResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=content,
                is_error=tool_failed,
            )
        except Exception as e:
            logger.error("Tool execution failed: %s", e)
            return ToolCallResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=f"Error: {e}",
                is_error=True,
            )

    def _agent_mode_conversation_updater(
        self,
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
        *,
        use_responses_api: bool = False,
    ) -> None:
        """Conversation updater callback for agent mode."""
        tool_outputs = None
        if tool_results:
            tool_outputs = [
                _create_tool_response(
                    tr.tool_call_id,
                    tr.content,
                    responses_api=use_responses_api,
                )
                for tr in tool_results
            ]
            tool_outputs = _combine_tool_responses(tool_outputs)
        # Pass the raw LLM response (not the LLMLoopResponse wrapper) —
        # _update_conversation expects the provider response object that
        # has .raw_response and .reasoning attributes.
        self._update_conversation(messages, response.raw_response, tool_outputs)

    async def _execute_tools_agent_mode(
        self,
        input_data: "OrchestratorBlock.Input",
        credentials: llm.APIKeyCredentials,
        tool_functions: list[dict[str, Any]],
        prompt: list[dict[str, Any]],
        graph_exec_id: str,
        node_id: str,
        node_exec_id: str,
        user_id: str,
        graph_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        execution_processor: "ExecutionProcessor",
    ):
        """Execute tools in agent mode using the shared tool-calling loop."""
        max_iterations = input_data.agent_mode_max_iterations
        use_responses_api = input_data.model.metadata.provider == "openai"

        execution_params = ExecutionParams(
            user_id=user_id,
            graph_id=graph_id,
            node_id=node_id,
            graph_version=graph_version,
            graph_exec_id=graph_exec_id,
            node_exec_id=node_exec_id,
            execution_context=execution_context,
        )

        # Bind callbacks using functools.partial
        bound_llm_caller = partial(
            self._agent_mode_llm_caller,
            credentials=credentials,
            input_data=input_data,
        )
        bound_tool_executor = partial(
            self._agent_mode_tool_executor,
            execution_params=execution_params,
            execution_processor=execution_processor,
            use_responses_api=use_responses_api,
        )
        bound_conversation_updater = partial(
            self._agent_mode_conversation_updater,
            use_responses_api=use_responses_api,
        )

        current_prompt = list(prompt)

        last_iter_msg = None
        if max_iterations > 0:
            last_iter_msg = (
                f"{MAIN_OBJECTIVE_PREFIX}This is your last iteration. "
                "Try to complete the task with the information you have. "
                "If you cannot fully complete it, provide a summary of what "
                "you've accomplished and what remains to be done. "
                "Prefer finishing with a clear response rather than making "
                "additional tool calls."
            )

        try:
            loop_result = ToolCallLoopResult(response_text="", messages=current_prompt)
            async for loop_result in tool_call_loop(
                messages=current_prompt,
                tools=tool_functions,
                llm_call=bound_llm_caller,
                execute_tool=bound_tool_executor,
                update_conversation=bound_conversation_updater,
                max_iterations=max_iterations,
                last_iteration_message=last_iter_msg,
            ):
                # Yield intermediate tool calls so the UI can show progress.
                # Only yield conversations when there are tool calls to report;
                # the final conversation state is always emitted once after the
                # loop (line below) to avoid duplicate yields when max_iterations
                # is reached.
                if loop_result.last_tool_calls:
                    yield "conversations", loop_result.messages
                for tc in loop_result.last_tool_calls:
                    yield (
                        "tool_calls",
                        {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    )
        except Exception as e:
            # Catch all errors (validation, network, API) so that the block
            # surfaces them as user-visible output instead of crashing.
            yield "error", str(e)
            return

        if loop_result.finished_naturally:
            yield "finished", loop_result.response_text
        else:
            yield (
                "finished",
                (
                    f"Agent mode completed after {loop_result.iterations} "
                    "iterations (limit reached)"
                ),
            )
        yield "conversations", loop_result.messages

    def _create_graph_mcp_server(
        self,
        tool_functions: list[dict[str, Any]],
        execution_params: ExecutionParams,
        execution_processor: "ExecutionProcessor",
    ):
        """Create an MCP server from graph-connected tool functions.

        Converts the OpenAI-format tool signatures (from _create_tool_node_signatures)
        into MCP tools that execute downstream blocks via _execute_single_tool_with_manager.
        """
        from claude_agent_sdk import create_sdk_mcp_server
        from claude_agent_sdk import tool as sdk_tool

        sdk_tools = []
        for tf in tool_functions:
            func_def = tf["function"]
            tool_name = func_def["name"]
            tool_desc = func_def.get("description", "")
            tool_params = func_def.get(
                "parameters", {"type": "object", "properties": {}}
            )

            # Build input schema for MCP (same as tool_adapter.py pattern).
            # Preserve additionalProperties to prevent hallucinated arguments.
            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": dict(tool_params.get("properties", {})),
                "required": list(tool_params.get("required", [])),
            }
            if "additionalProperties" in tool_params:
                input_schema["additionalProperties"] = tool_params[
                    "additionalProperties"
                ]

            def _make_handler(
                _tool_func=tf,
                _self=self,
                _exec_params=execution_params,
                _exec_processor=execution_processor,
            ):
                async def handler(args: dict[str, Any]) -> dict[str, Any]:
                    func = _tool_func["function"]

                    # Build ToolInfo using shared helper
                    tool_info = OrchestratorBlock._build_tool_info_from_args(
                        tool_call_id=f"sdk-{uuid_mod.uuid4().hex[:12]}",
                        tool_name=func["name"],
                        tool_args=args,
                        tool_def=_tool_func,
                    )

                    try:
                        result = await _self._execute_single_tool_with_manager(
                            tool_info, _exec_params, _exec_processor
                        )
                        # result is a tool response dict with "content" key
                        content = result.get("content", "Tool executed successfully")
                        if isinstance(content, str):
                            text = content
                        else:
                            text = json.dumps(content)
                        tool_failed = text.startswith("Tool execution failed:")
                        return {
                            "content": [{"type": "text", "text": text}],
                            "isError": tool_failed,
                        }
                    except Exception as e:
                        logger.error("SDK tool execution failed: %s", e)
                        return {
                            "content": [{"type": "text", "text": f"Error: {e}"}],
                            "isError": True,
                        }

                return handler

            decorated = sdk_tool(tool_name, tool_desc, input_schema)(_make_handler())
            sdk_tools.append(decorated)

        return create_sdk_mcp_server(
            name=OrchestratorBlock._SDK_MCP_SERVER_NAME,
            version="1.0.0",
            tools=sdk_tools,
        )

    async def _execute_tools_sdk_mode(
        self,
        input_data: "OrchestratorBlock.Input",
        credentials: llm.APIKeyCredentials,
        tool_functions: list[dict[str, Any]],
        prompt: list[dict[str, Any]],
        execution_params: ExecutionParams,
        execution_processor: "ExecutionProcessor",
    ):
        """Execute tools using the Claude Agent SDK.

        The SDK manages the conversation loop and tool calling natively.
        Graph-connected blocks are exposed as MCP tools.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        # Build MCP server from graph-connected tools
        mcp_server = self._create_graph_mcp_server(
            tool_functions, execution_params, execution_processor
        )

        # Build allowed tools list (MCP-prefixed names).
        # Derive the prefix from the class-level server name constant.
        MCP_PREFIX = f"mcp__{self._SDK_MCP_SERVER_NAME}__"
        allowed_tools = [
            f"{MCP_PREFIX}{tf['function']['name']}" for tf in tool_functions
        ]

        # Disable ALL known SDK built-in tools — only graph MCP tools available.
        # `allowed_tools` (above) is the primary restriction: the SDK only
        # enables tools explicitly listed there.  This blocklist is a
        # defense-in-depth measure in case the SDK's allowlist logic changes.
        # IMPORTANT: Keep this list in sync with the Claude Agent SDK.
        # If a new built-in tool is added in a future SDK version, it will
        # still be blocked by `allowed_tools` (only MCP-prefixed names are
        # allowed), but adding it here provides an extra safety layer.
        disallowed_tools = [
            "Bash",
            "WebFetch",
            "AskUserQuestion",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "Task",
            "WebSearch",
            "TodoWrite",
            "NotebookEdit",
        ]

        # Build SDK env — provider-aware credential routing.
        # Extended thinking does not support subscription-mode (platform-managed credits).
        # Use *credential* provider for routing (not model metadata provider),
        # because a user may select an Anthropic model but route through OpenRouter.
        provider = credentials.provider
        if not credentials.api_key:
            yield (
                "error",
                (
                    "Extended thinking requires direct API credentials and does not "
                    "support subscription mode. Please provide an Anthropic or OpenRouter API key."
                ),
            )
            return
        api_key = credentials.api_key.get_secret_value()
        if provider == "open_router":
            # Route through OpenRouter proxy: set base URL + auth token,
            # clear API key so the SDK uses AUTH_TOKEN instead.
            # NOTE: We use the platform's global OpenRouter base URL from
            # ChatConfig.  Per-credential base URLs are not yet supported;
            # if the user's credential targets a custom proxy, the SDK will
            # still route through the platform's configured endpoint.
            or_base = (copilot_config.base_url or "https://openrouter.ai/api").rstrip(
                "/"
            )
            if or_base.endswith("/v1"):
                or_base = or_base[:-3]
            sdk_env = {
                "ANTHROPIC_BASE_URL": or_base,
                "ANTHROPIC_AUTH_TOKEN": api_key,
                "ANTHROPIC_API_KEY": "",  # force CLI to use AUTH_TOKEN
            }
        else:
            # Direct Anthropic key
            sdk_env = {"ANTHROPIC_API_KEY": api_key}

        # Use an execution-specific working directory to prevent concurrent
        # SDK executions from colliding.  tempfile.mkdtemp() respects TMPDIR
        # and works in containerised environments with read-only root filesystems.
        sdk_cwd = tempfile.mkdtemp(
            prefix=f"orchestrator-sdk-{execution_params.graph_exec_id}-"
        )

        response_parts: list[str] = []
        conversation: list[dict[str, Any]] = list(prompt)  # Start with input prompt
        total_prompt_tokens = 0
        total_completion_tokens = 0

        sdk_error: Exception | None = None
        try:
            # Build SDK options
            options = ClaudeAgentOptions(
                system_prompt=input_data.sys_prompt or "",
                mcp_servers={self._SDK_MCP_SERVER_NAME: mcp_server},
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                cwd=sdk_cwd,
                env=sdk_env,
                model=input_data.model.value or None,
            )

            # Strip system messages from prompt — they're already passed via
            # ClaudeAgentOptions.system_prompt to avoid sending them twice.
            sdk_prompt = [p for p in prompt if p.get("role") != "system"]

            # Build user message from prompt.
            # The SDK's query() accepts a string or an async iterable of message dicts.
            # For multi-turn conversations, pass the full history as an async iterable
            # to preserve assistant replies, tool calls/results, and system messages.
            has_multi_turn = any(
                p.get("role") in ("assistant", "tool") for p in sdk_prompt
            )
            if has_multi_turn:

                async def _prompt_iter():
                    for p in sdk_prompt:
                        yield p

                user_message: str | AsyncIterable[dict[str, Any]] = _prompt_iter()
            else:
                # Single-turn: collapse user content into one string
                user_parts = []
                for p in sdk_prompt:
                    if p.get("role") == "user" and p.get("content"):
                        user_parts.append(str(p["content"]))
                user_message = (
                    "\n\n".join(user_parts) if user_parts else input_data.prompt
                )

            # Run SDK client with heartbeat-safe message iteration.
            # We must NOT cancel __anext__() mid-flight — doing so corrupts
            # the SDK's internal anyio memory stream (same pattern as
            # copilot/sdk/service.py:_iter_sdk_messages).

            _HEARTBEAT_INTERVAL = 10.0  # seconds
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_message)

                msg_iter = client.receive_response().__aiter__()
                pending_task: asyncio.Task[Any] | None = None

                async def _next_msg() -> Any:
                    return await msg_iter.__anext__()

                try:
                    while True:
                        if pending_task is None:
                            pending_task = asyncio.create_task(_next_msg())

                        done, _ = await asyncio.wait(
                            {pending_task}, timeout=_HEARTBEAT_INTERVAL
                        )

                        if not done:
                            # Heartbeat — SDK is still processing, keep waiting
                            continue

                        pending_task = None
                        try:
                            sdk_msg = done.pop().result()
                        except StopAsyncIteration:
                            break

                        if isinstance(sdk_msg, AssistantMessage):
                            text_parts = []
                            tool_use_parts = []
                            for content_block in sdk_msg.content:
                                if isinstance(content_block, TextBlock):
                                    text_parts.append(content_block.text)
                                    response_parts.append(content_block.text)
                                elif isinstance(content_block, ToolUseBlock):
                                    raw_name = getattr(content_block, "name", "unknown")
                                    # Strip MCP prefix for readability in
                                    # conversation history.
                                    clean_name = raw_name.removeprefix(MCP_PREFIX)
                                    tool_use_parts.append(
                                        {
                                            "tool": clean_name,
                                            "id": getattr(
                                                content_block, "id", "unknown"
                                            ),
                                        }
                                    )
                            if text_parts or tool_use_parts:
                                msg_content = "".join(text_parts)
                                if tool_use_parts:
                                    tool_summary = ", ".join(
                                        t["tool"] for t in tool_use_parts
                                    )
                                    if msg_content:
                                        msg_content += f"\n[Tool calls: {tool_summary}]"
                                    else:
                                        msg_content = f"[Tool calls: {tool_summary}]"
                                conversation.append(
                                    {
                                        "role": "assistant",
                                        "content": msg_content,
                                    }
                                )
                        elif isinstance(sdk_msg, UserMessage):
                            # Capture tool results so the conversation
                            # history records what each tool returned.
                            result_parts: list[str] = []
                            for block in getattr(sdk_msg, "content", []):
                                if isinstance(block, ToolResultBlock):
                                    content_val = getattr(block, "content", "")
                                    if isinstance(content_val, list):
                                        # list of text blocks
                                        for item in content_val:
                                            if isinstance(item, dict):
                                                result_parts.append(
                                                    item.get("text", "")
                                                )
                                    elif content_val:
                                        result_parts.append(str(content_val))
                            if result_parts:
                                conversation.append(
                                    {
                                        "role": "tool",
                                        "content": "\n".join(result_parts),
                                    }
                                )
                        elif isinstance(sdk_msg, ResultMessage):
                            if sdk_msg.usage:
                                total_prompt_tokens += getattr(
                                    sdk_msg.usage, "input_tokens", 0
                                )
                                total_completion_tokens += getattr(
                                    sdk_msg.usage, "output_tokens", 0
                                )
                finally:
                    if pending_task is not None and not pending_task.done():
                        pending_task.cancel()
                        try:
                            await pending_task
                        except (asyncio.CancelledError, StopAsyncIteration):
                            pass
        except Exception as e:
            # Surface SDK errors as user-visible output instead of crashing,
            # consistent with _execute_tools_agent_mode error handling.
            # Don't return yet — fall through to merge_stats below so
            # partial token usage is always recorded.
            sdk_error = e
        finally:
            # Always record usage stats, even on error.  The SDK may have
            # made LLM calls (consuming tokens) before the failure; dropping
            # those stats would under-count resource usage.
            # llm_call_count=1 is approximate; the SDK manages its own
            # multi-turn loop and only exposes aggregate usage.
            if total_prompt_tokens > 0 or total_completion_tokens > 0:
                self.merge_stats(
                    NodeExecutionStats(
                        input_token_count=total_prompt_tokens,
                        output_token_count=total_completion_tokens,
                        llm_call_count=1,
                    )
                )
            # Clean up execution-specific working directory.
            shutil.rmtree(sdk_cwd, ignore_errors=True)

        if sdk_error is not None:
            yield "error", str(sdk_error)
            return

        response_text = "".join(response_parts)

        yield "finished", response_text
        yield "conversations", conversation

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

        use_responses_api = input_data.model.metadata.provider == "openai"

        tool_output = []
        if pending_tool_calls and input_data.last_tool_output is not None:
            first_call_id = next(iter(pending_tool_calls.keys()))
            tool_output.append(
                _create_tool_response(
                    first_call_id,
                    input_data.last_tool_output,
                    responses_api=use_responses_api,
                )
            )

            prompt.extend(tool_output)
            remaining_pending_calls = get_pending_tool_calls(prompt)

            if remaining_pending_calls:
                yield "conversations", prompt
                return
        elif input_data.last_tool_output:
            logger.error(
                f"[OrchestratorBlock-node_exec_id={node_exec_id}] "
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
            input_data.prompt = await llm.fmt.format_string(input_data.prompt, values)
            input_data.sys_prompt = await llm.fmt.format_string(
                input_data.sys_prompt, values
            )

        if input_data.sys_prompt and not any(
            p.get("role") == "system"
            and isinstance(p.get("content"), str)
            and p["content"].startswith(MAIN_OBJECTIVE_PREFIX)
            for p in prompt
        ):
            prompt.append(
                {
                    "role": "system",
                    "content": MAIN_OBJECTIVE_PREFIX + input_data.sys_prompt,
                }
            )

        if input_data.prompt and not any(
            p.get("role") == "user"
            and isinstance(p.get("content"), str)
            and p["content"].startswith(MAIN_OBJECTIVE_PREFIX)
            for p in prompt
        ):
            prompt.append(
                {"role": "user", "content": MAIN_OBJECTIVE_PREFIX + input_data.prompt}
            )

        # Execute tools based on the selected mode
        if input_data.execution_mode == ExecutionMode.EXTENDED_THINKING:
            # Validate — Claude Code SDK only works with Claude models
            provider = input_data.model.metadata.provider
            model_name = input_data.model.value
            # All Claude models have metadata.provider == "anthropic", but
            # "open_router" is included defensively in case future models
            # use a different metadata provider for the same Anthropic API.
            if provider not in ("anthropic", "open_router"):
                raise ValueError(
                    f"Claude Code SDK mode requires an Anthropic-compatible "
                    f"provider (got provider={provider}). "
                    "Please select an Anthropic or OpenRouter provider, "
                    "or switch execution mode to 'built_in'."
                )
            # Safety-net: all Claude models have .value starting with "claude-".
            # This guards against non-Claude models that happen to use the
            # "anthropic" metadata provider (if any are added in the future).
            if not model_name.startswith("claude"):
                raise ValueError(
                    f"Claude Code SDK mode only supports Claude models "
                    f"(got model={model_name}). "
                    "Please select a Claude model, "
                    "or switch execution mode to 'built_in'."
                )
            # Claude Code SDK: SDK manages conversation + tool calling
            execution_params = ExecutionParams(
                user_id=user_id,
                graph_id=graph_id,
                node_id=node_id,
                graph_version=graph_version,
                graph_exec_id=graph_exec_id,
                node_exec_id=node_exec_id,
                execution_context=execution_context,
            )
            async for result in self._execute_tools_sdk_mode(
                input_data=input_data,
                credentials=credentials,
                tool_functions=tool_functions,
                prompt=prompt,
                execution_params=execution_params,
                execution_processor=execution_processor,
            ):
                yield result
            return

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
                    "[OrchestratorBlock|geid:%s|neid:%s] emit %s",
                    graph_exec_id,
                    node_exec_id,
                    emit_key,
                )
                yield emit_key, arg_value

        converted = _convert_raw_response_to_dict(response.raw_response)

        # Check for tool calls to avoid inserting reasoning between tool pairs
        if isinstance(converted, list):
            has_tool_calls = any(
                item.get("type") == "function_call" for item in converted
            )
        else:
            has_tool_calls = isinstance(converted.get("content"), list) and any(
                item.get("type") == "tool_use" for item in converted.get("content", [])
            )

        if response.reasoning and not has_tool_calls:
            prompt.append(
                {"role": "assistant", "content": f"[Reasoning]: {response.reasoning}"}
            )

        if isinstance(converted, list):
            prompt.extend(converted)
        else:
            prompt.append(converted)

        yield "conversations", prompt
