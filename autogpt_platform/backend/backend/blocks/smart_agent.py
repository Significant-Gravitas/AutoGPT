import logging
import os
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from claude_agent_sdk import query
from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    ToolUseBlock,
)
from pydantic import BaseModel, SecretStr

# Avoid circular imports by importing only essential types
# ExecutionParams, ToolInfo, and other classes will be imported dynamically when needed
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
)
from backend.data.dynamic_fields import is_tool_pin
from backend.data.execution import ExecutionContext
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util import json
from backend.util.clients import get_database_manager_async_client

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


def _create_tool_response(call_id: str, content: str) -> dict[str, Any]:
    """Create a tool response in the correct format."""
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


class ModelMetadata(NamedTuple):
    provider: str
    context_window: int
    max_output_tokens: int | None


class AgentModel(str, Enum):
    """Available models for the Smart Agent."""

    # Claude 4.x models (latest)
    CLAUDE_4_1_OPUS = "claude-opus-4-1-20250805"
    CLAUDE_4_OPUS = "claude-opus-4-20250514"
    CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
    CLAUDE_4_5_OPUS = "claude-opus-4-5-20251101"
    CLAUDE_4_5_SONNET = "claude-sonnet-4-5-20250929"
    CLAUDE_4_5_HAIKU = "claude-haiku-4-5-20251001"

    # Claude 3.x models (stable)
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    @property
    def metadata(self) -> ModelMetadata:
        return AGENT_MODEL_METADATA[self]

    @property
    def provider(self) -> str:
        return self.metadata.provider


# Agent model metadata mapping
AGENT_MODEL_METADATA = {
    # Claude 4.x models
    AgentModel.CLAUDE_4_1_OPUS: ModelMetadata(ProviderName.ANTHROPIC, 500000, 4096),
    AgentModel.CLAUDE_4_OPUS: ModelMetadata(ProviderName.ANTHROPIC, 500000, 4096),
    AgentModel.CLAUDE_4_SONNET: ModelMetadata(ProviderName.ANTHROPIC, 500000, 4096),
    AgentModel.CLAUDE_4_5_OPUS: ModelMetadata(ProviderName.ANTHROPIC, 500000, 8192),
    AgentModel.CLAUDE_4_5_SONNET: ModelMetadata(ProviderName.ANTHROPIC, 500000, 8192),
    AgentModel.CLAUDE_4_5_HAIKU: ModelMetadata(ProviderName.ANTHROPIC, 200000, 4096),
    # Claude 3.x models
    AgentModel.CLAUDE_3_7_SONNET: ModelMetadata(ProviderName.ANTHROPIC, 200000, 4096),
    AgentModel.CLAUDE_3_HAIKU: ModelMetadata(ProviderName.ANTHROPIC, 200000, 4096),
}

# Anthropic-only credentials for Claude models
ClaudeCredentials = CredentialsMetaInput[
    Literal[ProviderName.ANTHROPIC], Literal["api_key"]
]


def ClaudeCredentialsField() -> ClaudeCredentials:
    return CredentialsField(
        description="Anthropic API key for Claude Agent SDK access.",
        discriminator="model",
        discriminator_mapping={
            model.value: model.metadata.provider for model in AgentModel
        },
    )


# Test credentials for Claude models
TEST_CLAUDE_CREDENTIALS = APIKeyCredentials(
    id="test-claude-creds",
    provider=ProviderName.ANTHROPIC,
    api_key=SecretStr("mock-anthropic-api-key"),
    title="Mock Anthropic API key",
    expires_at=None,
)
TEST_CLAUDE_CREDENTIALS_INPUT = {
    "provider": TEST_CLAUDE_CREDENTIALS.provider,
    "id": TEST_CLAUDE_CREDENTIALS.id,
    "type": TEST_CLAUDE_CREDENTIALS.type,
    "title": TEST_CLAUDE_CREDENTIALS.title,
}


class SmartAgentBlock(Block):
    """
    A smart agent block that uses Claude Agent SDK for native agent capabilities
    while executing AutoGPT tool nodes.

    This block combines Claude's native agent functionality with AutoGPT's tool ecosystem:
    - Uses Claude Agent SDK for core agent intelligence
    - Discovers connected AutoGPT tool nodes like SmartDecisionMaker
    - When Claude calls tools, executes the actual AutoGPT tool nodes
    - Provides Claude with the tool execution results
    """

    class Input(BlockSchemaInput):
        task: str = SchemaField(
            description="The task for the agent to complete. Be specific about your requirements.",
            placeholder="Analyze the data file and create a summary report with key insights...",
        )
        model: AgentModel = SchemaField(
            title="Model",
            default=AgentModel.CLAUDE_4_5_SONNET,
            description="The model to use for the agent.",
            advanced=False,
        )
        credentials: ClaudeCredentials = ClaudeCredentialsField()
        max_iterations: int = SchemaField(
            default=15,
            description="Maximum number of agent iterations. Use -1 for unlimited (use carefully!).",
            advanced=False,
        )
        system_prompt: str = SchemaField(
            title="System Prompt",
            default="You are a helpful AI assistant with access to tools. Think step by step about which tools to use to complete the task efficiently. When you have completed the objective, provide a clear summary of the results.",
            description="System prompt to guide the agent's behavior.",
            advanced=True,
        )
        working_directory: str = SchemaField(
            default="/tmp/smart_agent",
            description="Working directory for the agent.",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: str = SchemaField(
            description="The final result or answer from the agent."
        )
        iterations_used: int = SchemaField(
            description="Number of iterations used to complete the task."
        )
        tools_used: list[str] = SchemaField(
            description="List of AutoGPT tools used during execution.",
            default_factory=list,
        )
        success: bool = SchemaField(
            description="Whether the task was completed successfully."
        )
        error: str = SchemaField(
            default="", description="Error message if the task failed."
        )
        # Tool output pins for connecting to other blocks (like SmartDecisionMakerBlock)
        tools: Any = SchemaField(
            description="Tool calls output for connecting to other AutoGPT blocks."
        )
        conversations: list[Any] = SchemaField(
            description="Conversation history with Claude Agent SDK.",
            default_factory=list,
        )

    def __init__(self):
        super().__init__(
            id="c1a2u3d4-e5a6-g7e8-n9t0-b1l2o3c4k5d6",
            description=(
                "An AI agent powered by Claude Agent SDK that executes connected AutoGPT tool nodes. "
                "Combines Claude's native agent capabilities with AutoGPT's tool ecosystem."
            ),
            categories={BlockCategory.AI},
            block_type=BlockType.AI,
            input_schema=SmartAgentBlock.Input,
            output_schema=SmartAgentBlock.Output,
            test_input={
                "task": "What tools are available?",
                "credentials": TEST_CLAUDE_CREDENTIALS_INPUT,
                "model": AgentModel.CLAUDE_4_5_SONNET,
            },
            test_output=[],
            test_credentials=TEST_CLAUDE_CREDENTIALS,
        )

    @staticmethod
    def cleanup(s: str):
        """Clean up block names for use as tool function names."""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", s).lower()

    async def _create_tool_node_signatures(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """
        Creates function signatures for connected tools.
        Args:
            node_id: The node_id for which to create function signatures.
        Returns:
            List of function signatures for tools
        """
        from backend.blocks.agent import AgentExecutorBlock

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
                # Dynamic import to avoid circular dependency
                from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock

                tool_func = (
                    await SmartDecisionMakerBlock._create_agent_function_signature(
                        sink_node, links
                    )
                )
                return_tool_functions.append(tool_func)
            else:
                # Dynamic import to avoid circular dependency
                from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock

                tool_func = (
                    await SmartDecisionMakerBlock._create_block_function_signature(
                        sink_node, links
                    )
                )
                return_tool_functions.append(tool_func)
        return return_tool_functions

    async def _execute_single_tool_with_manager(
        self,
        tool_info: ToolInfo,
        execution_params: ExecutionParams,
        execution_processor: "ExecutionProcessor",
    ) -> dict:
        """Execute a single tool using the execution manager for proper integration."""
        # Lazy imports to avoid circular dependencies
        from concurrent.futures import Future

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

            # Execute the node directly since we're in the SmartAgent context
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

    def _setup_environment(
        self, credentials: APIKeyCredentials, working_dir: str
    ) -> dict[str, str]:
        """Setup environment for Claude Agent SDK."""
        os.makedirs(working_dir, exist_ok=True)
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = credentials.api_key.get_secret_value()
        return env

    def _build_tool_descriptions_for_claude(
        self, tool_functions: list[dict[str, Any]]
    ) -> str:
        """Build description of available AutoGPT tools for Claude."""
        if not tool_functions:
            return "No tools are currently connected to this agent."

        tool_descriptions = ["Available AutoGPT tools:"]
        for tool_def in tool_functions:
            func_def = tool_def.get("function", {})
            name = func_def.get("name", "unknown")
            description = func_def.get("description", "No description")
            tool_descriptions.append(f"- {name}: {description}")

        tool_descriptions.append(
            "\nWhen you need to use a tool, call it with function calling syntax."
        )
        return "\n".join(tool_descriptions)

    def _extract_tool_calls_from_claude_message(
        self, message: AssistantMessage
    ) -> list[dict[str, Any]]:
        """Extract tool calls from Claude Agent SDK message."""
        tool_calls = []
        for content_block in message.content:
            if isinstance(content_block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": content_block.id,
                        "function": {
                            "name": content_block.name,
                            "arguments": content_block.input,
                        },
                    }
                )
        return tool_calls

    def _extract_text_content_from_claude_message(
        self, message: AssistantMessage
    ) -> str:
        """Extract text content from Claude Agent SDK message."""
        text_parts = []
        for content_block in message.content:
            if isinstance(content_block, TextBlock):
                text_parts.append(content_block.text)
        return "".join(text_parts)

    def _format_conversation_for_claude(self, conversation: list[dict]) -> str:
        """Format conversation history for Claude Agent SDK."""
        formatted = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                # Format tool response
                tool_id = msg.get("tool_call_id", "unknown")
                formatted.append(f"Tool result ({tool_id}): {content}")
            else:
                # Simple format for user/assistant messages
                formatted.append(f"{role.title()}: {content}")

        return "\n\n".join(formatted)

    def _normalize_tool_args(self, tool_args: Any) -> dict:
        """Normalize tool arguments to dict format."""
        if isinstance(tool_args, str):
            return json.loads(tool_args)
        elif isinstance(tool_args, dict):
            return tool_args
        else:
            return dict(tool_args) if tool_args else {}

    def _create_tool_info_from_claude_call(
        self, tool_call: dict[str, Any], tool_functions: list[dict[str, Any]]
    ) -> ToolInfo:
        """Convert Claude tool call to AutoGPT ToolInfo format."""
        tool_name = tool_call["function"]["name"]
        tool_args = self._normalize_tool_args(tool_call["function"]["arguments"])
        tool_id = tool_call["id"]

        # Find the AutoGPT tool definition
        tool_def = next(
            (
                tf
                for tf in tool_functions
                if tf.get("function", {}).get("name") == tool_name
            ),
            None,
        )

        if not tool_def:
            raise ValueError(f"AutoGPT tool '{tool_name}' not found")

        # Create mock tool call object for AutoGPT compatibility
        class MockToolCall:
            def __init__(self, tool_id: str, name: str, args: dict):
                self.id = tool_id
                self.function = type(
                    "Function", (), {"name": name, "arguments": json.dumps(args)}
                )()

        # Build input data from arguments
        field_mapping = tool_def["function"].get("_field_mapping", {})
        expected_args = tool_def["function"]["parameters"].get("properties", {})

        input_data = {
            field_mapping.get(clean_arg_name, clean_arg_name): tool_args.get(
                clean_arg_name
            )
            for clean_arg_name in expected_args
        }

        return ToolInfo(
            tool_call=MockToolCall(tool_id, tool_name, tool_args),
            tool_name=tool_name,
            tool_def=tool_def,
            input_data=input_data,
            field_mapping=field_mapping,
        )

    async def _attempt_claude_call_with_validation(
        self,
        prompt: str,
        options: ClaudeAgentOptions,
    ) -> AssistantMessage:
        """Claude SDK call - let generator cleanup happen naturally to avoid cancel scope issues."""
        try:
            # Simple approach: don't try to manually manage the generator lifecycle
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    return message

            raise ValueError("No AssistantMessage received from Claude SDK")

        except Exception as e:
            logger.error(f"Claude SDK call failed: {e}")
            raise
        # Note: No finally block - let the generator be cleaned up naturally by garbage collection

    async def _execute_tools_agent_mode(
        self,
        input_data: Input,
        credentials,
        tool_functions: list[dict[str, Any]],
        graph_exec_id: str,
        node_id: str,
        node_exec_id: str,
        user_id: str,
        graph_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        execution_processor: "ExecutionProcessor",
    ):
        """Execute tools in agent mode with a loop until finished, following SmartDecisionMakerBlock pattern."""
        max_iterations = input_data.max_iterations
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

        # Build enhanced task prompt with tool descriptions
        tool_descriptions = self._build_tool_descriptions_for_claude(tool_functions)
        enhanced_task = f"""{input_data.task}

{tool_descriptions}

Complete the task step by step using the available tools as needed."""

        # Start conversation with enhanced task
        current_conversation = [{"role": "user", "content": enhanced_task}]

        while max_iterations < 0 or iteration < max_iterations:
            iteration += 1
            logger.debug(f"Claude agent mode iteration {iteration}")

            # Prepare conversation for this iteration
            iteration_conversation = list(current_conversation)

            # On the last iteration, add encouragement to finish
            if max_iterations > 0 and iteration == max_iterations:
                last_iteration_message = {
                    "role": "system",
                    "content": f"This is your last iteration ({iteration}/{max_iterations}). "
                    "Try to complete the task with the information you have. "
                    "Prefer finishing with a clear response rather than making additional tool calls.",
                }
                iteration_conversation.append(last_iteration_message)

            # Format conversation for Claude SDK
            conversation_text = self._format_conversation_for_claude(
                iteration_conversation
            )

            # Setup Claude options for this iteration
            claude_options = ClaudeAgentOptions(
                system_prompt=input_data.system_prompt,
                model=input_data.model.value,
                max_turns=1,  # Single turn per iteration
                cwd=input_data.working_directory,
                env=self._setup_environment(credentials, input_data.working_directory),
                permission_mode="bypassPermissions",
            )

            # Get Claude response
            logger.debug(f"Claude agent iteration {iteration}: Making Claude SDK call")
            try:
                claude_response = await self._attempt_claude_call_with_validation(
                    conversation_text, claude_options
                )
                logger.debug(f"Claude agent iteration {iteration}: Received response")
            except Exception as e:
                logger.error(
                    f"Claude agent iteration {iteration}: Call failed with {type(e).__name__}: {str(e)}"
                )
                yield (
                    "error",
                    f"Claude call failed in agent mode iteration {iteration}: {str(e)}",
                )
                return

            # Process tool calls
            tool_calls = self._extract_tool_calls_from_claude_message(claude_response)
            text_content = self._extract_text_content_from_claude_message(
                claude_response
            )

            # Add Claude's response to conversation
            assistant_message = {
                "role": "assistant",
                "content": text_content,
                "tool_calls": tool_calls if tool_calls else [],
            }
            current_conversation.append(assistant_message)

            # If no tool calls, we're done
            if not tool_calls:
                yield "finished", text_content
                yield "conversations", current_conversation
                return

            # Execute tools and collect responses
            tool_outputs = []
            for tool_call in tool_calls:
                # Convert tool call to ToolInfo format for AutoGPT execution
                tool_info = self._create_tool_info_from_claude_call(
                    tool_call, tool_functions
                )

                try:
                    # Execute via AutoGPT's execution manager
                    tool_response = await self._execute_single_tool_with_manager(
                        tool_info, execution_params, execution_processor
                    )
                    tool_outputs.append(tool_response)
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    error_response = _create_tool_response(
                        tool_call["id"], f"Error: {str(e)}"
                    )
                    tool_outputs.append(error_response)

            # Add tool results to conversation
            current_conversation.extend(tool_outputs)

            # Yield intermediate conversation state
            yield "conversations", current_conversation

        # If we reach max iterations, yield the current state
        if max_iterations < 0:
            yield "finished", f"Agent mode completed after {iteration} iterations"
        else:
            yield (
                "finished",
                f"Agent mode completed after {max_iterations} iterations (limit reached)",
            )
        yield "conversations", current_conversation

    async def _execute_single_call_mode(
        self,
        input_data: Input,  # Used for configuration and consistency with agent mode
        tool_functions: list[dict[str, Any]],
        enhanced_task: str,
        claude_options: ClaudeAgentOptions | None,
    ):
        """Execute single call mode and yield tool outputs for external execution."""
        # Create Claude options for single call if not provided
        if claude_options is None:
            claude_options = ClaudeAgentOptions(
                system_prompt=input_data.system_prompt,
                model=input_data.model.value,
                max_turns=1,  # Single call mode
                cwd=input_data.working_directory,
                permission_mode="bypassPermissions",
            )
        else:
            # Override max_turns to 1 for single call
            claude_options.max_turns = 1

        try:
            claude_response = await self._attempt_claude_call_with_validation(
                enhanced_task, claude_options
            )
        except Exception as e:
            yield "error", f"Claude SDK error: {str(e)}"
            yield "success", False
            return

        if claude_response:
            text_content = self._extract_text_content_from_claude_message(
                claude_response
            )
            tool_calls = self._extract_tool_calls_from_claude_message(claude_response)

            if not tool_calls:
                # No tool calls - just return the result
                yield "result", text_content
                yield "success", True
                yield "tools", []  # No tools used
                return

            # Process and yield tool calls for external execution
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = self._normalize_tool_args(
                    tool_call["function"]["arguments"]
                )

                # Find the tool definition (fallback to first if only one available)
                tool_def = next(
                    (
                        tool
                        for tool in tool_functions
                        if tool["function"]["name"] == tool_name
                    ),
                    tool_functions[0] if len(tool_functions) == 1 else None,
                )
                if not tool_def:
                    continue

                # Get field mapping and sink node ID
                field_mapping = tool_def["function"].get("_field_mapping", {})
                sink_node_id = tool_def["function"]["_sink_node_id"]
                expected_args = tool_def["function"]["parameters"].get(
                    "properties", tool_args.keys()
                )

                # Yield tool outputs like SmartDecisionMakerBlock
                for clean_arg_name in expected_args:
                    original_field_name = field_mapping.get(
                        clean_arg_name, clean_arg_name
                    )
                    arg_value = tool_args.get(clean_arg_name)

                    # Create the same emit key format as SmartDecisionMakerBlock
                    sanitized_arg_name = self.cleanup(original_field_name)
                    emit_key = f"tools_^_{sink_node_id}_~_{sanitized_arg_name}"

                    logger.debug(f"Yielding tool output: {emit_key}")
                    yield emit_key, arg_value

            # Yield conversation and tool results
            yield (
                "conversations",
                [
                    {
                        "role": "assistant",
                        "content": text_content,
                        "tool_calls": tool_calls,
                    }
                ],
            )
            yield "tools", tool_calls
            yield "success", True
            return

        # If no messages received
        yield "error", "No response from Claude Agent SDK"
        yield "success", False

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials,
        graph_id: str,
        node_id: str,
        graph_exec_id: str,
        node_exec_id: str,
        user_id: str,
        graph_version: int,
        execution_context: ExecutionContext,
        execution_processor: "ExecutionProcessor",
        **kwargs,  # Additional execution context parameters
    ) -> BlockOutput:
        _ = kwargs  # Suppress unused parameter warning
        # Validate credentials
        if credentials.provider != ProviderName.ANTHROPIC:
            error_msg = f"SmartAgentBlock requires Anthropic/Claude credentials, but received {credentials.provider} credentials. Please configure Anthropic API key credentials."
            logger.error(error_msg)
            yield "error", error_msg
            yield "success", False
            return
        # Discover connected AutoGPT tool nodes
        try:
            tool_functions = await self._create_tool_node_signatures(node_id)
        except ValueError as e:
            if "no next node" in str(e).lower():
                # Agent can work without tools - just provide Claude with reasoning capability
                tool_functions = []
                logger.info("No tools connected - running as pure Claude Agent")
            else:
                raise

        yield "tool_functions", json.dumps(tool_functions)

        # Always run Claude Agent SDK in agent mode (iterative execution)
        async for result in self._execute_tools_agent_mode(
            input_data=input_data,
            credentials=credentials,
            tool_functions=tool_functions,
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
