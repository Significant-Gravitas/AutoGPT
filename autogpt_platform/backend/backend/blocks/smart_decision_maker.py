import json
import logging
import re
from typing import TYPE_CHECKING, Any

from autogpt_libs.utils.cache import thread_cached

import backend.blocks.llm as llm
from backend.blocks.agent import AgentExecutorBlock
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    get_block,
)
from backend.data.model import SchemaField

if TYPE_CHECKING:
    from backend.data.graph import Link, Node

logger = logging.getLogger(__name__)


@thread_cached
def get_database_manager_client():
    from backend.executor import DatabaseManager
    from backend.util.service import get_service_client

    return get_service_client(DatabaseManager)


class SmartDecisionMakerBlock(Block):
    """
    A block that uses a language model to make smart decisions based on a given prompt.
    """

    class Input(BlockSchema):
        prompt: str = SchemaField(
            description="The prompt to send to the language model.",
            placeholder="Enter your prompt here...",
        )
        model: llm.LlmModel = SchemaField(
            title="LLM Model",
            default=llm.LlmModel.GPT4O,
            description="The language model to use for answering the prompt.",
            advanced=False,
        )
        credentials: llm.AICredentials = llm.AICredentialsField()
        sys_prompt: str = SchemaField(
            title="System Prompt",
            default="Thinking carefully step by step decide which function to call. Always choose a function call from the list of function signatures.",
            description="The system prompt to provide additional context to the model.",
        )
        conversation_history: list[llm.Message] = SchemaField(
            default=[],
            description="The conversation history to provide context for the prompt.",
        )
        retry: int = SchemaField(
            title="Retry Count",
            default=3,
            description="Number of times to retry the LLM call if the response does not match the expected format.",
        )
        prompt_values: dict[str, str] = SchemaField(
            advanced=False,
            default={},
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

    class Output(BlockSchema):

        prompt: str = SchemaField(description="The prompt sent to the language model.")
        error: str = SchemaField(description="Error message if the API call failed.")
        function_signatures: list[dict[str, Any]] = SchemaField(
            description="The function signatures that are sent to the language model."
        )
        tools: Any = SchemaField(description="The tools that are available to use.")
        finished: str = SchemaField(
            description="The finished message to display to the user."
        )
        conversations: list[llm.Message] = SchemaField(
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
    def _create_block_function_signature(
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
        block = get_block(sink_node.block_id)
        if not block:
            raise ValueError(f"Block not found: {sink_node.block_id}")

        tool_function: dict[str, Any] = {
            "name": re.sub(r"[^a-zA-Z0-9_-]", "_", block.name).lower(),
            "description": block.description,
        }

        properties = {}
        required = []

        for link in links:
            sink_block_input_schema = block.input_schema
            description = (
                sink_block_input_schema.model_fields[link.sink_name].description
                if link.sink_name in sink_block_input_schema.model_fields
                and sink_block_input_schema.model_fields[link.sink_name].description
                else f"The {link.sink_name} of the tool"
            )
            properties[link.sink_name.lower()] = {
                "type": "string",
                "description": description,
            }

        tool_function["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
            "strict": True,
        }

        return {"type": "function", "function": tool_function}

    @staticmethod
    def _create_agent_function_signature(
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

        db_client = get_database_manager_client()
        sink_graph_meta = db_client.get_graph_metadata(graph_id.graph_version)
        if not sink_graph_meta:
            raise ValueError(
                f"Sink graph metadata not found: {graph_id} {graph_version}"
            )

        tool_function: dict[str, Any] = {
            "name": re.sub(r"[^a-zA-Z0-9_-]", "_", sink_graph_meta.name).lower(),
            "description": sink_graph_meta.description,
        }

        properties = {}
        required = []

        for link in links:
            sink_block_input_schema = sink_node.input_default["input_schema"]
            description = (
                sink_block_input_schema["properties"][link.sink_name]["description"]
                if "description"
                in sink_block_input_schema["properties"][link.sink_name]
                else f"The {link.sink_name} of the tool"
            )
            properties[link.sink_name.lower()] = {
                "type": "string",
                "description": description,
            }

        tool_function["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
            "strict": True,
        }

        return {"type": "function", "function": tool_function}

    @staticmethod
    def _create_function_signature(node_id: str) -> list[dict[str, Any]]:
        """
        Creates function signatures for tools linked to a specified node within a graph.

        This method filters the graph links to identify those that are tools and are
        connected to the given node_id. It then constructs function signatures for each
        tool based on the metadata and input schema of the linked nodes.

        Args:
            node_id: The node_id for which to create function signatures.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each representing a function signature
                                  for a tool, including its name, description, and parameters.

        Raises:
            ValueError: If no tool links are found for the specified node_id, or if a sink node
                        or its metadata cannot be found.
        """
        db_client = get_database_manager_client()
        tools = [
            (link, node)
            for link, node in db_client.get_connected_output_nodes(node_id)
            if link.source_name.startswith("tools_^_") and link.source_id == node_id
        ]
        if not tools:
            raise ValueError("There is no next node to execute.")

        return_tool_functions = []

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
                return_tool_functions.append(
                    SmartDecisionMakerBlock._create_agent_function_signature(
                        sink_node, links
                    )
                )
            else:
                return_tool_functions.append(
                    SmartDecisionMakerBlock._create_block_function_signature(
                        sink_node, links
                    )
                )

        return return_tool_functions

    def run(
        self,
        input_data: Input,
        *,
        credentials: llm.APIKeyCredentials,
        graph_id: str,
        node_id: str,
        graph_exec_id: str,
        node_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        tool_functions = self._create_function_signature(node_id)

        prompt = [p.model_dump() for p in input_data.conversation_history]

        values = input_data.prompt_values
        if values:
            input_data.prompt = llm.fmt.format_string(input_data.prompt, values)
            input_data.sys_prompt = llm.fmt.format_string(input_data.sys_prompt, values)

        if input_data.sys_prompt:
            prompt.append({"role": "system", "content": input_data.sys_prompt})

        if input_data.prompt:
            prompt.append({"role": "user", "content": input_data.prompt})

        response = llm.llm_call(
            credentials=credentials,
            llm_model=input_data.model,
            prompt=prompt,
            json_format=False,
            max_tokens=input_data.max_tokens,
            tools=tool_functions,
            ollama_host=input_data.ollama_host,
        )

        if not response.tool_calls:
            yield "finished", f"No Decision Made finishing task: {response.response}"
            assistant_response = response.response
        else:
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                for arg_name, arg_value in tool_args.items():
                    yield f"tools_^_{tool_name}_{arg_name}".lower(), arg_value

            assistant_response = "\n".join(
                f"[{c.function.name}] called with arguments: {c.function.arguments}"
                for c in response.tool_calls
            )

        input_data.conversation_history.append(
            llm.Message(role=llm.MessageRole.ASSISTANT, content=assistant_response)
        )
        yield "conversations", input_data.conversation_history
