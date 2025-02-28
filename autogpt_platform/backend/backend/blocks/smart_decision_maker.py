import json
import logging
import re
from typing import TYPE_CHECKING, Any, List

from autogpt_libs.utils.cache import thread_cached

import backend.blocks.llm as llm
from backend.blocks.agent import AgentExecutorBlock
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    get_blocks,
)
from backend.data.model import SchemaField

if TYPE_CHECKING:
    from backend.data.graph import Graph, Link, Node

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

    # If I import Graph here, it will break with a circular import.
    def _get_tool_graph_metadata(self, node_id: str, graph: "Graph") -> List["Graph"]:
        """
        Retrieves metadata for tool graphs linked to a specified node within a graph.

        This method identifies the tool links connected to the given node_id and fetches
        the metadata for each linked tool graph from the database.

        Args:
            node_id (str): The ID of the node for which tool graph metadata is to be retrieved.
            graph (Any): The graph object containing nodes and links.

        Returns:
            List[Any]: A list of metadata for the tool graphs linked to the specified node.
        """
        db_client = get_database_manager_client()
        graph_meta = []

        tool_links = {
            link.sink_id
            for link in graph.links
            if link.source_name.startswith("tools_^_") and link.source_id == node_id
        }

        for link_id in tool_links:
            node = next((node for node in graph.nodes if node.id == link_id), None)
            if node and node.block_id == AgentExecutorBlock().id:
                node_graph_meta = db_client.get_graph_metadata(
                    node.input_default["graph_id"], node.input_default["graph_version"]
                )
                if node_graph_meta:
                    graph_meta.append(node_graph_meta)

        return graph_meta

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
        block = get_blocks()[sink_node.block_id]
        if not block:
            raise ValueError(f"Block not found: {sink_node.block_id}")

        tool_function: dict[str, Any] = {
            "name": re.sub(r"[^a-zA-Z0-9_-]", "_", block().name).lower(),
            "description": block().description,
        }

        properties = {}
        required = []

        for link in links:
            sink_block_input_schema = block().input_schema
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
        sink_node: "Node", links: list["Link"], tool_graph_metadata: list["Graph"]
    ) -> dict[str, Any]:
        """
        Creates a function signature for an agent node.

        Args:
            sink_node: The agent node for which to create a function signature.
            links: The list of links connected to the sink node.
            tool_graph_metadata: List of metadata for available tool graphs.

        Returns:
            A dictionary representing the function signature in the format expected by LLM tools.

        Raises:
            ValueError: If the graph metadata for the specified graph_id and graph_version is not found.
        """
        graph_id = sink_node.input_default["graph_id"]
        graph_version = sink_node.input_default["graph_version"]

        sink_graph_meta = next(
            (
                meta
                for meta in tool_graph_metadata
                if meta.id == graph_id and meta.version == graph_version
            ),
            None,
        )

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
    def _create_function_signature(
        # If I import Graph here, it will break with a circular import.
        node_id: str,
        graph: "Graph",
        tool_graph_metadata: List["Graph"],
    ) -> list[dict[str, Any]]:
        """
        Creates function signatures for tools linked to a specified node within a graph.

        This method filters the graph links to identify those that are tools and are
        connected to the given node_id. It then constructs function signatures for each
        tool based on the metadata and input schema of the linked nodes.

        Args:
            node_id (str): The ID of the node for which tool function signatures are to be created.
            graph (Any): The graph object containing nodes and links.
            tool_graph_metadata (List[Any]): Metadata for the tool graphs, used to retrieve
                                             names and descriptions for the tools.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each representing a function signature
                                  for a tool, including its name, description, and parameters.

        Raises:
            ValueError: If no tool links are found for the specified node_id, or if a sink node
                        or its metadata cannot be found.
        """
        # Filter the graph links to find those that are tools and are linked to the specified node_id
        tool_links = [
            link
            for link in graph.links
            # NOTE: Maybe we can do a specific database call to only get relevant nodes
            # async def get_connected_output_nodes(source_node_id: str) -> list[Node]:
            #     links = await AgentNodeLink.prisma().find_many(
            #         where={"agentNodeSourceId": source_node_id},
            #         include={"AgentNode": {"include": AGENT_NODE_INCLUDE}},
            #     )
            #     return [NodeModel.from_db(link.AgentNodeSink) for link in links]
            if link.source_name.startswith("tools_^_") and link.source_id == node_id
        ]

        if not tool_links:
            raise ValueError(
                f"Expected at least one tool link in the graph. Node ID: {node_id}. Graph: {graph.links}"
            )

        return_tool_functions = []

        grouped_tool_links = {}

        for link in tool_links:
            grouped_tool_links.setdefault(link.sink_id, []).append(link)

        for _, links in grouped_tool_links.items():
            sink_node = next(
                (node for node in graph.nodes if node.id == links[0].sink_id), None
            )

            if not sink_node:
                raise ValueError(f"Sink node not found: {links[0].sink_id}")

            if sink_node.block_id == AgentExecutorBlock().id:
                return_tool_functions.append(
                    SmartDecisionMakerBlock._create_agent_function_signature(
                        sink_node, links, tool_graph_metadata
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
        db_client = get_database_manager_client()

        # Retrieve the current graph and node details
        graph = db_client.get_graph(
            graph_id=graph_id,
            user_id=user_id,
            ignore_ownership_if_listed_in_marketplace=True,
        )

        if not graph:
            raise ValueError(
                f"The currently running graph that is executing this node is not found {graph_id}"
            )

        tool_graph_metadata = self._get_tool_graph_metadata(node_id, graph)

        tool_functions = self._create_function_signature(
            node_id, graph, tool_graph_metadata
        )

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

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                for arg_name, arg_value in tool_args.items():
                    yield f"tools_^_{tool_name}_{arg_name}".lower(), arg_value
