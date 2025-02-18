import logging
from typing import Any

from autogpt_libs.utils.cache import thread_cached

import backend.blocks
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.graph import Graph
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)


@thread_cached
def get_database_manager_client():
    from backend.executor import DatabaseManager
    from backend.util.service import get_service_client

    return get_service_client(DatabaseManager)


class SmartDecisionMakerBlock(Block):
    # Note: Currently proving out the concept of determining the inputs a tool takes

    class Input(BlockSchema):
        # Note: This is a placeholder for the actual input schema
        text: str = SchemaField(description="The text to print to the console.")

    class Output(BlockSchema):
        # Starting with a single tool.
        tools: dict[str, dict[str, Any]] = SchemaField(
            description="The tools that are available to use."
        )

    def __init__(self):
        super().__init__(
            id="3b191d9f-356f-482d-8238-ba04b6d18381",
            description="Uses AI to intelligently decide what tool to use.",
            categories={BlockCategory.BASIC},
            input_schema=SmartDecisionMakerBlock.Input,
            output_schema=SmartDecisionMakerBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=[
                (
                    "tools",
                    {
                        "add_to_dictionary": {
                            "key": "greeting",
                            "value": "Hello, World!",
                        },
                        "print_to_console": {
                            "text": "Hello, World!",
                        },
                    },
                )
            ],
        )

    @staticmethod
    def _create_function_signature(node_id: str, graph: Graph) -> list[dict[str, Any]]:
        """
        Creates a list of function signatures for tools linked to a specific node in a graph.

        This method identifies all tool links associated with a given node ID within a graph,
        groups them by tool name, and constructs a function signature for each tool. Each
        function signature includes the tool's name, description, and parameters required
        for its execution.

        Parameters:
        - node_id (str): The ID of the node for which tool function signatures are to be created.
        - graph (GraphModel): The graph model containing nodes and links.

        Returns:
        - list[dict[str, Any]]: A list of dictionaries, each representing a tool function signature.
          Each dictionary contains:
            - "name": The name of the tool.
            - "description": A description of the tool or its multi-step process.
            - "parameters": A dictionary detailing the parameters required by the tool, including:
                - "type": The data type of the parameter.
                - "description": A description of the parameter.

        Raises:
        - ValueError: If no tool links are found for the given node ID or if no tool sink nodes are identified.
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
            if link.source_name.startswith("tools_") and link.source_id == node_id
        ]

        node_block_map = {node.id: node.block_id for node in graph.nodes}

        if not tool_links:
            raise ValueError(
                f"Expected at least one tool link in the graph. Node ID: {node_id}. Graph: {graph.links}"
            )

        tool_functions = []
        grouped_tool_links = {}

        for link in tool_links:
            grouped_tool_links.setdefault(link.sink_id, []).append(link)

        logger.warning(f"Grouped tool links: {grouped_tool_links}")

        for tool_name, links in grouped_tool_links.items():
            tool_sink_nodes = {link.sink_id for link in links}
            tool_function = {"name": tool_name}

            if len(tool_sink_nodes) == 1:
                tool_block = backend.blocks.AVAILABLE_BLOCKS[
                    node_block_map[next(iter(tool_sink_nodes))]
                ]
                tool_function["name"] = tool_block().name
                tool_function["description"] = tool_block().description
            elif len(tool_sink_nodes) > 1:
                tool_blocks = [
                    backend.blocks.AVAILABLE_BLOCKS[node_block_map[node_id]]
                    for node_id in tool_sink_nodes
                ]
                tool_function["name"] = tool_blocks[0]().name
                tool_function["description"] = (
                    "This tool is a multi-step tool that can be used to perform a task. "
                    "It includes blocks that do: "
                    + ", ".join(
                        block.name
                        for block in tool_blocks
                        if isinstance(block.name, str)
                    )
                )
            else:
                raise ValueError(
                    f"Expected at least one tool link in the graph: {tool_links}"
                )

            properties = {}
            required = []

            for link in links:
                required.append(link.sink_name)
                sink_block = backend.blocks.AVAILABLE_BLOCKS[
                    node_block_map[link.sink_id]
                ]
                sink_block_input = sink_block().input_schema.__fields__[link.sink_name]
                sink_type = sink_block_input.annotation
                sink_description = sink_block_input.description

                if sink_type not in ["string", "number", "boolean", "object"]:
                    sink_type = "string"

                properties[link.sink_name] = {
                    "type": sink_type,
                    "description": sink_description,
                }

            tool_function["parameters"] = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
                "strict": True,
            }

            tool_functions.append({"type": "function", "function": tool_function})

        return tool_functions

    def run(
        self,
        input_data: Input,
        *,
        graph_id: str,
        node_id: str,
        graph_exec_id: str,
        node_exec_id: str,
        user_id: str,
        **kwargs,
    ) -> BlockOutput:
        db_client = get_database_manager_client()

        # Retrieve the current graph and node details
        graph = db_client.get_graph(graph_id)

        if not graph:
            raise ValueError(f"Graph not found {graph_id}")

        tool_functions = self._create_function_signature(node_id, graph)

        yield "tools_#_add_to_dictionary_#_key", tool_functions
