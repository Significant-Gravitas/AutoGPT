import logging
from typing import Any, List

from autogpt_libs.utils.cache import thread_cached

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, BlockType
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
        prompt: str = SchemaField(description="The text to print to the console.")

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
            block_type=BlockType.AGENT,
            input_schema=SmartDecisionMakerBlock.Input,
            output_schema=SmartDecisionMakerBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=[],
        )

    # If I import Graph here, it will break with a circular import.
    def get_tool_graph_metadata(self, node_id: str, graph: Any) -> List[Any]:
        db_client = get_database_manager_client()
        graph_meta = []

        tool_links = {
            link.sink_id
            for link in graph.links
            if link.source_name.startswith("tools_") and link.source_id == node_id
        }

        for link_id in tool_links:
            node = next((node for node in graph.nodes if node.id == link_id), None)
            if node:
                node_graph_meta = db_client.get_graph_metadata(
                    node.input_default["graph_id"], node.input_default["graph_version"]
                )
                if node_graph_meta:
                    graph_meta.append(node_graph_meta)

        return graph_meta

    @staticmethod
    def _create_function_signature(
        # If I import Graph here, it will break with a circular import.
        node_id: str,
        graph: Any,
        tool_graph_metadata: List[Any],
    ) -> list[dict[str, Any]]:
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

        if not tool_links:
            raise ValueError(
                f"Expected at least one tool link in the graph. Node ID: {node_id}. Graph: {graph.links}"
            )

        return_tool_functions = []

        grouped_tool_links = {}

        for link in tool_links:
            grouped_tool_links.setdefault(link.sink_id, []).append(link)

        logger.warning(f"Grouped tool links: {grouped_tool_links}")

        for tool_name, links in grouped_tool_links.items():
            sink_node = next(
                (node for node in graph.nodes if node.id == links[0].sink_id), None
            )

            if not sink_node:
                raise ValueError(f"Sink node not found: {links[0].sink_id}")

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
                "name": sink_graph_meta.name,
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
                properties[link.sink_name] = {
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

            return_tool_functions.append(
                {"type": "function", "function": tool_function}
            )
        return return_tool_functions

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
        graph = db_client.get_graph(graph_id=graph_id, user_id=user_id)

        if not graph:
            raise ValueError(
                f"The currently running graph that is executing this node is not found {graph_id}"
            )

        tool_graph_metadata = self.get_tool_graph_metadata(node_id, graph)

        tool_functions = self._create_function_signature(
            node_id, graph, tool_graph_metadata
        )

        logger.warning(f"Tool functions: {tool_functions}")

        yield "tools_sample_tool_input_1", "Hello"
        yield "tools_sample_tool_input_2", "World"
