import asyncio
import json

from prisma.models import AgentGraph, AgentNode, AgentNodeLink, AgentNodeExecution

from autogpt_server.data.db import BaseDbModel


class Edge(BaseDbModel):
    input_node: str
    input_name: str
    output_node: str
    output_name: str

    @staticmethod
    def from_db(edge: AgentNodeLink):
        return Edge(
            id=edge.id,
            input_node=edge.agentNodeInputId,
            input_name=edge.inputName,
            output_node=edge.agentNodeOutputId,
            output_name=edge.outputName,
        )


class Node(BaseDbModel):
    block_name: str
    input_default: dict[str, str] = {}  # dict[input_name, input_value]
    input_schema: dict[str, str] = {}  # dict[input_name, type]
    output_schema: dict[str, str] = {}  # dict[output_name, type]
    input_nodes: dict[str, str] = {}  # dict[input_name, node_id]
    output_nodes: dict[str, str] = {}  # dict[output_name, node_id]

    @staticmethod
    def from_db(node: AgentNode):
        return Node(
            id=node.id,
            block_name=node.AgentBlock.name,
            input_default=json.loads(node.constantInput),
            input_schema=json.loads(node.AgentBlock.inputSchema),
            output_schema=json.loads(node.AgentBlock.outputSchema),
            input_nodes={v.outputName: v.agentNodeInputId for v in node.Input},
            output_nodes={v.inputName: v.agentNodeOutputId for v in node.Output},
        )


class Graph(BaseDbModel):
    name: str
    description: str
    nodes: list[Node]
    edges: list[Edge]

    @property
    def starting_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.input_nodes]

    @staticmethod
    def from_db(graph: AgentGraph):
        return Graph(
            id=graph.id,
            name=graph.name,
            description=graph.description,
            nodes=[Node.from_db(node) for node in graph.AgentNodes],
            edges=[
                Edge.from_db(edge) for node in graph.AgentNodes for edge in node.Output
            ],
        )


EXECUTION_NODE_INCLUDE = {
    "Input": True,
    "Output": True,
    "AgentBlock": True,
}


async def get_node(node_id: str) -> Node | None:
    node = await AgentNode.prisma().find_unique(
        where={"id": node_id},
        include=EXECUTION_NODE_INCLUDE,
    )
    return Node.from_db(node) if node else None


async def get_graph(graph_id: str) -> Graph:
    graph = await AgentGraph.prisma().find_unique(
        where={"id": graph_id},
        include={"AgentNodes": {"include": EXECUTION_NODE_INCLUDE}},
    )
    return Graph.from_db(graph) if graph else None


async def get_node_input(node: Node, exec_id: str) -> dict[str, str]:
    """
    Get execution node input data from the previous node execution result.
    Args:
        node: The execution node.
        exec_id: The execution ID.
    Returns:
        dictionary of input data, key is the input name, value is the input data.
    """
    query = AgentNodeExecution.prisma().find_many(
        where={  # type: ignore
            "executionId": exec_id,
            "agentNodeId": {"in": list(node.input_nodes.values())},
            "executionStatus": "COMPLETED",
        },
        distinct=["agentNodeId"],  # type: ignore
        order={"creationTime": "desc"},
    )

    latest_executions: dict[str, AgentNodeExecution] = {
        execution.agentNodeId: execution for execution in await query
    }

    return {
        **node.input_default,
        **{
            name: latest_executions[node_id].outputData
            for name, node_id in node.input_nodes.items()
            if node_id in latest_executions
        },
    }


async def create_graph(graph: Graph) -> Graph:
    await AgentGraph.prisma().create(
        data={
            "id": graph.id,
            "name": graph.name,
            "description": graph.description,
        }
    )

    # TODO: replace bulk creation using create_many
    await asyncio.gather(
        *[
            AgentNode.prisma().create(
                {
                    "id": node.id,
                    "agentBlockName": node.block_name,
                    "agentGraphId": graph.id,
                    "constantInput": json.dumps(node.input_default),
                }
            )
            for node in graph.nodes
        ]
    )

    # TODO: replace bulk creation using create_many
    await asyncio.gather(
        *[
            AgentNodeLink.prisma().create(
                {
                    "id": edge.id,
                    "inputName": edge.input_name,
                    "outputName": edge.output_name,
                    "agentNodeInputId": edge.input_node,
                    "agentNodeOutputId": edge.output_node,
                }
            )
            for edge in graph.edges
        ]
    )

    return await get_graph(graph.id)
