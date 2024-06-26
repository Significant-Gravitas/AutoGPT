import asyncio
import json
import uuid
from typing import Any

from prisma.models import AgentGraph, AgentNode, AgentNodeLink
from pydantic import BaseModel

from autogpt_server.data.db import BaseDbModel


class Link(BaseModel):
    name: str
    node_id: str
    
    def __init__(self, name: str, node_id: str):
        super().__init__(name=name, node_id=node_id)
        
    def __iter__(self):
        return iter((self.name, self.node_id))


class Node(BaseDbModel):
    block_id: str
    input_default: dict[str, Any] = {}  # dict[input_name, default_value]
    input_nodes: list[Link] = []  # dict[input_name, node_id]
    output_nodes: list[Link] = []  # dict[output_name, node_id]
    metadata: dict[str, Any] = {}

    @staticmethod
    def from_db(node: AgentNode):
        if not node.AgentBlock:
            raise ValueError(f"Invalid node {node.id}, invalid AgentBlock.")

        return Node(
            id=node.id,
            block_id=node.AgentBlock.id,
            input_default=json.loads(node.constantInput),
            input_nodes=[
                Link(v.sinkName, v.agentNodeSourceId)
                for v in node.Input or []
            ],
            output_nodes=[
                Link(v.sourceName, v.agentNodeSinkId)
                for v in node.Output or []
            ],
            metadata=json.loads(node.metadata),
        )

    def connect(self, node: "Node", source_name: str, sink_name: str):
        self.output_nodes.append(Link(source_name, node.id))
        node.input_nodes.append(Link(sink_name, self.id))


class Graph(BaseDbModel):
    name: str
    description: str
    nodes: list[Node]

    @property
    def starting_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.input_nodes]

    @staticmethod
    def from_db(graph: AgentGraph):
        return Graph(
            id=graph.id,
            name=graph.name or "",
            description=graph.description or "",
            nodes=[Node.from_db(node) for node in graph.AgentNodes or []],
        )


EXECUTION_NODE_INCLUDE = {
    "Input": True,
    "Output": True,
    "AgentBlock": True,
}


# --------------------- Model functions --------------------- #


async def get_node(node_id: str) -> Node | None:
    node = await AgentNode.prisma().find_unique_or_raise(
        where={"id": node_id},
        include=EXECUTION_NODE_INCLUDE,  # type: ignore
    )
    return Node.from_db(node) if node else None


async def get_graph_ids() -> list[str]:
    return [graph.id for graph in await AgentGraph.prisma().find_many()]  # type: ignore


async def get_graph(graph_id: str) -> Graph | None:
    graph = await AgentGraph.prisma().find_unique(
        where={"id": graph_id},
        include={"AgentNodes": {"include": EXECUTION_NODE_INCLUDE}},  # type: ignore
    )
    return Graph.from_db(graph) if graph else None


async def create_graph(graph: Graph) -> Graph:
    await AgentGraph.prisma().create(
        data={
            "id": graph.id,
            "name": graph.name,
            "description": graph.description,
        }
    )

    # TODO: replace bulk creation using create_many
    await asyncio.gather(*[
        AgentNode.prisma().create({
            "id": node.id,
            "agentBlockId": node.block_id,
            "agentGraphId": graph.id,
            "constantInput": json.dumps(node.input_default),
            "metadata": json.dumps(node.metadata),
        }) for node in graph.nodes
    ])

    edge_source_names = {
        (source_node.id, sink_node_id): output_name
        for source_node in graph.nodes
        for output_name, sink_node_id in source_node.output_nodes
    }
    edge_sink_names = {
        (source_node_id, sink_node.id): input_name
        for sink_node in graph.nodes
        for input_name, source_node_id in sink_node.input_nodes
    }

    # TODO: replace bulk creation using create_many
    await asyncio.gather(*[
        AgentNodeLink.prisma().create({
            "id": str(uuid.uuid4()),
            "sourceName": edge_source_names.get((input_node, output_node), ""),
            "sinkName": edge_sink_names.get((input_node, output_node), ""),
            "agentNodeSourceId": input_node,
            "agentNodeSinkId": output_node,
        })
        for input_node, output_node in edge_source_names.keys() | edge_sink_names.keys()
    ])

    if created_graph := await get_graph(graph.id):
        return created_graph

    raise ValueError(f"Failed to create graph {graph.id}.")
