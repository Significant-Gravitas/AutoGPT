import asyncio
import uuid

from typing import Any
from prisma.models import AgentGraph, AgentNode, AgentNodeLink
from pydantic import BaseModel, PrivateAttr

from autogpt_server.data.db import BaseDbModel
from autogpt_server.util import json


class Link(BaseModel):
    source_id: str
    sink_id: str
    source_name: str
    sink_name: str

    def __init__(self, source_id: str, sink_id: str, source_name: str, sink_name: str):
        super().__init__(
            source_id=source_id,
            sink_id=sink_id,
            source_name=source_name,
            sink_name=sink_name,
        )

    @staticmethod
    def from_db(link: AgentNodeLink):
        return Link(
            source_name=link.sourceName,
            source_id=link.agentNodeSourceId,
            sink_name=link.sinkName,
            sink_id=link.agentNodeSinkId,
        )

    def __hash__(self):
        return hash((self.source_id, self.sink_id, self.source_name, self.sink_name))


class Node(BaseDbModel):
    block_id: str
    input_default: dict[str, Any] = {}  # dict[input_name, default_value]
    metadata: dict[str, Any] = {}

    _input_links: list[Link] = PrivateAttr(default=[])
    _output_links: list[Link] = PrivateAttr(default=[])

    @property
    def input_links(self) -> list[Link]:
        return self._input_links

    @property
    def output_links(self) -> list[Link]:
        return self._output_links

    @staticmethod
    def from_db(node: AgentNode):
        if not node.AgentBlock:
            raise ValueError(f"Invalid node {node.id}, invalid AgentBlock.")
        obj = Node(
            id=node.id,
            block_id=node.AgentBlock.id,
            input_default=json.loads(node.constantInput),
            metadata=json.loads(node.metadata)
        )
        obj._input_links = [Link.from_db(link) for link in node.Input or []]
        obj._output_links = [Link.from_db(link) for link in node.Output or []]
        return obj


class Graph(BaseDbModel):
    name: str
    description: str
    nodes: list[Node]
    links: list[Link]

    @property
    def starting_nodes(self) -> list[Node]:
        outbound_nodes = {link.sink_id for link in self.links}
        return [node for node in self.nodes if node.id not in outbound_nodes]

    @staticmethod
    def from_db(graph: AgentGraph):
        return Graph(
            id=graph.id,
            name=graph.name or "",
            description=graph.description or "",
            nodes=[Node.from_db(node) for node in graph.AgentNodes or []],
            links=list({
                Link.from_db(link)
                for node in graph.AgentNodes or []
                for link in (node.Input or []) + (node.Output or [])
            })
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

    await asyncio.gather(*[
        AgentNodeLink.prisma().create({
            "id": str(uuid.uuid4()),
            "sourceName": link.source_name,
            "sinkName": link.sink_name,
            "agentNodeSourceId": link.source_id,
            "agentNodeSinkId": link.sink_id,
        })
        for link in graph.links
    ])

    if created_graph := await get_graph(graph.id):
        return created_graph

    raise ValueError(f"Failed to create graph {graph.id}.")
