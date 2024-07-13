import asyncio
import uuid
from typing import Any

import prisma.types
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
            input_default=json.loads(node.constantInput),  # type: ignore
            metadata=json.loads(node.metadata),  # type: ignore
        )
        obj._input_links = [Link.from_db(link) for link in node.Input or []]
        obj._output_links = [Link.from_db(link) for link in node.Output or []]
        return obj


class Graph(BaseModel):
    graph_id: str = ""
    version: int = 1
    is_active: bool = True
    is_template: bool = False
    name: str
    description: str
    nodes: list[Node]
    links: list[Link]

    def __init__(self, graph_id: str = "", **data: Any):
        data["graph_id"] = graph_id or str(uuid.uuid4())
        super().__init__(**data)

    @property
    def starting_nodes(self) -> list[Node]:
        outbound_nodes = {link.sink_id for link in self.links}
        return [node for node in self.nodes if node.id not in outbound_nodes]

    @staticmethod
    def from_db(graph: AgentGraph):
        return Graph(
            id=graph.graph_id,
            version=graph.version,
            is_active=graph.is_active,
            is_template=graph.is_template,
            name=graph.name or "",
            description=graph.description or "",
            nodes=[Node.from_db(node) for node in graph.AgentNodes or []],
            links=list(
                {
                    Link.from_db(link)
                    for node in graph.AgentNodes or []
                    for link in (node.Input or []) + (node.Output or [])
                }
            ),
        )


class GraphMeta(BaseModel):
    graph_id: str
    version: int
    name: str
    description: str
    is_active: bool
    is_template: bool

    @staticmethod
    def from_db(graph: AgentGraph):
        return GraphMeta(
            graph_id=graph.graph_id,
            version=graph.version,
            name=graph.name or "",
            description=graph.description or "",
            is_active=graph.is_active,
            is_template=graph.is_template,
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


# TODO: Delete this
async def get_graph_ids() -> list[str]:
    return [
        graph.graph_id
        for graph in await AgentGraph.prisma().find_many(where={"is_template": False})
    ]  # type: ignore


async def get_graphs_meta(
    is_template: bool = False, is_active: bool = True, latest_version: bool = True
) -> list[GraphMeta]:
    """
    Retrieves graph metadata based on the provided parameters.
    Defauly behaviour is to get the latest active graph for each graph_id.

    Args:
        is_template (bool): Indicates whether the graph is a template.
        is_active (bool): Indicates whether the graph is active.
        latest_version (bool): Indicates whether to retrieve only the latest version of each graph.

    Returns:
        list[GraphMeta]: A list of GraphMeta objects representing the retrieved graph metadata.
    """
    where_clause = {"is_template": is_template}

    if not is_template:
        where_clause["is_active"] = is_active

    where_clause = prisma.types.AgentGraphWhereInput(**where_clause)  # type: ignore
    order_by = {"version": "desc"}  # type: ignore

    graphs = await AgentGraph.prisma().find_many(
        where=where_clause,
        distinct=["graph_id"] if latest_version else None,
        order=order_by,  # type: ignore
    )

    if not graphs:
        return []

    return [GraphMeta.from_db(graph) for graph in graphs]  # type: ignore


async def get_graph(graph_id: str, version: int | None = None) -> Graph:
    """
    Retrieves a graph from the DB.
    Defaults to the current active version if `version` is not passed.
    """
    where_clause: prisma.types.AgentGraphWhereInput = {"graph_id": graph_id}
    if version is not None:
        where_clause["version"] = version
    else:
        where_clause["is_active"] = True

    graph = await AgentGraph.prisma().find_first_or_raise(
        where=where_clause,
        include={"AgentNodes": {"include": EXECUTION_NODE_INCLUDE}},  # type: ignore
        distinct=["graph_id"] if not version else None,
        order={"version": "desc"},
    )
    return Graph.from_db(graph)


async def deactivate_other_graph_versions(graph_id: str, except_version: int) -> bool:
    success = await AgentGraph.prisma().update_many(
        data={"is_active": False},
        where={"graph_id": graph_id, "version": {"not": except_version}},
    )
    return success is not None


async def get_graph_all_versions(graph_id: str) -> list[Graph]:
    graph_history = await AgentGraph.prisma().find_many(
        where={"graph_id": graph_id},
        order={"version": "desc"},
        include={"AgentNodes": {"include": EXECUTION_NODE_INCLUDE}},  # type: ignore
    )

    if not graph_history:
        return []

    return [Graph.from_db(graph) for graph in graph_history]


async def create_graph(graph: Graph) -> Graph:
    await AgentGraph.prisma().create(
        data={
            "graph_id": graph.graph_id,
            "version": graph.version,
            "name": graph.name,
            "description": graph.description,
            "is_template": graph.is_template,
            "is_active": graph.is_active,
        }
    )

    # TODO: replace bulk creation using create_many
    await asyncio.gather(
        *[
            AgentNode.prisma().create(
                {
                    "id": node.id,
                    "agentBlockId": node.block_id,
                    "agentGraphId": graph.graph_id,
                    "agentGraphVersion": graph.version,
                    "constantInput": json.dumps(node.input_default),
                    "metadata": json.dumps(node.metadata),
                }
            )
            for node in graph.nodes
        ]
    )

    await asyncio.gather(
        *[
            AgentNodeLink.prisma().create(
                {
                    "id": str(uuid.uuid4()),
                    "sourceName": link.source_name,
                    "sinkName": link.sink_name,
                    "agentNodeSourceId": link.source_id,
                    "agentNodeSinkId": link.sink_id,
                }
            )
            for link in graph.links
        ]
    )

    if created_graph := await get_graph(graph.graph_id, graph.version):
        return created_graph

    raise ValueError(f"Failed to create graph {graph.graph_id}:{graph.version}.")
