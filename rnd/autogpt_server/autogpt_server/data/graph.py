import asyncio
import uuid
from pathlib import Path
from typing import Any, Literal

import prisma.types
from prisma.models import AgentGraph, AgentNode, AgentNodeLink
from pydantic import PrivateAttr

from autogpt_server.data.block import BlockInput
from autogpt_server.data.db import BaseDbModel
from autogpt_server.util import json


class Link(BaseDbModel):
    source_id: str
    sink_id: str
    source_name: str
    sink_name: str
    is_static: bool = False

    @staticmethod
    def from_db(link: AgentNodeLink):
        return Link(
            id=link.id,
            source_name=link.sourceName,
            source_id=link.agentNodeSourceId,
            sink_name=link.sinkName,
            sink_id=link.agentNodeSinkId,
            is_static=link.isStatic,
        )

    def __hash__(self):
        return hash((self.source_id, self.sink_id, self.source_name, self.sink_name))


class Node(BaseDbModel):
    block_id: str
    input_default: BlockInput = {}  # dict[input_name, default_value]
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
            metadata=json.loads(node.metadata),
        )
        obj._input_links = [Link.from_db(link) for link in node.Input or []]
        obj._output_links = [Link.from_db(link) for link in node.Output or []]
        return obj


class GraphMeta(BaseDbModel):
    version: int = 1
    is_active: bool = True
    is_template: bool = False

    name: str
    description: str

    @staticmethod
    def from_db(graph: AgentGraph):
        return GraphMeta(
            id=graph.id,
            version=graph.version,
            is_active=graph.isActive,
            is_template=graph.isTemplate,
            name=graph.name or "",
            description=graph.description or "",
        )


class Graph(GraphMeta):
    nodes: list[Node]
    links: list[Link]

    @property
    def starting_nodes(self) -> list[Node]:
        outbound_nodes = {link.sink_id for link in self.links}
        return [node for node in self.nodes if node.id not in outbound_nodes]

    @staticmethod
    def from_db(graph: AgentGraph):
        return Graph(
            **GraphMeta.from_db(graph).model_dump(),
            nodes=[Node.from_db(node) for node in graph.AgentNodes or []],
            links=list(
                {
                    Link.from_db(link)
                    for node in graph.AgentNodes or []
                    for link in (node.Input or []) + (node.Output or [])
                }
            ),
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


async def get_graphs_meta(
    filter_by: Literal["active", "template"] | None = "active"
) -> list[GraphMeta]:
    """
    Retrieves graph metadata objects.
    Default behaviour is to get all currently active graphs.

    Args:
        filter_by: An optional filter to either select templates or active graphs.

    Returns:
        list[GraphMeta]: A list of objects representing the retrieved graph metadata.
    """
    where_clause: prisma.types.AgentGraphWhereInput = {}

    if filter_by == "active":
        where_clause["isActive"] = True
    elif filter_by == "template":
        where_clause["isTemplate"] = True

    graphs = await AgentGraph.prisma().find_many(
        where=where_clause,
        distinct=["id"],
        order={"version": "desc"},
    )

    if not graphs:
        return []

    return [GraphMeta.from_db(graph) for graph in graphs]


async def get_graph(
    graph_id: str, version: int | None = None, template: bool = False
) -> Graph | None:
    """
    Retrieves a graph from the DB.
    Defaults to the version with `is_active` if `version` is not passed,
    or the latest version with `is_template` if `template=True`.

    Returns `None` if the record is not found.
    """
    where_clause: prisma.types.AgentGraphWhereInput = {
        "id": graph_id,
        "isTemplate": template,
    }
    if version is not None:
        where_clause["version"] = version
    elif not template:
        where_clause["isActive"] = True

    graph = await AgentGraph.prisma().find_first(
        where=where_clause,
        include={"AgentNodes": {"include": EXECUTION_NODE_INCLUDE}},  # type: ignore
        order={"version": "desc"},
    )
    return Graph.from_db(graph) if graph else None


async def set_graph_active_version(graph_id: str, version: int) -> None:
    updated_graph = await AgentGraph.prisma().update(
        data={"isActive": True},
        where={"graphVersionId": {"id": graph_id, "version": version}},
    )
    if not updated_graph:
        raise Exception(f"Graph #{graph_id} v{version} not found")

    # Deactivate all other versions
    await AgentGraph.prisma().update_many(
        data={"isActive": False},
        where={"id": graph_id, "version": {"not": version}},
    )


async def get_graph_all_versions(graph_id: str) -> list[Graph]:
    graph_versions = await AgentGraph.prisma().find_many(
        where={"id": graph_id},
        order={"version": "desc"},
        include={"AgentNodes": {"include": EXECUTION_NODE_INCLUDE}},  # type: ignore
    )

    if not graph_versions:
        return []

    return [Graph.from_db(graph) for graph in graph_versions]


async def create_graph(graph: Graph) -> Graph:
    await AgentGraph.prisma().create(
        data={
            "id": graph.id,
            "version": graph.version,
            "name": graph.name,
            "description": graph.description,
            "isTemplate": graph.is_template,
            "isActive": graph.is_active,
        }
    )

    await asyncio.gather(
        *[
            AgentNode.prisma().create(
                {
                    "id": node.id,
                    "agentBlockId": node.block_id,
                    "agentGraphId": graph.id,
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
                    "isStatic": link.is_static,
                }
            )
            for link in graph.links
        ]
    )

    if created_graph := await get_graph(
        graph.id, graph.version, template=graph.is_template
    ):
        return created_graph

    raise ValueError(f"Created graph {graph.id} v{graph.version} is not in DB")


# --------------------- Helper functions --------------------- #


TEMPLATES_DIR = Path(__file__).parent.parent.parent / "graph_templates"


async def import_packaged_templates() -> None:
    templates_in_db = await get_graphs_meta(filter_by="template")

    print("Loading templates...")
    for template_file in TEMPLATES_DIR.glob("*.json"):
        template_data = json.loads(template_file.read_bytes())

        template = Graph.model_validate(template_data)
        if not template.is_template:
            print(f"WARNING: pre-packaged graph file {template_file} is not a template")
            continue
        if (
            exists := next((t for t in templates_in_db if t.id == template.id), None)
        ) and exists.version >= template.version:
            continue
        await create_graph(template)
        print(f"Loaded template '{template.name}' ({template.id})")
