import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import prisma.types
from prisma.models import AgentGraph, AgentGraphExecution, AgentNode, AgentNodeLink
from prisma.types import AgentGraphInclude
from pydantic import BaseModel, PrivateAttr
from pydantic_core import PydanticUndefinedType

from backend.blocks.basic import AgentInputBlock, AgentOutputBlock
from backend.data.block import BlockInput, get_block, get_blocks
from backend.data.db import BaseDbModel, transaction
from backend.data.execution import ExecutionStatus
from backend.util import json

logger = logging.getLogger(__name__)


class InputSchemaItem(BaseModel):
    node_id: str
    description: str | None = None
    title: str | None = None


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


class ExecutionMeta(BaseDbModel):
    execution_id: str
    started_at: datetime
    ended_at: datetime
    duration: float
    total_run_time: float
    status: ExecutionStatus

    @staticmethod
    def from_agent_graph_execution(execution: AgentGraphExecution):
        now = datetime.now(timezone.utc)
        start_time = execution.startedAt or execution.createdAt
        end_time = execution.updatedAt or now
        duration = (end_time - start_time).total_seconds()

        total_run_time = 0
        if execution.AgentNodeExecutions:
            for node_execution in execution.AgentNodeExecutions:
                node_start = node_execution.startedTime or now
                node_end = node_execution.endedTime or now
                total_run_time += (node_end - node_start).total_seconds()

        return ExecutionMeta(
            id=execution.id,
            execution_id=execution.id,
            started_at=start_time,
            ended_at=end_time,
            duration=duration,
            total_run_time=total_run_time,
            status=ExecutionStatus(execution.executionStatus),
        )


class GraphMeta(BaseDbModel):
    version: int = 1
    is_active: bool = True
    is_template: bool = False
    name: str
    description: str
    executions: list[ExecutionMeta] | None = None

    @staticmethod
    def from_db(graph: AgentGraph):
        if graph.AgentGraphExecution:
            executions = [
                ExecutionMeta.from_agent_graph_execution(execution)
                for execution in graph.AgentGraphExecution
            ]
        else:
            executions = None

        return GraphMeta(
            id=graph.id,
            version=graph.version,
            is_active=graph.isActive,
            is_template=graph.isTemplate,
            name=graph.name or "",
            description=graph.description or "",
            executions=executions,
        )


class Graph(GraphMeta):
    nodes: list[Node]
    links: list[Link]
    subgraphs: dict[str, list[str]] = {}  # subgraph_id -> [node_id]

    @property
    def starting_nodes(self) -> list[Node]:
        outbound_nodes = {link.sink_id for link in self.links}
        input_nodes = {
            v.id
            for v in self.nodes
            if isinstance(get_block(v.block_id), AgentInputBlock)
        }
        return [
            node
            for node in self.nodes
            if node.id not in outbound_nodes or node.id in input_nodes
        ]

    @property
    def ending_nodes(self) -> list[Node]:
        return [
            v for v in self.nodes if isinstance(get_block(v.block_id), AgentOutputBlock)
        ]

    @property
    def subgraph_map(self) -> dict[str, str]:
        """
        Returns a mapping of node_id to subgraph_id.
        A node in the main graph will be mapped to the graph's id.
        """
        subgraph_map = {
            node_id: subgraph_id
            for subgraph_id, node_ids in self.subgraphs.items()
            for node_id in node_ids
        }
        subgraph_map.update(
            {node.id: self.id for node in self.nodes if node.id not in subgraph_map}
        )
        return subgraph_map

    def reassign_ids(self, reassign_graph_id: bool = False):
        """
        Reassigns all IDs in the graph to new UUIDs.
        This method can be used before storing a new graph to the database.
        """
        self.validate_graph()

        id_map = {
            **{node.id: str(uuid.uuid4()) for node in self.nodes},
            **{subgraph_id: str(uuid.uuid4()) for subgraph_id in self.subgraphs},
        }

        if reassign_graph_id:
            self.id = str(uuid.uuid4())

        for node in self.nodes:
            node.id = id_map[node.id]

        for link in self.links:
            link.source_id = id_map[link.source_id]
            link.sink_id = id_map[link.sink_id]

        self.subgraphs = {
            id_map[subgraph_id]: [id_map[node_id] for node_id in node_ids]
            for subgraph_id, node_ids in self.subgraphs.items()
        }

    def validate_graph(self, for_run: bool = False):
        def sanitize(name):
            return name.split("_#_")[0].split("_@_")[0].split("_$_")[0]

        # Nodes: required fields are filled or connected, except for InputBlock.
        for node in self.nodes:
            block = get_block(node.block_id)
            if block is None:
                raise ValueError(f"Invalid block {node.block_id} for node #{node.id}")

            if not for_run:
                continue  # Skip input completion validation, unless when executing.

            provided_inputs = set(
                [sanitize(name) for name in node.input_default]
                + [sanitize(link.sink_name) for link in node.input_links]
            )
            for name in block.input_schema.get_required_fields():
                if name not in provided_inputs and not isinstance(
                    block, AgentInputBlock
                ):
                    raise ValueError(
                        f"Node {block.name} #{node.id} required input missing: `{name}`"
                    )
        node_map = {v.id: v for v in self.nodes}

        def is_static_output_block(nid: str) -> bool:
            bid = node_map[nid].block_id
            b = get_block(bid)
            return b.static_output if b else False

        def is_input_output_block(nid: str) -> bool:
            bid = node_map[nid].block_id
            b = get_block(bid)
            return isinstance(b, AgentInputBlock) or isinstance(b, AgentOutputBlock)

        # subgraphs: all nodes in subgraph must be present in the graph.
        for subgraph_id, node_ids in self.subgraphs.items():
            for node_id in node_ids:
                if node_id not in node_map:
                    raise ValueError(f"Subgraph {subgraph_id}'s node {node_id} invalid")
        subgraph_map = self.subgraph_map

        # Links: links are connected and the connected pin data type are compatible.
        for link in self.links:
            source = (link.source_id, link.source_name)
            sink = (link.sink_id, link.sink_name)
            suffix = f"Link {source} <-> {sink}"

            for i, (node_id, name) in enumerate([source, sink]):
                node = node_map.get(node_id)
                if not node:
                    raise ValueError(
                        f"{suffix}, {node_id} is invalid node id, available nodes: {node_map.keys()}"
                    )

                block = get_block(node.block_id)
                if not block:
                    blocks = {v.id: v.name for v in get_blocks().values()}
                    raise ValueError(
                        f"{suffix}, {node.block_id} is invalid block id, available blocks: {blocks}"
                    )

                sanitized_name = sanitize(name)
                if i == 0:
                    fields = f"Valid output fields: {block.output_schema.get_fields()}"
                else:
                    fields = f"Valid input fields: {block.input_schema.get_fields()}"
                if sanitized_name not in fields:
                    raise ValueError(f"{suffix}, `{name}` invalid, {fields}")

            if (
                subgraph_map.get(link.source_id) != subgraph_map.get(link.sink_id)
                and not is_input_output_block(link.source_id)
                and not is_input_output_block(link.sink_id)
            ):
                raise ValueError(f"{suffix}, Connecting nodes from different subgraph.")

            if is_static_output_block(link.source_id):
                link.is_static = True  # Each value block output should be static.

            # TODO: Add type compatibility check here.

    def get_input_schema(self) -> list[InputSchemaItem]:
        """
        Walks the graph and returns all the inputs that are either not:
        - static
        - provided by parent node
        """
        input_schema = []
        for node in self.nodes:
            block = get_block(node.block_id)
            if not block:
                continue

            for input_name, input_schema_item in (
                block.input_schema.jsonschema().get("properties", {}).items()
            ):
                # Check if the input is not static and not provided by a parent node
                if (
                    input_name not in node.input_default
                    and not any(
                        link.sink_name == input_name for link in node.input_links
                    )
                    and isinstance(
                        block.input_schema.model_fields.get(input_name).default,
                        PydanticUndefinedType,
                    )
                ):
                    input_schema.append(
                        InputSchemaItem(
                            node_id=node.id,
                            description=input_schema_item.get("description"),
                            title=input_schema_item.get("title"),
                        )
                    )

        return input_schema

    @staticmethod
    def from_db(graph: AgentGraph, hide_credentials: bool = False):
        nodes = [
            *(graph.AgentNodes or []),
            *(
                node
                for subgraph in graph.AgentSubGraphs or []
                for node in subgraph.AgentNodes or []
            ),
        ]
        return Graph(
            **GraphMeta.from_db(graph).model_dump(),
            nodes=[Graph._process_node(node, hide_credentials) for node in nodes],
            links=list(
                {
                    Link.from_db(link)
                    for node in nodes
                    for link in (node.Input or []) + (node.Output or [])
                }
            ),
            subgraphs={
                subgraph.id: [node.id for node in subgraph.AgentNodes or []]
                for subgraph in graph.AgentSubGraphs or []
            },
        )

    @staticmethod
    def _process_node(node: AgentNode, hide_credentials: bool) -> Node:
        node_dict = node.model_dump()
        if hide_credentials and "constantInput" in node_dict:
            constant_input = json.loads(node_dict["constantInput"])
            constant_input = Graph._hide_credentials_in_input(constant_input)
            node_dict["constantInput"] = json.dumps(constant_input)
        return Node.from_db(AgentNode(**node_dict))

    @staticmethod
    def _hide_credentials_in_input(input_data: dict[str, Any]) -> dict[str, Any]:
        sensitive_keys = ["credentials", "api_key", "password", "token", "secret"]
        result = {}
        for key, value in input_data.items():
            if isinstance(value, dict):
                result[key] = Graph._hide_credentials_in_input(value)
            elif isinstance(value, str) and any(
                sensitive_key in key.lower() for sensitive_key in sensitive_keys
            ):
                # Skip this key-value pair in the result
                continue
            else:
                result[key] = value
        return result


AGENT_NODE_INCLUDE: prisma.types.AgentNodeInclude = {
    "Input": True,
    "Output": True,
    "AgentBlock": True,
}

__SUBGRAPH_INCLUDE = {"AgentNodes": {"include": AGENT_NODE_INCLUDE}}

AGENT_GRAPH_INCLUDE: prisma.types.AgentGraphInclude = {
    **__SUBGRAPH_INCLUDE,
    "AgentSubGraphs": {"include": __SUBGRAPH_INCLUDE},  # type: ignore
}


# --------------------- Model functions --------------------- #


async def get_node(node_id: str) -> Node:
    node = await AgentNode.prisma().find_unique_or_raise(
        where={"id": node_id},
        include=AGENT_NODE_INCLUDE,
    )
    return Node.from_db(node)


async def get_graphs_meta(
    user_id: str,
    include_executions: bool = False,
    filter_by: Literal["active", "template"] | None = "active",
) -> list[GraphMeta]:
    """
    Retrieves graph metadata objects.
    Default behaviour is to get all currently active graphs.

    Args:
        include_executions: Whether to include executions in the graph metadata.
        filter_by: An optional filter to either select templates or active graphs.
        user_id: The ID of the user that owns the graph.

    Returns:
        list[GraphMeta]: A list of objects representing the retrieved graph metadata.
    """
    where_clause: prisma.types.AgentGraphWhereInput = {}

    if filter_by == "active":
        where_clause["isActive"] = True
    elif filter_by == "template":
        where_clause["isTemplate"] = True

    where_clause["userId"] = user_id

    graphs = await AgentGraph.prisma().find_many(
        where=where_clause,
        distinct=["id"],
        order={"version": "desc"},
        include=(
            AgentGraphInclude(
                AgentGraphExecution={"include": {"AgentNodeExecutions": True}}
            )
            if include_executions
            else None
        ),
    )

    if not graphs:
        return []

    return [GraphMeta.from_db(graph) for graph in graphs]


async def get_graph(
    graph_id: str,
    version: int | None = None,
    template: bool = False,
    user_id: str | None = None,
    hide_credentials: bool = False,
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

    if user_id and not template:
        where_clause["userId"] = user_id

    graph = await AgentGraph.prisma().find_first(
        where=where_clause,
        include=AGENT_GRAPH_INCLUDE,
        order={"version": "desc"},
    )
    return Graph.from_db(graph, hide_credentials) if graph else None


async def set_graph_active_version(graph_id: str, version: int, user_id: str) -> None:
    # Check if the graph belongs to the user
    graph = await AgentGraph.prisma().find_first(
        where={
            "id": graph_id,
            "version": version,
            "userId": user_id,
        }
    )
    if not graph:
        raise Exception(f"Graph #{graph_id} v{version} not found or not owned by user")

    updated_graph = await AgentGraph.prisma().update(
        data={"isActive": True},
        where={
            "graphVersionId": {"id": graph_id, "version": version},
        },
    )
    if not updated_graph:
        raise Exception(f"Graph #{graph_id} v{version} not found")

    # Deactivate all other versions
    await AgentGraph.prisma().update_many(
        data={"isActive": False},
        where={"id": graph_id, "version": {"not": version}, "userId": user_id},
    )


async def get_graph_all_versions(graph_id: str, user_id: str) -> list[Graph]:
    graph_versions = await AgentGraph.prisma().find_many(
        where={"id": graph_id, "userId": user_id},
        order={"version": "desc"},
        include=AGENT_GRAPH_INCLUDE,
    )

    if not graph_versions:
        return []

    return [Graph.from_db(graph) for graph in graph_versions]


async def delete_graph(graph_id: str, user_id: str) -> int:
    entries_count = await AgentGraph.prisma().delete_many(
        where={"id": graph_id, "userId": user_id}
    )
    if entries_count:
        logger.info(f"Deleted {entries_count} graph entries for Graph #{graph_id}")
    return entries_count


async def create_graph(graph: Graph, user_id: str) -> Graph:
    async with transaction() as tx:
        await __create_graph(tx, graph, user_id)

    if created_graph := await get_graph(
        graph.id, graph.version, graph.is_template, user_id=user_id
    ):
        return created_graph

    raise ValueError(f"Created graph {graph.id} v{graph.version} is not in DB")


async def __create_graph(tx, graph: Graph, user_id: str):
    await AgentGraph.prisma(tx).create(
        data={
            "id": graph.id,
            "version": graph.version,
            "name": graph.name,
            "description": graph.description,
            "isTemplate": graph.is_template,
            "isActive": graph.is_active,
            "userId": user_id,
        }
    )

    await asyncio.gather(
        *[
            AgentGraph.prisma(tx).create(
                data={
                    "id": subgraph_id,
                    "agentGraphParentId": graph.id,
                    "version": graph.version,
                    "name": f"SubGraph of {graph.name}",
                    "description": f"Sub-Graph of {graph.id}",
                    "isTemplate": graph.is_template,
                    "isActive": graph.is_active,
                    "userId": user_id,
                }
            )
            for subgraph_id in graph.subgraphs
        ]
    )

    subgraph_map = graph.subgraph_map

    await asyncio.gather(
        *[
            AgentNode.prisma(tx).create(
                {
                    "id": node.id,
                    "agentBlockId": node.block_id,
                    "agentGraphId": subgraph_map.get(node.id, graph.id),
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
            AgentNodeLink.prisma(tx).create(
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
