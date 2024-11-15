import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal, Type

import prisma
from prisma.models import AgentGraph, AgentGraphExecution, AgentNode, AgentNodeLink
from prisma.types import AgentGraphWhereInput
from pydantic.fields import computed_field

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import AgentInputBlock, AgentOutputBlock
from backend.data.block import BlockInput, BlockType, get_block, get_blocks
from backend.data.db import BaseDbModel, transaction
from backend.data.execution import ExecutionStatus
from backend.data.includes import AGENT_GRAPH_INCLUDE, AGENT_NODE_INCLUDE
from backend.util import json

logger = logging.getLogger(__name__)


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
    input_links: list[Link] = []
    output_links: list[Link] = []

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
        obj.input_links = [Link.from_db(link) for link in node.Input or []]
        obj.output_links = [Link.from_db(link) for link in node.Output or []]
        return obj


class GraphExecution(BaseDbModel):
    execution_id: str
    started_at: datetime
    ended_at: datetime
    duration: float
    total_run_time: float
    status: ExecutionStatus

    @staticmethod
    def from_db(execution: AgentGraphExecution):
        now = datetime.now(timezone.utc)
        start_time = execution.startedAt or execution.createdAt
        end_time = execution.updatedAt or now
        duration = (end_time - start_time).total_seconds()
        total_run_time = duration

        if execution.stats:
            stats = json.loads(execution.stats)
            duration = stats.get("walltime", duration)
            total_run_time = stats.get("nodes_walltime", total_run_time)

        return GraphExecution(
            id=execution.id,
            execution_id=execution.id,
            started_at=start_time,
            ended_at=end_time,
            duration=duration,
            total_run_time=total_run_time,
            status=ExecutionStatus(execution.executionStatus),
        )


class Graph(BaseDbModel):
    version: int = 1
    is_active: bool = True
    is_template: bool = False
    name: str
    description: str
    executions: list[GraphExecution] = []
    nodes: list[Node] = []
    links: list[Link] = []

    @staticmethod
    def _generate_schema(
        type_class: Type[AgentInputBlock.Input] | Type[AgentOutputBlock.Input],
        data: list[dict],
    ) -> dict[str, Any]:
        props = []
        for p in data:
            try:
                props.append(type_class(**p))
            except Exception as e:
                logger.warning(f"Invalid {type_class}: {p}, {e}")

        return {
            "type": "object",
            "properties": {
                p.name: {
                    "secret": p.secret,
                    "advanced": p.advanced,
                    "title": p.title or p.name,
                    **({"description": p.description} if p.description else {}),
                    **({"default": p.value} if p.value is not None else {}),
                }
                for p in props
            },
            "required": [p.name for p in props if p.value is None],
        }

    @computed_field
    @property
    def input_schema(self) -> dict[str, Any]:
        return self._generate_schema(
            AgentInputBlock.Input,
            [
                node.input_default
                for node in self.nodes
                if (b := get_block(node.block_id))
                and b.block_type == BlockType.INPUT
                and "name" in node.input_default
            ],
        )

    @computed_field
    @property
    def output_schema(self) -> dict[str, Any]:
        return self._generate_schema(
            AgentOutputBlock.Input,
            [
                node.input_default
                for node in self.nodes
                if (b := get_block(node.block_id))
                and b.block_type == BlockType.OUTPUT
                and "name" in node.input_default
            ],
        )

    @property
    def starting_nodes(self) -> list[Node]:
        outbound_nodes = {link.sink_id for link in self.links}
        input_nodes = {
            v.id
            for v in self.nodes
            if (b := get_block(v.block_id)) and b.block_type == BlockType.INPUT
        }
        return [
            node
            for node in self.nodes
            if node.id not in outbound_nodes or node.id in input_nodes
        ]

    def reassign_ids(self, user_id: str, reassign_graph_id: bool = False):
        """
        Reassigns all IDs in the graph to new UUIDs.
        This method can be used before storing a new graph to the database.
        """

        # Reassign Graph ID
        id_map = {node.id: str(uuid.uuid4()) for node in self.nodes}
        if reassign_graph_id:
            self.id = str(uuid.uuid4())

        # Reassign Node IDs
        for node in self.nodes:
            node.id = id_map[node.id]

        # Reassign Link IDs
        for link in self.links:
            link.source_id = id_map[link.source_id]
            link.sink_id = id_map[link.sink_id]

        # Reassign User IDs for agent blocks
        for node in self.nodes:
            if node.block_id != AgentExecutorBlock().id:
                continue
            node.input_default["user_id"] = user_id
            node.input_default.setdefault("data", {})

        self.validate_graph()

    def validate_graph(self, for_run: bool = False):
        def sanitize(name):
            return name.split("_#_")[0].split("_@_")[0].split("_$_")[0]

        input_links = defaultdict(list)
        for link in self.links:
            input_links[link.sink_id].append(link)

        # Nodes: required fields are filled or connected
        for node in self.nodes:
            block = get_block(node.block_id)
            if block is None:
                raise ValueError(f"Invalid block {node.block_id} for node #{node.id}")

            provided_inputs = set(
                [sanitize(name) for name in node.input_default]
                + [sanitize(link.sink_name) for link in input_links.get(node.id, [])]
            )
            for name in block.input_schema.get_required_fields():
                if name not in provided_inputs and (
                    for_run  # Skip input completion validation, unless when executing.
                    or block.block_type == BlockType.INPUT
                    or block.block_type == BlockType.OUTPUT
                    or block.block_type == BlockType.AGENT
                ):
                    raise ValueError(
                        f"Node {block.name} #{node.id} required input missing: `{name}`"
                    )

        node_map = {v.id: v for v in self.nodes}

        def is_static_output_block(nid: str) -> bool:
            bid = node_map[nid].block_id
            b = get_block(bid)
            return b.static_output if b else False

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
                    blocks = {v().id: v().name for v in get_blocks().values()}
                    raise ValueError(
                        f"{suffix}, {node.block_id} is invalid block id, available blocks: {blocks}"
                    )

                sanitized_name = sanitize(name)
                vals = node.input_default
                if i == 0:
                    fields = (
                        block.output_schema.get_fields()
                        if block.block_type != BlockType.AGENT
                        else vals.get("output_schema", {}).get("properties", {}).keys()
                    )
                else:
                    fields = (
                        block.input_schema.get_fields()
                        if block.block_type != BlockType.AGENT
                        else vals.get("input_schema", {}).get("properties", {}).keys()
                    )
                if sanitized_name not in fields:
                    fields_msg = f"Allowed fields: {fields}"
                    raise ValueError(f"{suffix}, `{name}` invalid, {fields_msg}")

            if is_static_output_block(link.source_id):
                link.is_static = True  # Each value block output should be static.

    @staticmethod
    def from_db(graph: AgentGraph, hide_credentials: bool = False):
        executions = [
            GraphExecution.from_db(execution)
            for execution in graph.AgentGraphExecution or []
        ]
        nodes = graph.AgentNodes or []

        return Graph(
            id=graph.id,
            version=graph.version,
            is_active=graph.isActive,
            is_template=graph.isTemplate,
            name=graph.name or "",
            description=graph.description or "",
            executions=executions,
            nodes=[Graph._process_node(node, hide_credentials) for node in nodes],
            links=list(
                {
                    Link.from_db(link)
                    for node in nodes
                    for link in (node.Input or []) + (node.Output or [])
                }
            ),
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


# --------------------- Model functions --------------------- #


async def get_node(node_id: str) -> Node:
    node = await AgentNode.prisma().find_unique_or_raise(
        where={"id": node_id},
        include=AGENT_NODE_INCLUDE,
    )
    return Node.from_db(node)


async def get_graphs(
    user_id: str,
    include_executions: bool = False,
    filter_by: Literal["active", "template"] | None = "active",
) -> list[Graph]:
    """
    Retrieves graph metadata objects.
    Default behaviour is to get all currently active graphs.

    Args:
        include_executions: Whether to include executions in the graph metadata.
        filter_by: An optional filter to either select templates or active graphs.
        user_id: The ID of the user that owns the graph.

    Returns:
        list[Graph]: A list of objects representing the retrieved graph metadata.
    """
    where_clause: AgentGraphWhereInput = {}

    if filter_by == "active":
        where_clause["isActive"] = True
    elif filter_by == "template":
        where_clause["isTemplate"] = True

    where_clause["userId"] = user_id

    graph_include = AGENT_GRAPH_INCLUDE
    graph_include["AgentGraphExecution"] = include_executions

    graphs = await AgentGraph.prisma().find_many(
        where=where_clause,
        distinct=["id"],
        order={"version": "desc"},
        include=graph_include,
    )

    return [Graph.from_db(graph) for graph in graphs]


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
    where_clause: AgentGraphWhereInput = {
        "id": graph_id,
        "isTemplate": template,
    }
    if version is not None:
        where_clause["version"] = version
    elif not template:
        where_clause["isActive"] = True

    if user_id is not None and not template:
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
            AgentNode.prisma(tx).create(
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


# ------------------------ UTILITIES ------------------------ #


async def fix_llm_provider_credentials():
    """Fix node credentials with provider `llm`"""
    from autogpt_libs.supabase_integration_credentials_store import (
        SupabaseIntegrationCredentialsStore,
    )

    from .redis import get_redis
    from .user import get_user_integrations

    store = SupabaseIntegrationCredentialsStore(get_redis())

    broken_nodes = await prisma.get_client().query_raw(
        """
        SELECT    "User".id            user_id,
                  node.id              node_id,
                  node."constantInput" node_preset_input
        FROM      platform."AgentNode"  node
        LEFT JOIN platform."AgentGraph" graph
        ON        node."agentGraphId" = graph.id
        LEFT JOIN platform."User"       "User"
        ON        graph."userId" = "User".id
        WHERE     node."constantInput"::jsonb->'credentials'->>'provider' = 'llm'
        ORDER BY  user_id;
        """
    )
    logger.info(f"Fixing LLM credential inputs on {len(broken_nodes)} nodes")

    user_id: str = ""
    user_integrations = None
    for node in broken_nodes:
        if node["user_id"] != user_id:
            # Save queries by only fetching once per user
            user_id = node["user_id"]
            user_integrations = await get_user_integrations(user_id)
        elif not user_integrations:
            raise RuntimeError(f"Impossible state while processing node {node}")

        node_id: str = node["node_id"]
        node_preset_input: dict = json.loads(node["node_preset_input"])
        credentials_meta: dict = node_preset_input["credentials"]

        credentials = next(
            (
                c
                for c in user_integrations.credentials
                if c.id == credentials_meta["id"]
            ),
            None,
        )
        if not credentials:
            continue
        if credentials.type != "api_key":
            logger.warning(
                f"User {user_id} credentials {credentials.id} with provider 'llm' "
                f"has invalid type '{credentials.type}'"
            )
            continue

        api_key = credentials.api_key.get_secret_value()
        if api_key.startswith("sk-ant-api03-"):
            credentials.provider = credentials_meta["provider"] = "anthropic"
        elif api_key.startswith("sk-"):
            credentials.provider = credentials_meta["provider"] = "openai"
        elif api_key.startswith("gsk_"):
            credentials.provider = credentials_meta["provider"] = "groq"
        else:
            logger.warning(
                f"Could not identify provider from key prefix {api_key[:13]}*****"
            )
            continue

        store.update_creds(user_id, credentials)
        await AgentNode.prisma().update(
            where={"id": node_id},
            data={"constantInput": json.dumps(node_preset_input)},
        )
