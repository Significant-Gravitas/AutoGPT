import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Type

import prisma
from prisma import Json
from prisma.models import (
    AgentGraph,
    AgentGraphExecution,
    AgentNode,
    AgentNodeLink,
    StoreListingVersion,
)
from prisma.types import AgentGraphExecutionWhereInput, AgentGraphWhereInput
from pydantic.fields import Field, computed_field

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import AgentInputBlock, AgentOutputBlock
from backend.util import type as type_utils

from .block import Block, BlockInput, BlockSchema, BlockType, get_block, get_blocks
from .db import BaseDbModel, transaction
from .execution import ExecutionResult, ExecutionStatus
from .includes import AGENT_GRAPH_INCLUDE, AGENT_NODE_INCLUDE
from .integrations import Webhook

logger = logging.getLogger(__name__)

_INPUT_BLOCK_ID = AgentInputBlock().id
_OUTPUT_BLOCK_ID = AgentOutputBlock().id


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

    webhook_id: Optional[str] = None


class NodeModel(Node):
    graph_id: str
    graph_version: int

    webhook: Optional[Webhook] = None

    @property
    def block(self) -> Block[BlockSchema, BlockSchema]:
        block = get_block(self.block_id)
        if not block:
            raise ValueError(f"Block #{self.block_id} does not exist")
        return block

    @staticmethod
    def from_db(node: AgentNode, for_export: bool = False) -> "NodeModel":
        obj = NodeModel(
            id=node.id,
            block_id=node.agentBlockId,
            input_default=type_utils.convert(node.constantInput, dict[str, Any]),
            metadata=type_utils.convert(node.metadata, dict[str, Any]),
            graph_id=node.agentGraphId,
            graph_version=node.agentGraphVersion,
            webhook_id=node.webhookId,
            webhook=Webhook.from_db(node.Webhook) if node.Webhook else None,
        )
        obj.input_links = [Link.from_db(link) for link in node.Input or []]
        obj.output_links = [Link.from_db(link) for link in node.Output or []]
        if for_export:
            return obj.stripped_for_export()
        return obj

    def is_triggered_by_event_type(self, event_type: str) -> bool:
        if not (block := get_block(self.block_id)):
            raise ValueError(f"Block #{self.block_id} not found for node #{self.id}")
        if not block.webhook_config:
            raise TypeError("This method can't be used on non-webhook blocks")
        if not block.webhook_config.event_filter_input:
            return True
        event_filter = self.input_default.get(block.webhook_config.event_filter_input)
        if not event_filter:
            raise ValueError(f"Event filter is not configured on node #{self.id}")
        return event_type in [
            block.webhook_config.event_format.format(event=k)
            for k in event_filter
            if event_filter[k] is True
        ]

    def stripped_for_export(self) -> "NodeModel":
        """
        Returns a copy of the node model, stripped of any non-transferable properties
        """
        stripped_node = self.model_copy(deep=True)
        # Remove credentials from node input
        if stripped_node.input_default:
            stripped_node.input_default = NodeModel._filter_secrets_from_node_input(
                stripped_node.input_default, self.block.input_schema.jsonschema()
            )

        if (
            stripped_node.block.block_type == BlockType.INPUT
            and "value" in stripped_node.input_default
        ):
            stripped_node.input_default["value"] = ""

        # Remove webhook info
        stripped_node.webhook_id = None
        stripped_node.webhook = None

        return stripped_node

    @staticmethod
    def _filter_secrets_from_node_input(
        input_data: dict[str, Any], schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        sensitive_keys = ["credentials", "api_key", "password", "token", "secret"]
        field_schemas = schema.get("properties", {}) if schema else {}
        result = {}
        for key, value in input_data.items():
            field_schema: dict | None = field_schemas.get(key)
            if (field_schema and field_schema.get("secret", False)) or any(
                sensitive_key in key.lower() for sensitive_key in sensitive_keys
            ):
                # This is a secret value -> filter this key-value pair out
                continue
            elif isinstance(value, dict):
                result[key] = NodeModel._filter_secrets_from_node_input(
                    value, field_schema
                )
            else:
                result[key] = value
        return result


# Fix 2-way reference Node <-> Webhook
Webhook.model_rebuild()


class GraphExecutionMeta(BaseDbModel):
    execution_id: str
    started_at: datetime
    ended_at: datetime
    cost: Optional[int] = Field(..., description="Execution cost in credits")
    duration: float
    total_run_time: float
    status: ExecutionStatus
    graph_id: str
    graph_version: int
    preset_id: Optional[str]

    @staticmethod
    def from_db(_graph_exec: AgentGraphExecution):
        now = datetime.now(timezone.utc)
        start_time = _graph_exec.startedAt or _graph_exec.createdAt
        end_time = _graph_exec.updatedAt or now
        duration = (end_time - start_time).total_seconds()
        total_run_time = duration

        try:
            stats = type_utils.convert(_graph_exec.stats or {}, dict[str, Any])
        except ValueError:
            stats = {}

        duration = stats.get("walltime", duration)
        total_run_time = stats.get("nodes_walltime", total_run_time)

        return GraphExecutionMeta(
            id=_graph_exec.id,
            execution_id=_graph_exec.id,
            started_at=start_time,
            ended_at=end_time,
            cost=stats.get("cost", None),
            duration=duration,
            total_run_time=total_run_time,
            status=ExecutionStatus(_graph_exec.executionStatus),
            graph_id=_graph_exec.agentGraphId,
            graph_version=_graph_exec.agentGraphVersion,
            preset_id=_graph_exec.agentPresetId,
        )


class GraphExecution(GraphExecutionMeta):
    inputs: dict[str, Any]
    outputs: dict[str, list[Any]]
    node_executions: list[ExecutionResult]

    @staticmethod
    def from_db(_graph_exec: AgentGraphExecution):
        if _graph_exec.AgentNodeExecutions is None:
            raise ValueError("Node executions must be included in query")

        graph_exec = GraphExecutionMeta.from_db(_graph_exec)

        node_executions = [
            ExecutionResult.from_db(ne, _graph_exec.userId)
            for ne in _graph_exec.AgentNodeExecutions
        ]

        inputs = {
            **{
                # inputs from Agent Input Blocks
                exec.input_data["name"]: exec.input_data.get("value")
                for exec in node_executions
                if exec.block_id == _INPUT_BLOCK_ID
            },
            **{
                # input from webhook-triggered block
                "payload": exec.input_data["payload"]
                for exec in node_executions
                if (block := get_block(exec.block_id))
                and block.block_type in [BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL]
            },
        }

        outputs: dict[str, list] = defaultdict(list)
        for exec in node_executions:
            if exec.block_id == _OUTPUT_BLOCK_ID:
                outputs[exec.input_data["name"]].append(
                    exec.input_data.get("value", None)
                )

        return GraphExecution(
            **{
                field_name: getattr(graph_exec, field_name)
                for field_name in graph_exec.model_fields
            },
            inputs=inputs,
            outputs=outputs,
            node_executions=node_executions,
        )


class BaseGraph(BaseDbModel):
    version: int = 1
    is_active: bool = True
    name: str
    description: str
    nodes: list[Node] = []
    links: list[Link] = []

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
                    # Default value has to be set for advanced fields.
                    "advanced": p.advanced and p.value is not None,
                    "title": p.title or p.name,
                    **({"description": p.description} if p.description else {}),
                    **({"default": p.value} if p.value is not None else {}),
                }
                for p in props
            },
            "required": [p.name for p in props if p.value is None],
        }


class Graph(BaseGraph):
    sub_graphs: list[BaseGraph] = []  # Flattened sub-graphs, only used in export


class GraphModel(Graph):
    user_id: str
    nodes: list[NodeModel] = []  # type: ignore

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
        if reassign_graph_id:
            graph_id_map = {
                self.id: str(uuid.uuid4()),
                **{sub_graph.id: str(uuid.uuid4()) for sub_graph in self.sub_graphs},
            }
        else:
            graph_id_map = {}

        self._reassign_ids(self, user_id, graph_id_map)
        for sub_graph in self.sub_graphs:
            self._reassign_ids(sub_graph, user_id, graph_id_map)

    @staticmethod
    def _reassign_ids(
        graph: BaseGraph,
        user_id: str,
        graph_id_map: dict[str, str],
    ):
        # Reassign Graph ID
        if graph.id in graph_id_map:
            graph.id = graph_id_map[graph.id]

        # Reassign Node IDs
        id_map = {node.id: str(uuid.uuid4()) for node in graph.nodes}
        for node in graph.nodes:
            node.id = id_map[node.id]

        # Reassign Link IDs
        for link in graph.links:
            link.source_id = id_map[link.source_id]
            link.sink_id = id_map[link.sink_id]

        # Reassign User IDs for agent blocks
        for node in graph.nodes:
            if node.block_id != AgentExecutorBlock().id:
                continue
            node.input_default["user_id"] = user_id
            node.input_default.setdefault("data", {})
            if (graph_id := node.input_default.get("graph_id")) in graph_id_map:
                node.input_default["graph_id"] = graph_id_map[graph_id]

    def validate_graph(self, for_run: bool = False):
        self._validate_graph(self, for_run)
        for sub_graph in self.sub_graphs:
            self._validate_graph(sub_graph, for_run)

    @staticmethod
    def _validate_graph(graph: BaseGraph, for_run: bool = False):
        def sanitize(name):
            sanitized_name = name.split("_#_")[0].split("_@_")[0].split("_$_")[0]
            if sanitized_name.startswith("tools_^_"):
                return sanitized_name.split("_^_")[0]
            return sanitized_name

        # Validate smart decision maker nodes
        smart_decision_maker_nodes = set()
        agent_nodes = set()
        nodes_block = {
            node.id: block
            for node in graph.nodes
            if (block := get_block(node.block_id)) is not None
        }

        for node in graph.nodes:
            if (block := nodes_block.get(node.id)) is None:
                raise ValueError(f"Invalid block {node.block_id} for node #{node.id}")

            # Smart decision maker nodes
            if block.block_type == BlockType.AI:
                smart_decision_maker_nodes.add(node.id)
            # Agent nodes
            elif block.block_type == BlockType.AGENT:
                agent_nodes.add(node.id)

        input_links = defaultdict(list)

        for link in graph.links:
            input_links[link.sink_id].append(link)

        # Nodes: required fields are filled or connected and dependencies are satisfied
        for node in graph.nodes:
            if (block := nodes_block.get(node.id)) is None:
                raise ValueError(f"Invalid block {node.block_id} for node #{node.id}")

            provided_inputs = set(
                [sanitize(name) for name in node.input_default]
                + [sanitize(link.sink_name) for link in input_links.get(node.id, [])]
            )
            for name in block.input_schema.get_required_fields():
                if (
                    name not in provided_inputs
                    and not (
                        name == "payload"
                        and block.block_type
                        in (BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)
                    )
                    and (
                        for_run  # Skip input completion validation, unless when executing.
                        or block.block_type
                        in [
                            BlockType.INPUT,
                            BlockType.OUTPUT,
                            BlockType.AGENT,
                        ]
                    )
                ):
                    raise ValueError(
                        f"Node {block.name} #{node.id} required input missing: `{name}`"
                    )

            # Get input schema properties and check dependencies
            input_schema = block.input_schema.model_fields
            required_fields = block.input_schema.get_required_fields()

            def has_value(name):
                return (
                    node is not None
                    and name in node.input_default
                    and node.input_default[name] is not None
                    and str(node.input_default[name]).strip() != ""
                ) or (name in input_schema and input_schema[name].default is not None)

            # Validate dependencies between fields
            for field_name, field_info in input_schema.items():
                # Apply input dependency validation only on run & field with depends_on
                json_schema_extra = field_info.json_schema_extra or {}
                dependencies = json_schema_extra.get("depends_on", [])
                if not for_run or not dependencies:
                    continue

                # Check if dependent field has value in input_default
                field_has_value = has_value(field_name)
                field_is_required = field_name in required_fields

                # Check for missing dependencies when dependent field is present
                missing_deps = [dep for dep in dependencies if not has_value(dep)]
                if missing_deps and (field_has_value or field_is_required):
                    raise ValueError(
                        f"Node {block.name} #{node.id}: Field `{field_name}` requires [{', '.join(missing_deps)}] to be set"
                    )

        node_map = {v.id: v for v in graph.nodes}

        def is_static_output_block(nid: str) -> bool:
            bid = node_map[nid].block_id
            b = get_block(bid)
            return b.static_output if b else False

        # Links: links are connected and the connected pin data type are compatible.
        for link in graph.links:
            source = (link.source_id, link.source_name)
            sink = (link.sink_id, link.sink_name)
            prefix = f"Link {source} <-> {sink}"

            for i, (node_id, name) in enumerate([source, sink]):
                node = node_map.get(node_id)
                if not node:
                    raise ValueError(
                        f"{prefix}, {node_id} is invalid node id, available nodes: {node_map.keys()}"
                    )

                block = get_block(node.block_id)
                if not block:
                    blocks = {v().id: v().name for v in get_blocks().values()}
                    raise ValueError(
                        f"{prefix}, {node.block_id} is invalid block id, available blocks: {blocks}"
                    )

                sanitized_name = sanitize(name)
                vals = node.input_default
                if i == 0:
                    fields = (
                        block.output_schema.get_fields()
                        if block.block_type not in [BlockType.AGENT]
                        else vals.get("output_schema", {}).get("properties", {}).keys()
                    )
                else:
                    fields = (
                        block.input_schema.get_fields()
                        if block.block_type not in [BlockType.AGENT]
                        else vals.get("input_schema", {}).get("properties", {}).keys()
                    )
                if sanitized_name not in fields and not name.startswith("tools_^_"):
                    fields_msg = f"Allowed fields: {fields}"
                    raise ValueError(f"{prefix}, `{name}` invalid, {fields_msg}")

            if is_static_output_block(link.source_id):
                link.is_static = True  # Each value block output should be static.

    @staticmethod
    def from_db(
        graph: AgentGraph,
        for_export: bool = False,
        sub_graphs: list[AgentGraph] | None = None,
    ):
        return GraphModel(
            id=graph.id,
            user_id=graph.userId if not for_export else "",
            version=graph.version,
            is_active=graph.isActive,
            name=graph.name or "",
            description=graph.description or "",
            nodes=[
                NodeModel.from_db(node, for_export) for node in graph.AgentNodes or []
            ],
            links=list(
                {
                    Link.from_db(link)
                    for node in graph.AgentNodes or []
                    for link in (node.Input or []) + (node.Output or [])
                }
            ),
            sub_graphs=[
                GraphModel.from_db(sub_graph, for_export)
                for sub_graph in sub_graphs or []
            ],
        )


# --------------------- CRUD functions --------------------- #


async def get_node(node_id: str) -> NodeModel:
    node = await AgentNode.prisma().find_unique_or_raise(
        where={"id": node_id},
        include=AGENT_NODE_INCLUDE,
    )
    return NodeModel.from_db(node)


async def set_node_webhook(node_id: str, webhook_id: str | None) -> NodeModel:
    node = await AgentNode.prisma().update(
        where={"id": node_id},
        data=(
            {"Webhook": {"connect": {"id": webhook_id}}}
            if webhook_id
            else {"Webhook": {"disconnect": True}}
        ),
        include=AGENT_NODE_INCLUDE,
    )
    if not node:
        raise ValueError(f"Node #{node_id} not found")
    return NodeModel.from_db(node)


async def get_graphs(
    user_id: str,
    filter_by: Literal["active"] | None = "active",
) -> list[GraphModel]:
    """
    Retrieves graph metadata objects.
    Default behaviour is to get all currently active graphs.

    Args:
        filter_by: An optional filter to either select graphs.
        user_id: The ID of the user that owns the graph.

    Returns:
        list[GraphModel]: A list of objects representing the retrieved graphs.
    """
    where_clause: AgentGraphWhereInput = {"userId": user_id}

    if filter_by == "active":
        where_clause["isActive"] = True

    graphs = await AgentGraph.prisma().find_many(
        where=where_clause,
        distinct=["id"],
        order={"version": "desc"},
        include=AGENT_GRAPH_INCLUDE,
    )

    graph_models = []
    for graph in graphs:
        try:
            graph_models.append(GraphModel.from_db(graph))
        except Exception as e:
            logger.error(f"Error processing graph {graph.id}: {e}")
            continue

    return graph_models


async def get_graph_executions(
    graph_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[GraphExecutionMeta]:
    where_filter: AgentGraphExecutionWhereInput = {
        "isDeleted": False,
    }
    if user_id:
        where_filter["userId"] = user_id
    if graph_id:
        where_filter["agentGraphId"] = graph_id

    executions = await AgentGraphExecution.prisma().find_many(
        where=where_filter,
        order={"createdAt": "desc"},
    )
    return [GraphExecutionMeta.from_db(execution) for execution in executions]


async def get_execution_meta(
    user_id: str, execution_id: str
) -> GraphExecutionMeta | None:
    execution = await AgentGraphExecution.prisma().find_first(
        where={"id": execution_id, "isDeleted": False, "userId": user_id}
    )
    return GraphExecutionMeta.from_db(execution) if execution else None


async def get_execution(user_id: str, execution_id: str) -> GraphExecution | None:
    execution = await AgentGraphExecution.prisma().find_first(
        where={"id": execution_id, "isDeleted": False, "userId": user_id},
        include={
            "AgentNodeExecutions": {
                "include": {"AgentNode": True, "Input": True, "Output": True},
                "order_by": [
                    {"queuedTime": "asc"},
                    {  # Fallback: Incomplete execs has no queuedTime.
                        "addedTime": "asc"
                    },
                ],
            },
        },
    )
    return GraphExecution.from_db(execution) if execution else None


async def get_graph_metadata(graph_id: str, version: int | None = None) -> Graph | None:
    where_clause: AgentGraphWhereInput = {
        "id": graph_id,
    }

    if version is not None:
        where_clause["version"] = version

    graph = await AgentGraph.prisma().find_first(
        where=where_clause,
        include=AGENT_GRAPH_INCLUDE,
        order={"version": "desc"},
    )

    if not graph:
        return None

    return Graph(
        id=graph.id,
        name=graph.name or "",
        description=graph.description or "",
        version=graph.version,
        is_active=graph.isActive,
    )


async def get_graph(
    graph_id: str,
    version: int | None = None,
    user_id: str | None = None,
    for_export: bool = False,
) -> GraphModel | None:
    """
    Retrieves a graph from the DB.
    Defaults to the version with `is_active` if `version` is not passed.

    Returns `None` if the record is not found.
    """
    where_clause: AgentGraphWhereInput = {
        "id": graph_id,
    }

    if version is not None:
        where_clause["version"] = version

    graph = await AgentGraph.prisma().find_first(
        where=where_clause,
        include=AGENT_GRAPH_INCLUDE,
        order={"version": "desc"},
    )

    # For access, the graph must be owned by the user or listed in the store
    if graph is None or (
        graph.userId != user_id
        and not (
            await StoreListingVersion.prisma().find_first(
                where={
                    "agentId": graph_id,
                    "agentVersion": version or graph.version,
                    "isDeleted": False,
                    "StoreListing": {"is": {"isApproved": True}},
                }
            )
        )
    ):
        return None

    if for_export:
        sub_graphs = await _get_sub_graphs(graph)
        return GraphModel.from_db(
            graph=graph,
            sub_graphs=sub_graphs,
            for_export=for_export,
        )

    return GraphModel.from_db(graph, for_export)


async def _get_sub_graphs(graph: AgentGraph) -> list[AgentGraph]:
    """
    Iteratively fetches all sub-graphs of a given graph, and flattens them into a list.
    This call involves a DB fetch in batch, breadth-first, per-level of graph depth.
    On each DB fetch we will only fetch the sub-graphs that are not already in the list.
    """
    sub_graphs = {graph.id: graph}
    search_graphs = [graph]
    agent_block_id = AgentExecutorBlock().id

    while search_graphs:
        sub_graph_ids = [
            (graph_id, graph_version)
            for graph in search_graphs
            for node in graph.AgentNodes or []
            if (
                node.AgentBlock
                and node.AgentBlock.id == agent_block_id
                and (graph_id := dict(node.constantInput).get("graph_id"))
                and (graph_version := dict(node.constantInput).get("graph_version"))
            )
        ]
        if not sub_graph_ids:
            break

        graphs = await AgentGraph.prisma().find_many(
            where={
                "OR": [
                    {
                        "id": graph_id,
                        "version": graph_version,
                        "userId": graph.userId,  # Ensure the sub-graph is owned by the same user
                    }
                    for graph_id, graph_version in sub_graph_ids
                ]  # type: ignore
            },
            include=AGENT_GRAPH_INCLUDE,
        )

        search_graphs = [graph for graph in graphs if graph.id not in sub_graphs]
        sub_graphs.update({graph.id: graph for graph in search_graphs})

    return [g for g in sub_graphs.values() if g.id != graph.id]


async def get_connected_output_nodes(node_id: str) -> list[tuple[Link, Node]]:
    links = await AgentNodeLink.prisma().find_many(
        where={"agentNodeSourceId": node_id},
        include={"AgentNodeSink": {"include": AGENT_NODE_INCLUDE}},  # type: ignore
    )
    return [
        (Link.from_db(link), NodeModel.from_db(link.AgentNodeSink))
        for link in links
        if link.AgentNodeSink
    ]


async def set_graph_active_version(graph_id: str, version: int, user_id: str) -> None:
    # Activate the requested version if it exists and is owned by the user.
    updated_count = await AgentGraph.prisma().update_many(
        data={"isActive": True},
        where={
            "id": graph_id,
            "version": version,
            "userId": user_id,
        },
    )
    if updated_count == 0:
        raise Exception(f"Graph #{graph_id} v{version} not found or not owned by user")

    # Deactivate all other versions.
    await AgentGraph.prisma().update_many(
        data={"isActive": False},
        where={
            "id": graph_id,
            "version": {"not": version},
            "userId": user_id,
            "isActive": True,
        },
    )


async def get_graph_all_versions(graph_id: str, user_id: str) -> list[GraphModel]:
    graph_versions = await AgentGraph.prisma().find_many(
        where={"id": graph_id, "userId": user_id},
        order={"version": "desc"},
        include=AGENT_GRAPH_INCLUDE,
    )

    if not graph_versions:
        return []

    return [GraphModel.from_db(graph) for graph in graph_versions]


async def delete_graph(graph_id: str, user_id: str) -> int:
    entries_count = await AgentGraph.prisma().delete_many(
        where={"id": graph_id, "userId": user_id}
    )
    if entries_count:
        logger.info(f"Deleted {entries_count} graph entries for Graph #{graph_id}")
    return entries_count


async def create_graph(graph: Graph, user_id: str) -> GraphModel:
    async with transaction() as tx:
        await __create_graph(tx, graph, user_id)

    if created_graph := await get_graph(graph.id, graph.version, user_id=user_id):
        return created_graph

    raise ValueError(f"Created graph {graph.id} v{graph.version} is not in DB")


async def __create_graph(tx, graph: Graph, user_id: str):
    graphs = [graph] + graph.sub_graphs

    await AgentGraph.prisma(tx).create_many(
        data=[
            {
                "id": graph.id,
                "version": graph.version,
                "name": graph.name,
                "description": graph.description,
                "isActive": graph.is_active,
                "userId": user_id,
            }
            for graph in graphs
        ]
    )

    await AgentNode.prisma(tx).create_many(
        data=[
            {
                "id": node.id,
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
                "agentBlockId": node.block_id,
                "constantInput": Json(node.input_default),
                "metadata": Json(node.metadata),
                "webhookId": node.webhook_id,
            }
            for graph in graphs
            for node in graph.nodes
        ]
    )

    await AgentNodeLink.prisma(tx).create_many(
        data=[
            {
                "id": str(uuid.uuid4()),
                "sourceName": link.source_name,
                "sinkName": link.sink_name,
                "agentNodeSourceId": link.source_id,
                "agentNodeSinkId": link.sink_id,
                "isStatic": link.is_static,
            }
            for graph in graphs
            for link in graph.links
        ]
    )


# ------------------------ UTILITIES ------------------------ #


def make_graph_model(creatable_graph: Graph, user_id: str) -> GraphModel:
    """
    Convert a Graph to a GraphModel, setting graph_id and graph_version on all nodes.

    Args:
        creatable_graph (Graph): The creatable graph to convert.
        user_id (str): The ID of the user creating the graph.

    Returns:
        GraphModel: The converted Graph object.
    """
    # Create a new Graph object, inheriting properties from CreatableGraph
    return GraphModel(
        **creatable_graph.model_dump(exclude={"nodes"}),
        user_id=user_id,
        nodes=[
            NodeModel(
                **creatable_node.model_dump(),
                graph_id=creatable_graph.id,
                graph_version=creatable_graph.version,
            )
            for creatable_node in creatable_graph.nodes
        ],
    )


async def fix_llm_provider_credentials():
    """Fix node credentials with provider `llm`"""
    from backend.integrations.credentials_store import IntegrationCredentialsStore

    from .user import get_user_integrations

    store = IntegrationCredentialsStore()

    broken_nodes = []
    try:
        broken_nodes = await prisma.get_client().query_raw(
            """
            SELECT    graph."userId"       user_id,
                  node.id              node_id,
                  node."constantInput" node_preset_input
        FROM      platform."AgentNode"  node
        LEFT JOIN platform."AgentGraph" graph
        ON        node."agentGraphId" = graph.id
        WHERE     node."constantInput"::jsonb->'credentials'->>'provider' = 'llm'
        ORDER BY  graph."userId";
        """
        )
        logger.info(f"Fixing LLM credential inputs on {len(broken_nodes)} nodes")
    except Exception as e:
        logger.error(f"Error fixing LLM credential inputs: {e}")

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
        node_preset_input: dict = node["node_preset_input"]
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
            data={"constantInput": Json(node_preset_input)},
        )
