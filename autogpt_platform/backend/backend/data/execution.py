import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from multiprocessing import Manager
from queue import Empty
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    Literal,
    Mapping,
    Optional,
    TypeVar,
    cast,
    overload,
)

from prisma import Json
from prisma.enums import AgentExecutionStatus
from prisma.models import (
    AgentGraphExecution,
    AgentNodeExecution,
    AgentNodeExecutionInputOutput,
    AgentNodeExecutionKeyValueData,
)
from prisma.types import (
    AgentGraphExecutionUpdateManyMutationInput,
    AgentGraphExecutionWhereInput,
    AgentNodeExecutionCreateInput,
    AgentNodeExecutionInputOutputCreateInput,
    AgentNodeExecutionKeyValueDataCreateInput,
    AgentNodeExecutionUpdateInput,
    AgentNodeExecutionWhereInput,
)
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from pydantic.fields import Field

from backend.util import type as type_utils
from backend.util.exceptions import DatabaseError
from backend.util.json import SafeJson
from backend.util.models import Pagination
from backend.util.retry import func_retry
from backend.util.settings import Config
from backend.util.truncate import truncate

from .block import (
    BlockInput,
    BlockType,
    CompletedBlockOutput,
    get_block,
    get_io_block_ids,
    get_webhook_block_ids,
)
from .db import BaseDbModel, query_raw_with_schema
from .event_bus import AsyncRedisEventBus, RedisEventBus
from .includes import (
    EXECUTION_RESULT_INCLUDE,
    EXECUTION_RESULT_ORDER,
    GRAPH_EXECUTION_INCLUDE_WITH_NODES,
    graph_execution_include,
)
from .model import CredentialsMetaInput, GraphExecutionStats, NodeExecutionStats

T = TypeVar("T")

logger = logging.getLogger(__name__)
config = Config()


# -------------------------- Models -------------------------- #


class BlockErrorStats(BaseModel):
    """Typed data structure for block error statistics."""

    block_id: str
    total_executions: int
    failed_executions: int

    @property
    def error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.failed_executions / self.total_executions) * 100


ExecutionStatus = AgentExecutionStatus
NodeInputMask = Mapping[str, JsonValue]
NodesInputMasks = Mapping[str, NodeInputMask]

# dest: source
VALID_STATUS_TRANSITIONS = {
    ExecutionStatus.QUEUED: [
        ExecutionStatus.INCOMPLETE,
    ],
    ExecutionStatus.RUNNING: [
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.QUEUED,
        ExecutionStatus.TERMINATED,  # For resuming halted execution
    ],
    ExecutionStatus.COMPLETED: [
        ExecutionStatus.RUNNING,
    ],
    ExecutionStatus.FAILED: [
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.QUEUED,
        ExecutionStatus.RUNNING,
    ],
    ExecutionStatus.TERMINATED: [
        ExecutionStatus.INCOMPLETE,
        ExecutionStatus.QUEUED,
        ExecutionStatus.RUNNING,
    ],
}


class GraphExecutionMeta(BaseDbModel):
    id: str  # type: ignore # Override base class to make this required
    user_id: str
    graph_id: str
    graph_version: int
    inputs: Optional[BlockInput]  # no default -> required in the OpenAPI spec
    credential_inputs: Optional[dict[str, CredentialsMetaInput]]
    nodes_input_masks: Optional[dict[str, BlockInput]]
    preset_id: Optional[str]
    status: ExecutionStatus
    started_at: datetime
    ended_at: datetime
    is_shared: bool = False
    share_token: Optional[str] = None

    class Stats(BaseModel):
        model_config = ConfigDict(
            extra="allow",
            arbitrary_types_allowed=True,
        )

        cost: int = Field(
            default=0,
            description="Execution cost (cents)",
        )
        duration: float = Field(
            default=0,
            description="Seconds from start to end of run",
        )
        duration_cpu_only: float = Field(
            default=0,
            description="CPU sec of duration",
        )
        node_exec_time: float = Field(
            default=0,
            description="Seconds of total node runtime",
        )
        node_exec_time_cpu_only: float = Field(
            default=0,
            description="CPU sec of node_exec_time",
        )
        node_exec_count: int = Field(
            default=0,
            description="Number of node executions",
        )
        node_error_count: int = Field(
            default=0,
            description="Number of node errors",
        )
        error: str | None = Field(
            default=None,
            description="Error message if any",
        )
        activity_status: str | None = Field(
            default=None,
            description="AI-generated summary of what the agent did",
        )
        correctness_score: float | None = Field(
            default=None,
            description="AI-generated score (0.0-1.0) indicating how well the execution achieved its intended purpose",
        )

        def to_db(self) -> GraphExecutionStats:
            return GraphExecutionStats(
                cost=self.cost,
                walltime=self.duration,
                cputime=self.duration_cpu_only,
                nodes_walltime=self.node_exec_time,
                nodes_cputime=self.node_exec_time_cpu_only,
                node_count=self.node_exec_count,
                node_error_count=self.node_error_count,
                error=self.error,
                activity_status=self.activity_status,
                correctness_score=self.correctness_score,
            )

    stats: Stats | None

    @staticmethod
    def from_db(_graph_exec: AgentGraphExecution):
        now = datetime.now(timezone.utc)
        # TODO: make started_at and ended_at optional
        start_time = _graph_exec.startedAt or _graph_exec.createdAt
        end_time = _graph_exec.updatedAt or now

        try:
            stats = GraphExecutionStats.model_validate(_graph_exec.stats)
        except ValueError as e:
            if _graph_exec.stats is not None:
                logger.warning(
                    "Failed to parse invalid graph execution stats "
                    f"{_graph_exec.stats}: {e}"
                )
            stats = None

        return GraphExecutionMeta(
            id=_graph_exec.id,
            user_id=_graph_exec.userId,
            graph_id=_graph_exec.agentGraphId,
            graph_version=_graph_exec.agentGraphVersion,
            inputs=cast(BlockInput | None, _graph_exec.inputs),
            credential_inputs=(
                {
                    name: CredentialsMetaInput.model_validate(cmi)
                    for name, cmi in cast(dict, _graph_exec.credentialInputs).items()
                }
                if _graph_exec.credentialInputs
                else None
            ),
            nodes_input_masks=cast(
                dict[str, BlockInput] | None, _graph_exec.nodesInputMasks
            ),
            preset_id=_graph_exec.agentPresetId,
            status=ExecutionStatus(_graph_exec.executionStatus),
            started_at=start_time,
            ended_at=end_time,
            stats=(
                GraphExecutionMeta.Stats(
                    cost=stats.cost,
                    duration=stats.walltime,
                    duration_cpu_only=stats.cputime,
                    node_exec_time=stats.nodes_walltime,
                    node_exec_time_cpu_only=stats.nodes_cputime,
                    node_exec_count=stats.node_count,
                    node_error_count=stats.node_error_count,
                    error=(
                        str(stats.error)
                        if isinstance(stats.error, Exception)
                        else stats.error
                    ),
                    activity_status=stats.activity_status,
                    correctness_score=stats.correctness_score,
                )
                if stats
                else None
            ),
            is_shared=_graph_exec.isShared,
            share_token=_graph_exec.shareToken,
        )


class GraphExecution(GraphExecutionMeta):
    inputs: BlockInput  # type: ignore - incompatible override is intentional
    outputs: CompletedBlockOutput

    @staticmethod
    def from_db(_graph_exec: AgentGraphExecution):
        if _graph_exec.NodeExecutions is None:
            raise ValueError("Node executions must be included in query")

        graph_exec = GraphExecutionMeta.from_db(_graph_exec)

        complete_node_executions = sorted(
            [
                NodeExecutionResult.from_db(ne, _graph_exec.userId)
                for ne in _graph_exec.NodeExecutions
                if ne.executionStatus != ExecutionStatus.INCOMPLETE
            ],
            key=lambda ne: (ne.queue_time is None, ne.queue_time or ne.add_time),
        )

        inputs = {
            **(
                graph_exec.inputs
                or {
                    # fallback: extract inputs from Agent Input Blocks
                    exec.input_data["name"]: exec.input_data.get("value")
                    for exec in complete_node_executions
                    if (
                        (block := get_block(exec.block_id))
                        and block.block_type == BlockType.INPUT
                    )
                }
            ),
            **{
                # input from webhook-triggered block
                "payload": exec.input_data["payload"]
                for exec in complete_node_executions
                if (
                    (block := get_block(exec.block_id))
                    and block.block_type
                    in [BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL]
                )
            },
        }

        outputs: CompletedBlockOutput = defaultdict(list)
        for exec in complete_node_executions:
            if (
                block := get_block(exec.block_id)
            ) and block.block_type == BlockType.OUTPUT:
                outputs[exec.input_data["name"]].append(exec.input_data.get("value"))

        return GraphExecution(
            **{
                field_name: getattr(graph_exec, field_name)
                for field_name in GraphExecutionMeta.model_fields
                if field_name != "inputs"
            },
            inputs=inputs,
            outputs=outputs,
        )


class GraphExecutionWithNodes(GraphExecution):
    node_executions: list["NodeExecutionResult"]

    @staticmethod
    def from_db(_graph_exec: AgentGraphExecution):
        if _graph_exec.NodeExecutions is None:
            raise ValueError("Node executions must be included in query")

        graph_exec_with_io = GraphExecution.from_db(_graph_exec)

        node_executions = sorted(
            [
                NodeExecutionResult.from_db(ne, _graph_exec.userId)
                for ne in _graph_exec.NodeExecutions
            ],
            key=lambda ne: (ne.queue_time is None, ne.queue_time or ne.add_time),
        )

        return GraphExecutionWithNodes(
            **{
                field_name: getattr(graph_exec_with_io, field_name)
                for field_name in GraphExecution.model_fields
            },
            node_executions=node_executions,
        )

    def to_graph_execution_entry(
        self,
        user_context: "UserContext",
        compiled_nodes_input_masks: Optional[NodesInputMasks] = None,
        parent_graph_exec_id: Optional[str] = None,
    ):
        return GraphExecutionEntry(
            user_id=self.user_id,
            graph_id=self.graph_id,
            graph_version=self.graph_version or 0,
            graph_exec_id=self.id,
            nodes_input_masks=compiled_nodes_input_masks,
            user_context=user_context,
            parent_graph_exec_id=parent_graph_exec_id,
        )


class NodeExecutionResult(BaseModel):
    user_id: str
    graph_id: str
    graph_version: int
    graph_exec_id: str
    node_exec_id: str
    node_id: str
    block_id: str
    status: ExecutionStatus
    input_data: BlockInput
    output_data: CompletedBlockOutput
    add_time: datetime
    queue_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None

    @staticmethod
    def from_db(_node_exec: AgentNodeExecution, user_id: Optional[str] = None):
        try:
            stats = NodeExecutionStats.model_validate(_node_exec.stats or {})
        except (ValueError, ValidationError):
            stats = NodeExecutionStats()

        if stats.cleared_inputs:
            input_data: BlockInput = defaultdict()
            for name, messages in stats.cleared_inputs.items():
                input_data[name] = messages[-1] if messages else ""
        elif _node_exec.executionData:
            input_data = type_utils.convert(_node_exec.executionData, dict[str, Any])
        else:
            input_data: BlockInput = defaultdict()
            for data in _node_exec.Input or []:
                input_data[data.name] = type_utils.convert(data.data, JsonValue)

        output_data: CompletedBlockOutput = defaultdict(list)

        if stats.cleared_outputs:
            for name, messages in stats.cleared_outputs.items():
                output_data[name].extend(messages)
        else:
            for data in _node_exec.Output or []:
                output_data[data.name].append(type_utils.convert(data.data, JsonValue))

        graph_execution: AgentGraphExecution | None = _node_exec.GraphExecution
        if graph_execution:
            user_id = graph_execution.userId
        elif not user_id:
            raise ValueError(
                "AgentGraphExecution must be included or user_id passed in"
            )

        return NodeExecutionResult(
            user_id=user_id,
            graph_id=graph_execution.agentGraphId if graph_execution else "",
            graph_version=graph_execution.agentGraphVersion if graph_execution else 0,
            graph_exec_id=_node_exec.agentGraphExecutionId,
            block_id=_node_exec.Node.agentBlockId if _node_exec.Node else "",
            node_exec_id=_node_exec.id,
            node_id=_node_exec.agentNodeId,
            status=_node_exec.executionStatus,
            input_data=input_data,
            output_data=output_data,
            add_time=_node_exec.addedTime,
            queue_time=_node_exec.queuedTime,
            start_time=_node_exec.startedTime,
            end_time=_node_exec.endedTime,
        )

    def to_node_execution_entry(
        self, user_context: "UserContext"
    ) -> "NodeExecutionEntry":
        return NodeExecutionEntry(
            user_id=self.user_id,
            graph_exec_id=self.graph_exec_id,
            graph_id=self.graph_id,
            node_exec_id=self.node_exec_id,
            node_id=self.node_id,
            block_id=self.block_id,
            inputs=self.input_data,
            user_context=user_context,
        )


# --------------------- Model functions --------------------- #


async def get_graph_executions(
    graph_exec_id: Optional[str] = None,
    graph_id: Optional[str] = None,
    user_id: Optional[str] = None,
    statuses: Optional[list[ExecutionStatus]] = None,
    created_time_gte: Optional[datetime] = None,
    created_time_lte: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> list[GraphExecutionMeta]:
    """⚠️ **Optional `user_id` check**: MUST USE check in user-facing endpoints."""
    where_filter: AgentGraphExecutionWhereInput = {
        "isDeleted": False,
    }
    if graph_exec_id:
        where_filter["id"] = graph_exec_id
    if user_id:
        where_filter["userId"] = user_id
    if graph_id:
        where_filter["agentGraphId"] = graph_id
    if created_time_gte or created_time_lte:
        where_filter["createdAt"] = {
            "gte": created_time_gte or datetime.min.replace(tzinfo=timezone.utc),
            "lte": created_time_lte or datetime.max.replace(tzinfo=timezone.utc),
        }
    if statuses:
        where_filter["OR"] = [{"executionStatus": status} for status in statuses]

    executions = await AgentGraphExecution.prisma().find_many(
        where=where_filter,
        order={"createdAt": "desc"},
        take=limit,
    )
    return [GraphExecutionMeta.from_db(execution) for execution in executions]


async def get_graph_executions_count(
    user_id: Optional[str] = None,
    graph_id: Optional[str] = None,
    statuses: Optional[list[ExecutionStatus]] = None,
    created_time_gte: Optional[datetime] = None,
    created_time_lte: Optional[datetime] = None,
) -> int:
    """
    Get count of graph executions with optional filters.

    Args:
        user_id: Optional user ID to filter by
        graph_id: Optional graph ID to filter by
        statuses: Optional list of execution statuses to filter by
        created_time_gte: Optional minimum creation time
        created_time_lte: Optional maximum creation time

    Returns:
        Count of matching graph executions
    """
    where_filter: AgentGraphExecutionWhereInput = {
        "isDeleted": False,
    }

    if user_id:
        where_filter["userId"] = user_id

    if graph_id:
        where_filter["agentGraphId"] = graph_id

    if created_time_gte or created_time_lte:
        where_filter["createdAt"] = {
            "gte": created_time_gte or datetime.min.replace(tzinfo=timezone.utc),
            "lte": created_time_lte or datetime.max.replace(tzinfo=timezone.utc),
        }
    if statuses:
        where_filter["OR"] = [{"executionStatus": status} for status in statuses]

    count = await AgentGraphExecution.prisma().count(where=where_filter)
    return count


class GraphExecutionsPaginated(BaseModel):
    """Response schema for paginated graph executions."""

    executions: list[GraphExecutionMeta]
    pagination: Pagination


async def get_graph_executions_paginated(
    user_id: str,
    graph_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
    statuses: Optional[list[ExecutionStatus]] = None,
    created_time_gte: Optional[datetime] = None,
    created_time_lte: Optional[datetime] = None,
) -> GraphExecutionsPaginated:
    """Get paginated graph executions for a specific graph."""
    where_filter: AgentGraphExecutionWhereInput = {
        "isDeleted": False,
        "userId": user_id,
    }

    if graph_id:
        where_filter["agentGraphId"] = graph_id
    if created_time_gte or created_time_lte:
        where_filter["createdAt"] = {
            "gte": created_time_gte or datetime.min.replace(tzinfo=timezone.utc),
            "lte": created_time_lte or datetime.max.replace(tzinfo=timezone.utc),
        }
    if statuses:
        where_filter["OR"] = [{"executionStatus": status} for status in statuses]

    total_count = await AgentGraphExecution.prisma().count(where=where_filter)
    total_pages = (total_count + page_size - 1) // page_size

    offset = (page - 1) * page_size
    executions = await AgentGraphExecution.prisma().find_many(
        where=where_filter,
        order={"createdAt": "desc"},
        take=page_size,
        skip=offset,
    )

    return GraphExecutionsPaginated(
        executions=[GraphExecutionMeta.from_db(execution) for execution in executions],
        pagination=Pagination(
            total_items=total_count,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
        ),
    )


async def get_graph_execution_meta(
    user_id: str, execution_id: str
) -> GraphExecutionMeta | None:
    execution = await AgentGraphExecution.prisma().find_first(
        where={"id": execution_id, "isDeleted": False, "userId": user_id}
    )
    return GraphExecutionMeta.from_db(execution) if execution else None


@overload
async def get_graph_execution(
    user_id: str,
    execution_id: str,
    include_node_executions: Literal[True],
) -> GraphExecutionWithNodes | None: ...


@overload
async def get_graph_execution(
    user_id: str,
    execution_id: str,
    include_node_executions: Literal[False] = False,
) -> GraphExecution | None: ...


@overload
async def get_graph_execution(
    user_id: str,
    execution_id: str,
    include_node_executions: bool = False,
) -> GraphExecution | GraphExecutionWithNodes | None: ...


async def get_graph_execution(
    user_id: str,
    execution_id: str,
    include_node_executions: bool = False,
) -> GraphExecution | GraphExecutionWithNodes | None:
    execution = await AgentGraphExecution.prisma().find_first(
        where={"id": execution_id, "isDeleted": False, "userId": user_id},
        include=(
            GRAPH_EXECUTION_INCLUDE_WITH_NODES
            if include_node_executions
            else graph_execution_include(
                [*get_io_block_ids(), *get_webhook_block_ids()]
            )
        ),
    )
    if not execution:
        return None

    return (
        GraphExecutionWithNodes.from_db(execution)
        if include_node_executions
        else GraphExecution.from_db(execution)
    )


async def get_child_graph_executions(
    parent_exec_id: str,
) -> list[GraphExecutionMeta]:
    """
    Get all child executions of a parent execution.

    Args:
        parent_exec_id: Parent graph execution ID

    Returns:
        List of child graph executions
    """
    children = await AgentGraphExecution.prisma().find_many(
        where={"parentGraphExecutionId": parent_exec_id, "isDeleted": False}
    )

    return [GraphExecutionMeta.from_db(child) for child in children]


async def create_graph_execution(
    graph_id: str,
    graph_version: int,
    starting_nodes_input: list[tuple[str, BlockInput]],  # list[(node_id, BlockInput)]
    inputs: Mapping[str, JsonValue],
    user_id: str,
    preset_id: Optional[str] = None,
    credential_inputs: Optional[Mapping[str, CredentialsMetaInput]] = None,
    nodes_input_masks: Optional[NodesInputMasks] = None,
    parent_graph_exec_id: Optional[str] = None,
) -> GraphExecutionWithNodes:
    """
    Create a new AgentGraphExecution record.
    Returns:
        The id of the AgentGraphExecution and the list of ExecutionResult for each node.
    """
    result = await AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph_id,
            "agentGraphVersion": graph_version,
            "executionStatus": ExecutionStatus.INCOMPLETE,
            "inputs": SafeJson(inputs),
            "credentialInputs": (
                SafeJson(credential_inputs) if credential_inputs else Json({})
            ),
            "nodesInputMasks": (
                SafeJson(nodes_input_masks) if nodes_input_masks else Json({})
            ),
            "NodeExecutions": {
                "create": [
                    AgentNodeExecutionCreateInput(
                        agentNodeId=node_id,
                        executionStatus=ExecutionStatus.QUEUED,
                        queuedTime=datetime.now(tz=timezone.utc),
                        Input={
                            "create": [
                                {"name": name, "data": SafeJson(data)}
                                for name, data in node_input.items()
                            ]
                        },
                    )
                    for node_id, node_input in starting_nodes_input
                ]
            },
            "userId": user_id,
            "agentPresetId": preset_id,
            "parentGraphExecutionId": parent_graph_exec_id,
        },
        include=GRAPH_EXECUTION_INCLUDE_WITH_NODES,
    )

    return GraphExecutionWithNodes.from_db(result)


async def upsert_execution_input(
    node_id: str,
    graph_exec_id: str,
    input_name: str,
    input_data: JsonValue,
    node_exec_id: str | None = None,
) -> tuple[str, BlockInput]:
    """
    Insert AgentNodeExecutionInputOutput record for as one of AgentNodeExecution.Input.
    If there is no AgentNodeExecution that has no `input_name` as input, create new one.

    Args:
        node_id: The id of the AgentNode.
        graph_exec_id: The id of the AgentGraphExecution.
        input_name: The name of the input data.
        input_data: The input data to be inserted.
        node_exec_id: [Optional] The id of the AgentNodeExecution that has no `input_name` as input. If not provided, it will find the eligible incomplete AgentNodeExecution or create a new one.

    Returns:
        str: The id of the created or existing AgentNodeExecution.
        dict[str, Any]: Node input data; key is the input name, value is the input data.
    """
    existing_exec_query_filter: AgentNodeExecutionWhereInput = {
        "agentGraphExecutionId": graph_exec_id,
        "agentNodeId": node_id,
        "executionStatus": ExecutionStatus.INCOMPLETE,
        "Input": {
            "none": {
                "name": input_name,
                "time": {"gte": datetime.now(tz=timezone.utc) - timedelta(days=1)},
            }
        },
    }
    if node_exec_id:
        existing_exec_query_filter["id"] = node_exec_id

    existing_execution = await AgentNodeExecution.prisma().find_first(
        where=existing_exec_query_filter,
        order={"addedTime": "asc"},
        include={"Input": True},
    )
    json_input_data = SafeJson(input_data)

    if existing_execution:
        await AgentNodeExecutionInputOutput.prisma().create(
            data=AgentNodeExecutionInputOutputCreateInput(
                name=input_name,
                data=json_input_data,
                referencedByInputExecId=existing_execution.id,
            )
        )
        return existing_execution.id, {
            **{
                input_data.name: type_utils.convert(input_data.data, JsonValue)
                for input_data in existing_execution.Input or []
            },
            input_name: input_data,
        }

    elif not node_exec_id:
        result = await AgentNodeExecution.prisma().create(
            data=AgentNodeExecutionCreateInput(
                agentNodeId=node_id,
                agentGraphExecutionId=graph_exec_id,
                executionStatus=ExecutionStatus.INCOMPLETE,
                Input={"create": {"name": input_name, "data": json_input_data}},
            )
        )
        return result.id, {input_name: input_data}

    else:
        raise ValueError(
            f"NodeExecution {node_exec_id} not found or already has input {input_name}."
        )


async def upsert_execution_output(
    node_exec_id: str,
    output_name: str,
    output_data: Any | None,
) -> None:
    """
    Insert AgentNodeExecutionInputOutput record for as one of AgentNodeExecution.Output.
    """
    data: AgentNodeExecutionInputOutputCreateInput = {
        "name": output_name,
        "referencedByOutputExecId": node_exec_id,
    }
    if output_data is not None:
        data["data"] = SafeJson(output_data)
    await AgentNodeExecutionInputOutput.prisma().create(data=data)


async def update_graph_execution_start_time(
    graph_exec_id: str,
) -> GraphExecution | None:
    res = await AgentGraphExecution.prisma().update(
        where={"id": graph_exec_id},
        data={
            "executionStatus": ExecutionStatus.RUNNING,
            "startedAt": datetime.now(tz=timezone.utc),
        },
        include=graph_execution_include(
            [*get_io_block_ids(), *get_webhook_block_ids()]
        ),
    )
    return GraphExecution.from_db(res) if res else None


async def update_graph_execution_stats(
    graph_exec_id: str,
    status: ExecutionStatus | None = None,
    stats: GraphExecutionStats | None = None,
) -> GraphExecution | None:
    if not status and not stats:
        raise ValueError(
            f"Must provide either status or stats to update for execution {graph_exec_id}"
        )

    update_data: AgentGraphExecutionUpdateManyMutationInput = {}

    if stats:
        stats_dict = stats.model_dump()
        if isinstance(stats_dict.get("error"), Exception):
            stats_dict["error"] = str(stats_dict["error"])
        update_data["stats"] = SafeJson(stats_dict)

    if status:
        update_data["executionStatus"] = status

    where_clause: AgentGraphExecutionWhereInput = {"id": graph_exec_id}

    if status:
        if allowed_from := VALID_STATUS_TRANSITIONS.get(status, []):
            # Add OR clause to check if current status is one of the allowed source statuses
            where_clause["AND"] = [
                {"id": graph_exec_id},
                {"OR": [{"executionStatus": s} for s in allowed_from]},
            ]
        else:
            raise ValueError(
                f"Status {status} cannot be set via update for execution {graph_exec_id}. "
                f"This status can only be set at creation or is not a valid target status."
            )

    await AgentGraphExecution.prisma().update_many(
        where=where_clause,
        data=update_data,
    )

    graph_exec = await AgentGraphExecution.prisma().find_unique_or_raise(
        where={"id": graph_exec_id},
        include=graph_execution_include(
            [*get_io_block_ids(), *get_webhook_block_ids()]
        ),
    )

    return GraphExecution.from_db(graph_exec)


async def update_node_execution_status_batch(
    node_exec_ids: list[str],
    status: ExecutionStatus,
    stats: dict[str, Any] | None = None,
):
    await AgentNodeExecution.prisma().update_many(
        where={"id": {"in": node_exec_ids}},
        data=_get_update_status_data(status, None, stats),
    )


async def update_node_execution_status(
    node_exec_id: str,
    status: ExecutionStatus,
    execution_data: BlockInput | None = None,
    stats: dict[str, Any] | None = None,
) -> NodeExecutionResult:
    if status == ExecutionStatus.QUEUED and execution_data is None:
        raise ValueError("Execution data must be provided when queuing an execution.")

    res = await AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data=_get_update_status_data(status, execution_data, stats),
        include=EXECUTION_RESULT_INCLUDE,
    )
    if not res:
        raise ValueError(f"Execution {node_exec_id} not found.")

    return NodeExecutionResult.from_db(res)


def _get_update_status_data(
    status: ExecutionStatus,
    execution_data: BlockInput | None = None,
    stats: dict[str, Any] | None = None,
) -> AgentNodeExecutionUpdateInput:
    now = datetime.now(tz=timezone.utc)
    update_data: AgentNodeExecutionUpdateInput = {"executionStatus": status}

    if status == ExecutionStatus.QUEUED:
        update_data["queuedTime"] = now
    elif status == ExecutionStatus.RUNNING:
        update_data["startedTime"] = now
    elif status in (ExecutionStatus.FAILED, ExecutionStatus.COMPLETED):
        update_data["endedTime"] = now

    if execution_data:
        update_data["executionData"] = SafeJson(execution_data)
    if stats:
        update_data["stats"] = SafeJson(stats)

    return update_data


async def delete_graph_execution(
    graph_exec_id: str, user_id: str, soft_delete: bool = True
) -> None:
    if soft_delete:
        deleted_count = await AgentGraphExecution.prisma().update_many(
            where={"id": graph_exec_id, "userId": user_id}, data={"isDeleted": True}
        )
    else:
        deleted_count = await AgentGraphExecution.prisma().delete_many(
            where={"id": graph_exec_id, "userId": user_id}
        )
    if deleted_count < 1:
        raise DatabaseError(
            f"Could not delete graph execution #{graph_exec_id}: not found"
        )


async def get_node_execution(node_exec_id: str) -> NodeExecutionResult | None:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    execution = await AgentNodeExecution.prisma().find_first(
        where={"id": node_exec_id},
        include=EXECUTION_RESULT_INCLUDE,
    )
    if not execution:
        return None
    return NodeExecutionResult.from_db(execution)


async def get_node_executions(
    graph_exec_id: str | None = None,
    node_id: str | None = None,
    block_ids: list[str] | None = None,
    statuses: list[ExecutionStatus] | None = None,
    limit: int | None = None,
    created_time_gte: datetime | None = None,
    created_time_lte: datetime | None = None,
    include_exec_data: bool = True,
) -> list[NodeExecutionResult]:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    where_clause: AgentNodeExecutionWhereInput = {}
    if graph_exec_id:
        where_clause["agentGraphExecutionId"] = graph_exec_id
    if node_id:
        where_clause["agentNodeId"] = node_id
    if block_ids:
        where_clause["Node"] = {"is": {"agentBlockId": {"in": block_ids}}}
    if statuses:
        where_clause["OR"] = [{"executionStatus": status} for status in statuses]

    if created_time_gte or created_time_lte:
        where_clause["addedTime"] = {
            "gte": created_time_gte or datetime.min.replace(tzinfo=timezone.utc),
            "lte": created_time_lte or datetime.max.replace(tzinfo=timezone.utc),
        }

    executions = await AgentNodeExecution.prisma().find_many(
        where=where_clause,
        include=(
            EXECUTION_RESULT_INCLUDE
            if include_exec_data
            else {"Node": True, "GraphExecution": True}
        ),
        order=EXECUTION_RESULT_ORDER,
        take=limit,
    )
    res = [NodeExecutionResult.from_db(execution) for execution in executions]
    return res


async def get_latest_node_execution(
    node_id: str, graph_eid: str
) -> NodeExecutionResult | None:
    """⚠️ No `user_id` check: DO NOT USE without check in user-facing endpoints."""
    execution = await AgentNodeExecution.prisma().find_first(
        where={
            "agentGraphExecutionId": graph_eid,
            "agentNodeId": node_id,
            "OR": [
                {"executionStatus": ExecutionStatus.QUEUED},
                {"executionStatus": ExecutionStatus.RUNNING},
                {"executionStatus": ExecutionStatus.COMPLETED},
                {"executionStatus": ExecutionStatus.TERMINATED},
                {"executionStatus": ExecutionStatus.FAILED},
            ],
        },
        include=EXECUTION_RESULT_INCLUDE,
        order=EXECUTION_RESULT_ORDER,
    )
    if not execution:
        return None
    return NodeExecutionResult.from_db(execution)


# ----------------- Execution Infrastructure ----------------- #


class UserContext(BaseModel):
    """Generic user context for graph execution containing user-specific settings."""

    timezone: str


class GraphExecutionEntry(BaseModel):
    user_id: str
    graph_exec_id: str
    graph_id: str
    graph_version: int
    nodes_input_masks: Optional[NodesInputMasks] = None
    user_context: UserContext
    parent_graph_exec_id: Optional[str] = None


class NodeExecutionEntry(BaseModel):
    user_id: str
    graph_exec_id: str
    graph_id: str
    node_exec_id: str
    node_id: str
    block_id: str
    inputs: BlockInput
    user_context: UserContext


class ExecutionQueue(Generic[T]):
    """
    Queue for managing the execution of agents.
    This will be shared between different processes
    """

    def __init__(self):
        self.queue = Manager().Queue()

    def add(self, execution: T) -> T:
        self.queue.put(execution)
        return execution

    def get(self) -> T:
        return self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()

    def get_or_none(self) -> T | None:
        try:
            return self.queue.get_nowait()
        except Empty:
            return None


# --------------------- Event Bus --------------------- #


class ExecutionEventType(str, Enum):
    GRAPH_EXEC_UPDATE = "graph_execution_update"
    NODE_EXEC_UPDATE = "node_execution_update"
    ERROR_COMMS_UPDATE = "error_comms_update"


class GraphExecutionEvent(GraphExecution):
    event_type: Literal[ExecutionEventType.GRAPH_EXEC_UPDATE] = (
        ExecutionEventType.GRAPH_EXEC_UPDATE
    )


class NodeExecutionEvent(NodeExecutionResult):
    event_type: Literal[ExecutionEventType.NODE_EXEC_UPDATE] = (
        ExecutionEventType.NODE_EXEC_UPDATE
    )


class SharedExecutionResponse(BaseModel):
    """Public-safe response for shared executions"""

    id: str
    graph_name: str
    graph_description: Optional[str]
    status: ExecutionStatus
    created_at: datetime
    outputs: CompletedBlockOutput  # Only the final outputs, no intermediate data
    # Deliberately exclude: user_id, inputs, credentials, node details


ExecutionEvent = Annotated[
    GraphExecutionEvent | NodeExecutionEvent, Field(discriminator="event_type")
]


class RedisExecutionEventBus(RedisEventBus[ExecutionEvent]):
    Model = ExecutionEvent  # type: ignore

    @property
    def event_bus_name(self) -> str:
        return config.execution_event_bus_name

    def publish(self, res: GraphExecution | NodeExecutionResult):
        if isinstance(res, GraphExecution):
            self._publish_graph_exec_update(res)
        else:
            self._publish_node_exec_update(res)

    def _publish_node_exec_update(self, res: NodeExecutionResult):
        event = NodeExecutionEvent.model_validate(res.model_dump())
        self._publish(event, f"{res.user_id}/{res.graph_id}/{res.graph_exec_id}")

    def _publish_graph_exec_update(self, res: GraphExecution):
        event = GraphExecutionEvent.model_validate(res.model_dump())
        self._publish(event, f"{res.user_id}/{res.graph_id}/{res.id}")

    def _publish(self, event: ExecutionEvent, channel: str):
        """
        truncate inputs and outputs to avoid large payloads
        """
        limit = config.max_message_size_limit // 2
        if isinstance(event, GraphExecutionEvent):
            event.inputs = truncate(event.inputs, limit)
            event.outputs = truncate(event.outputs, limit)
        elif isinstance(event, NodeExecutionEvent):
            event.input_data = truncate(event.input_data, limit)
            event.output_data = truncate(event.output_data, limit)

        super().publish_event(event, channel)

    def listen(
        self, user_id: str, graph_id: str = "*", graph_exec_id: str = "*"
    ) -> Generator[ExecutionEvent, None, None]:
        for event in self.listen_events(f"{user_id}/{graph_id}/{graph_exec_id}"):
            yield event


class AsyncRedisExecutionEventBus(AsyncRedisEventBus[ExecutionEvent]):
    Model = ExecutionEvent  # type: ignore

    @property
    def event_bus_name(self) -> str:
        return config.execution_event_bus_name

    @func_retry
    async def publish(self, res: GraphExecutionMeta | NodeExecutionResult):
        if isinstance(res, GraphExecutionMeta):
            await self._publish_graph_exec_update(res)
        else:
            await self._publish_node_exec_update(res)

    async def _publish_node_exec_update(self, res: NodeExecutionResult):
        event = NodeExecutionEvent.model_validate(res.model_dump())
        await self._publish(event, f"{res.user_id}/{res.graph_id}/{res.graph_exec_id}")

    async def _publish_graph_exec_update(self, res: GraphExecutionMeta):
        # GraphExecutionEvent requires inputs and outputs fields that GraphExecutionMeta doesn't have
        # Add default empty values for compatibility
        event_data = res.model_dump()
        event_data.setdefault("inputs", {})
        event_data.setdefault("outputs", {})
        event = GraphExecutionEvent.model_validate(event_data)
        await self._publish(event, f"{res.user_id}/{res.graph_id}/{res.id}")

    async def _publish(self, event: ExecutionEvent, channel: str):
        """
        truncate inputs and outputs to avoid large payloads
        """
        limit = config.max_message_size_limit // 2
        if isinstance(event, GraphExecutionEvent):
            event.inputs = truncate(event.inputs, limit)
            event.outputs = truncate(event.outputs, limit)
        elif isinstance(event, NodeExecutionEvent):
            event.input_data = truncate(event.input_data, limit)
            event.output_data = truncate(event.output_data, limit)

        await super().publish_event(event, channel)

    async def listen(
        self, user_id: str, graph_id: str = "*", graph_exec_id: str = "*"
    ) -> AsyncGenerator[ExecutionEvent, None]:
        async for event in self.listen_events(f"{user_id}/{graph_id}/{graph_exec_id}"):
            yield event


# --------------------- KV Data Functions --------------------- #


async def get_execution_kv_data(user_id: str, key: str) -> Any | None:
    """
    Get key-value data for a user and key.

    Args:
        user_id: The id of the User.
        key: The key to retrieve data for.

    Returns:
        The data associated with the key, or None if not found.
    """
    kv_data = await AgentNodeExecutionKeyValueData.prisma().find_unique(
        where={"userId_key": {"userId": user_id, "key": key}}
    )
    return (
        type_utils.convert(kv_data.data, type[Any])
        if kv_data and kv_data.data
        else None
    )


async def set_execution_kv_data(
    user_id: str, node_exec_id: str, key: str, data: Any
) -> Any | None:
    """
    Set key-value data for a user and key.

    Args:
        user_id: The id of the User.
        node_exec_id: The id of the AgentNodeExecution.
        key: The key to store data under.
        data: The data to store.
    """
    resp = await AgentNodeExecutionKeyValueData.prisma().upsert(
        where={"userId_key": {"userId": user_id, "key": key}},
        data={
            "create": AgentNodeExecutionKeyValueDataCreateInput(
                userId=user_id,
                agentNodeExecutionId=node_exec_id,
                key=key,
                data=SafeJson(data) if data is not None else None,
            ),
            "update": {
                "agentNodeExecutionId": node_exec_id,
                "data": SafeJson(data) if data is not None else None,
            },
        },
    )
    return type_utils.convert(resp.data, type[Any]) if resp and resp.data else None


async def get_block_error_stats(
    start_time: datetime, end_time: datetime
) -> list[BlockErrorStats]:
    """Get block execution stats using efficient SQL aggregation."""

    query_template = """
    SELECT 
        n."agentBlockId" as block_id,
        COUNT(*) as total_executions,
        SUM(CASE WHEN ne."executionStatus" = 'FAILED' THEN 1 ELSE 0 END) as failed_executions
    FROM {schema_prefix}"AgentNodeExecution" ne
    JOIN {schema_prefix}"AgentNode" n ON ne."agentNodeId" = n.id
    WHERE ne."addedTime" >= $1::timestamp AND ne."addedTime" <= $2::timestamp
    GROUP BY n."agentBlockId"
    HAVING COUNT(*) >= 10
    """

    result = await query_raw_with_schema(query_template, start_time, end_time)

    # Convert to typed data structures
    return [
        BlockErrorStats(
            block_id=row["block_id"],
            total_executions=int(row["total_executions"]),
            failed_executions=int(row["failed_executions"]),
        )
        for row in result
    ]


async def update_graph_execution_share_status(
    execution_id: str,
    user_id: str,
    is_shared: bool,
    share_token: str | None,
    shared_at: datetime | None,
) -> None:
    """Update the sharing status of a graph execution."""
    await AgentGraphExecution.prisma().update(
        where={"id": execution_id},
        data={
            "isShared": is_shared,
            "shareToken": share_token,
            "sharedAt": shared_at,
        },
    )


async def get_graph_execution_by_share_token(
    share_token: str,
) -> SharedExecutionResponse | None:
    """Get a shared execution with limited public-safe data."""
    execution = await AgentGraphExecution.prisma().find_first(
        where={
            "shareToken": share_token,
            "isShared": True,
            "isDeleted": False,
        },
        include={
            "AgentGraph": True,
            "NodeExecutions": {
                "include": {
                    "Output": True,
                    "Node": {
                        "include": {
                            "AgentBlock": True,
                        }
                    },
                },
            },
        },
    )

    if not execution:
        return None

    # Extract outputs from OUTPUT blocks only (consistent with GraphExecution.from_db)
    outputs: CompletedBlockOutput = defaultdict(list)
    if execution.NodeExecutions:
        for node_exec in execution.NodeExecutions:
            if node_exec.Node and node_exec.Node.agentBlockId:
                # Get the block definition to check its type
                block = get_block(node_exec.Node.agentBlockId)

                if block and block.block_type == BlockType.OUTPUT:
                    # For OUTPUT blocks, the data is stored in executionData or Input
                    # The executionData contains the structured input with 'name' and 'value' fields
                    if hasattr(node_exec, "executionData") and node_exec.executionData:
                        exec_data = type_utils.convert(
                            node_exec.executionData, dict[str, Any]
                        )
                        if "name" in exec_data:
                            name = exec_data["name"]
                            value = exec_data.get("value")
                            outputs[name].append(value)
                    elif node_exec.Input:
                        # Build input_data from Input relation
                        input_data = {}
                        for data in node_exec.Input:
                            if data.name and data.data is not None:
                                input_data[data.name] = type_utils.convert(
                                    data.data, JsonValue
                                )

                        if "name" in input_data:
                            name = input_data["name"]
                            value = input_data.get("value")
                            outputs[name].append(value)

    return SharedExecutionResponse(
        id=execution.id,
        graph_name=(
            execution.AgentGraph.name
            if (execution.AgentGraph and execution.AgentGraph.name)
            else "Untitled Agent"
        ),
        graph_description=(
            execution.AgentGraph.description if execution.AgentGraph else None
        ),
        status=ExecutionStatus(execution.executionStatus),
        created_at=execution.createdAt,
        outputs=outputs,
    )
