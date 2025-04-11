import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from multiprocessing import Manager
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Generator,
    Generic,
    Literal,
    Optional,
    TypeVar,
    overload,
)

from prisma import Json
from prisma.enums import AgentExecutionStatus
from prisma.models import (
    AgentGraphExecution,
    AgentNodeExecution,
    AgentNodeExecutionInputOutput,
)
from prisma.types import (
    AgentGraphExecutionCreateInput,
    AgentGraphExecutionWhereInput,
    AgentNodeExecutionCreateInput,
    AgentNodeExecutionInputOutputCreateInput,
    AgentNodeExecutionUpdateInput,
    AgentNodeExecutionWhereInput,
)
from pydantic import BaseModel
from pydantic.fields import Field

from backend.server.v2.store.exceptions import DatabaseError
from backend.util import type as type_utils
from backend.util.settings import Config

from .block import BlockInput, BlockType, CompletedBlockOutput, get_block
from .db import BaseDbModel
from .includes import (
    EXECUTION_RESULT_INCLUDE,
    GRAPH_EXECUTION_INCLUDE,
    GRAPH_EXECUTION_INCLUDE_WITH_NODES,
)
from .model import GraphExecutionStats, NodeExecutionStats
from .queue import AsyncRedisEventBus, RedisEventBus

T = TypeVar("T")

logger = logging.getLogger(__name__)
config = Config()


# -------------------------- Models -------------------------- #


ExecutionStatus = AgentExecutionStatus


class GraphExecutionMeta(BaseDbModel):
    user_id: str
    graph_id: str
    graph_version: int
    preset_id: Optional[str] = None
    status: ExecutionStatus
    started_at: datetime
    ended_at: datetime

    class Stats(BaseModel):
        cost: int = Field(..., description="Execution cost (cents)")
        duration: float = Field(..., description="Seconds from start to end of run")
        node_exec_time: float = Field(..., description="Seconds of total node runtime")
        node_exec_count: int = Field(..., description="Number of node executions")

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
            preset_id=_graph_exec.agentPresetId,
            status=ExecutionStatus(_graph_exec.executionStatus),
            started_at=start_time,
            ended_at=end_time,
            stats=(
                GraphExecutionMeta.Stats(
                    cost=stats.cost,
                    duration=stats.walltime,
                    node_exec_time=stats.nodes_walltime,
                    node_exec_count=stats.node_count,
                )
                if stats
                else None
            ),
        )


class GraphExecution(GraphExecutionMeta):
    inputs: BlockInput
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
            **{
                # inputs from Agent Input Blocks
                exec.input_data["name"]: exec.input_data.get("value")
                for exec in complete_node_executions
                if (
                    (block := get_block(exec.block_id))
                    and block.block_type == BlockType.INPUT
                )
            },
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
                outputs[exec.input_data["name"]].append(
                    exec.input_data.get("value", None)
                )

        return GraphExecution(
            **{
                field_name: getattr(graph_exec, field_name)
                for field_name in GraphExecutionMeta.model_fields
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

    def to_graph_execution_entry(self):
        return GraphExecutionEntry(
            user_id=self.user_id,
            graph_id=self.graph_id,
            graph_version=self.graph_version or 0,
            graph_exec_id=self.id,
            start_node_execs=[
                NodeExecutionEntry(
                    user_id=self.user_id,
                    graph_exec_id=node_exec.graph_exec_id,
                    graph_id=node_exec.graph_id,
                    node_exec_id=node_exec.node_exec_id,
                    node_id=node_exec.node_id,
                    block_id=node_exec.block_id,
                    data=node_exec.input_data,
                )
                for node_exec in self.node_executions
            ],
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
        if _node_exec.executionData:
            # Execution that has been queued for execution will persist its data.
            input_data = type_utils.convert(_node_exec.executionData, dict[str, Any])
        else:
            # For incomplete execution, executionData will not be yet available.
            input_data: BlockInput = defaultdict()
            for data in _node_exec.Input or []:
                input_data[data.name] = type_utils.convert(data.data, type[Any])

        output_data: CompletedBlockOutput = defaultdict(list)
        for data in _node_exec.Output or []:
            output_data[data.name].append(type_utils.convert(data.data, type[Any]))

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


# --------------------- Model functions --------------------- #


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
            else GRAPH_EXECUTION_INCLUDE
        ),
    )
    if not execution:
        return None

    return (
        GraphExecutionWithNodes.from_db(execution)
        if include_node_executions
        else GraphExecution.from_db(execution)
    )


async def create_graph_execution(
    graph_id: str,
    graph_version: int,
    nodes_input: list[tuple[str, BlockInput]],
    user_id: str,
    preset_id: str | None = None,
) -> GraphExecutionWithNodes:
    """
    Create a new AgentGraphExecution record.
    Returns:
        The id of the AgentGraphExecution and the list of ExecutionResult for each node.
    """
    result = await AgentGraphExecution.prisma().create(
        data=AgentGraphExecutionCreateInput(
            agentGraphId=graph_id,
            agentGraphVersion=graph_version,
            executionStatus=ExecutionStatus.QUEUED,
            NodeExecutions={
                "create": [
                    AgentNodeExecutionCreateInput(
                        agentNodeId=node_id,
                        executionStatus=ExecutionStatus.QUEUED,
                        queuedTime=datetime.now(tz=timezone.utc),
                        Input={
                            "create": [
                                {"name": name, "data": Json(data)}
                                for name, data in node_input.items()
                            ]
                        },
                    )
                    for node_id, node_input in nodes_input
                ]
            },
            userId=user_id,
            agentPresetId=preset_id,
        ),
        include=GRAPH_EXECUTION_INCLUDE_WITH_NODES,
    )

    return GraphExecutionWithNodes.from_db(result)


async def upsert_execution_input(
    node_id: str,
    graph_exec_id: str,
    input_name: str,
    input_data: Any,
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
        "agentNodeId": node_id,
        "agentGraphExecutionId": graph_exec_id,
        "executionStatus": ExecutionStatus.INCOMPLETE,
        "Input": {"every": {"name": {"not": input_name}}},
    }
    if node_exec_id:
        existing_exec_query_filter["id"] = node_exec_id

    existing_execution = await AgentNodeExecution.prisma().find_first(
        where=existing_exec_query_filter,
        order={"addedTime": "asc"},
        include={"Input": True},
    )
    json_input_data = Json(input_data)

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
                input_data.name: type_utils.convert(input_data.data, type[Any])
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
    output_data: Any,
) -> None:
    """
    Insert AgentNodeExecutionInputOutput record for as one of AgentNodeExecution.Output.
    """
    await AgentNodeExecutionInputOutput.prisma().create(
        data=AgentNodeExecutionInputOutputCreateInput(
            name=output_name,
            data=Json(output_data),
            referencedByOutputExecId=node_exec_id,
        )
    )


async def update_graph_execution_start_time(
    graph_exec_id: str,
) -> GraphExecution | None:
    count = await AgentGraphExecution.prisma().update_many(
        where={
            "id": graph_exec_id,
            "executionStatus": ExecutionStatus.QUEUED,
        },
        data={
            "executionStatus": ExecutionStatus.RUNNING,
            "startedAt": datetime.now(tz=timezone.utc),
        },
    )
    if count == 0:
        return None

    res = await AgentGraphExecution.prisma().find_unique(
        where={"id": graph_exec_id},
        include=GRAPH_EXECUTION_INCLUDE,
    )
    return GraphExecution.from_db(res) if res else None


async def update_graph_execution_stats(
    graph_exec_id: str,
    status: ExecutionStatus,
    stats: GraphExecutionStats | None = None,
) -> GraphExecution | None:
    data = stats.model_dump() if stats else {}
    if isinstance(data.get("error"), Exception):
        data["error"] = str(data["error"])

    updated_count = await AgentGraphExecution.prisma().update_many(
        where={
            "id": graph_exec_id,
            "OR": [
                {"executionStatus": ExecutionStatus.RUNNING},
                {"executionStatus": ExecutionStatus.QUEUED},
            ],
        },
        data={
            "executionStatus": status,
            "stats": Json(data),
        },
    )
    if updated_count == 0:
        return None

    graph_exec = await AgentGraphExecution.prisma().find_unique_or_raise(
        where={"id": graph_exec_id},
        include=GRAPH_EXECUTION_INCLUDE,
    )
    return GraphExecution.from_db(graph_exec)


async def update_node_execution_stats(node_exec_id: str, stats: NodeExecutionStats):
    data = stats.model_dump()
    if isinstance(data["error"], Exception):
        data["error"] = str(data["error"])
    await AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={"stats": Json(data)},
    )


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
        update_data["executionData"] = Json(execution_data)
    if stats:
        update_data["stats"] = Json(stats)

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


async def get_node_execution_results(
    graph_exec_id: str,
    block_ids: list[str] | None = None,
    statuses: list[ExecutionStatus] | None = None,
    limit: int | None = None,
) -> list[NodeExecutionResult]:
    where_clause: AgentNodeExecutionWhereInput = {
        "agentGraphExecutionId": graph_exec_id,
    }
    if block_ids:
        where_clause["Node"] = {"is": {"agentBlockId": {"in": block_ids}}}
    if statuses:
        where_clause["OR"] = [{"executionStatus": status} for status in statuses]

    executions = await AgentNodeExecution.prisma().find_many(
        where=where_clause,
        include=EXECUTION_RESULT_INCLUDE,
        take=limit,
    )
    res = [NodeExecutionResult.from_db(execution) for execution in executions]
    return res


async def get_graph_executions_in_timerange(
    user_id: str, start_time: str, end_time: str
) -> list[GraphExecution]:
    try:
        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "startedAt": {
                    "gte": datetime.fromisoformat(start_time),
                    "lte": datetime.fromisoformat(end_time),
                },
                "userId": user_id,
                "isDeleted": False,
            },
            include=GRAPH_EXECUTION_INCLUDE,
        )
        return [GraphExecution.from_db(execution) for execution in executions]
    except Exception as e:
        raise DatabaseError(
            f"Failed to get executions in timerange {start_time} to {end_time} for user {user_id}: {e}"
        ) from e


async def get_latest_node_execution(
    node_id: str, graph_eid: str
) -> NodeExecutionResult | None:
    execution = await AgentNodeExecution.prisma().find_first(
        where={
            "agentNodeId": node_id,
            "agentGraphExecutionId": graph_eid,
            "NOT": [{"executionStatus": ExecutionStatus.INCOMPLETE}],
        },
        order=[
            {"queuedTime": "desc"},
            {"addedTime": "desc"},
        ],
        include=EXECUTION_RESULT_INCLUDE,
    )
    if not execution:
        return None
    return NodeExecutionResult.from_db(execution)


async def get_incomplete_node_executions(
    node_id: str, graph_eid: str
) -> list[NodeExecutionResult]:
    executions = await AgentNodeExecution.prisma().find_many(
        where={
            "agentNodeId": node_id,
            "agentGraphExecutionId": graph_eid,
            "executionStatus": ExecutionStatus.INCOMPLETE,
        },
        include=EXECUTION_RESULT_INCLUDE,
    )
    return [NodeExecutionResult.from_db(execution) for execution in executions]


# ----------------- Execution Infrastructure ----------------- #


class GraphExecutionEntry(BaseModel):
    user_id: str
    graph_exec_id: str
    graph_id: str
    graph_version: int
    start_node_execs: list["NodeExecutionEntry"]


class NodeExecutionEntry(BaseModel):
    user_id: str
    graph_exec_id: str
    graph_id: str
    node_exec_id: str
    node_id: str
    block_id: str
    data: BlockInput


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


# --------------------- Event Bus --------------------- #


class ExecutionEventType(str, Enum):
    GRAPH_EXEC_UPDATE = "graph_execution_update"
    NODE_EXEC_UPDATE = "node_execution_update"


class GraphExecutionEvent(GraphExecution):
    event_type: Literal[ExecutionEventType.GRAPH_EXEC_UPDATE] = (
        ExecutionEventType.GRAPH_EXEC_UPDATE
    )


class NodeExecutionEvent(NodeExecutionResult):
    event_type: Literal[ExecutionEventType.NODE_EXEC_UPDATE] = (
        ExecutionEventType.NODE_EXEC_UPDATE
    )


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
            self.publish_graph_exec_update(res)
        else:
            self.publish_node_exec_update(res)

    def publish_node_exec_update(self, res: NodeExecutionResult):
        event = NodeExecutionEvent.model_validate(res.model_dump())
        self.publish_event(event, f"{res.user_id}/{res.graph_id}/{res.graph_exec_id}")

    def publish_graph_exec_update(self, res: GraphExecution):
        event = GraphExecutionEvent.model_validate(res.model_dump())
        self.publish_event(event, f"{res.user_id}/{res.graph_id}/{res.id}")

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

    async def publish(self, res: GraphExecutionMeta | NodeExecutionResult):
        if isinstance(res, GraphExecutionMeta):
            await self.publish_graph_exec_update(res)
        else:
            await self.publish_node_exec_update(res)

    async def publish_node_exec_update(self, res: NodeExecutionResult):
        event = NodeExecutionEvent.model_validate(res.model_dump())
        await self.publish_event(
            event, f"{res.user_id}/{res.graph_id}/{res.graph_exec_id}"
        )

    async def publish_graph_exec_update(self, res: GraphExecutionMeta):
        event = GraphExecutionEvent.model_validate(res.model_dump())
        await self.publish_event(event, f"{res.user_id}/{res.graph_id}/{res.id}")

    async def listen(
        self, user_id: str, graph_id: str = "*", graph_exec_id: str = "*"
    ) -> AsyncGenerator[ExecutionEvent, None]:
        async for event in self.listen_events(f"{user_id}/{graph_id}/{graph_exec_id}"):
            yield event
