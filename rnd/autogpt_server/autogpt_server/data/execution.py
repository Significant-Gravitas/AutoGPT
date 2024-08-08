from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from multiprocessing import Manager
from typing import Any, Generic, TypeVar

from prisma.models import (
    AgentGraphExecution,
    AgentNodeExecution,
    AgentNodeExecutionInputOutput,
)
from prisma.types import AgentGraphExecutionWhereInput
from pydantic import BaseModel

from autogpt_server.data.block import BlockData, BlockInput, CompletedBlockOutput
from autogpt_server.util import json


class GraphExecution(BaseModel):
    graph_exec_id: str
    start_node_execs: list["NodeExecution"]


class NodeExecution(BaseModel):
    graph_exec_id: str
    node_exec_id: str
    node_id: str
    data: BlockInput


class ExecutionStatus(str, Enum):
    INCOMPLETE = "INCOMPLETE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


T = TypeVar("T")


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


class ExecutionResult(BaseModel):
    graph_id: str
    graph_version: int
    graph_exec_id: str
    node_exec_id: str
    node_id: str
    status: ExecutionStatus
    input_data: BlockInput
    output_data: CompletedBlockOutput
    add_time: datetime
    queue_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None

    @staticmethod
    def from_db(execution: AgentNodeExecution):
        if execution.executionData:
            # Execution that has been queued for execution will persist its data.
            input_data = json.loads(execution.executionData)
        else:
            # For incomplete execution, executionData will not be yet available.
            input_data: BlockInput = defaultdict()
            for data in execution.Input or []:
                input_data[data.name] = json.loads(data.data)

        output_data: CompletedBlockOutput = defaultdict(list)
        for data in execution.Output or []:
            output_data[data.name].append(json.loads(data.data))

        graph_execution: AgentGraphExecution | None = execution.AgentGraphExecution

        return ExecutionResult(
            graph_id=graph_execution.agentGraphId if graph_execution else "",
            graph_version=graph_execution.agentGraphVersion if graph_execution else 0,
            graph_exec_id=execution.agentGraphExecutionId,
            node_exec_id=execution.id,
            node_id=execution.agentNodeId,
            status=ExecutionStatus(execution.executionStatus),
            input_data=input_data,
            output_data=output_data,
            add_time=execution.addedTime,
            queue_time=execution.queuedTime,
            start_time=execution.startedTime,
            end_time=execution.endedTime,
        )


# --------------------- Model functions --------------------- #

EXECUTION_RESULT_INCLUDE = {
    "Input": True,
    "Output": True,
    "AgentNode": True,
    "AgentGraphExecution": True,
}


async def create_graph_execution(
    graph_id: str,
    graph_version: int,
    nodes_input: list[tuple[str, BlockInput]],
    user_id: str,
) -> tuple[str, list[ExecutionResult]]:
    """
    Create a new AgentGraphExecution record.
    Returns:
        The id of the AgentGraphExecution and the list of ExecutionResult for each node.
    """
    result = await AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph_id,
            "agentGraphVersion": graph_version,
            "AgentNodeExecutions": {
                "create": [  # type: ignore
                    {
                        "agentNodeId": node_id,
                        "executionStatus": ExecutionStatus.INCOMPLETE,
                        "Input": {
                            "create": [
                                {"name": name, "data": json.dumps(data)}
                                for name, data in node_input.items()
                            ]
                        },
                    }
                    for node_id, node_input in nodes_input
                ]
            },
            "userId": user_id,
        },
        include={
            "AgentNodeExecutions": {"include": EXECUTION_RESULT_INCLUDE}  # type: ignore
        },
    )

    return result.id, [
        ExecutionResult.from_db(execution)
        for execution in result.AgentNodeExecutions or []
    ]


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
        * The id of the created or existing AgentNodeExecution.
        * Dict of node input data, key is the input name, value is the input data.
    """
    existing_execution = await AgentNodeExecution.prisma().find_first(
        where={  # type: ignore
            **({"id": node_exec_id} if node_exec_id else {}),
            "agentNodeId": node_id,
            "agentGraphExecutionId": graph_exec_id,
            "executionStatus": ExecutionStatus.INCOMPLETE,
            "Input": {"every": {"name": {"not": input_name}}},
        },
        order={"addedTime": "asc"},
        include={"Input": True},
    )
    json_input_data = json.dumps(input_data)

    if existing_execution:
        await AgentNodeExecutionInputOutput.prisma().create(
            data={
                "name": input_name,
                "data": json_input_data,
                "referencedByInputExecId": existing_execution.id,
            }
        )
        return existing_execution.id, {
            **{
                input_data.name: json.loads(input_data.data)
                for input_data in existing_execution.Input or []
            },
            input_name: input_data,
        }

    elif not node_exec_id:
        result = await AgentNodeExecution.prisma().create(
            data={
                "agentNodeId": node_id,
                "agentGraphExecutionId": graph_exec_id,
                "executionStatus": ExecutionStatus.INCOMPLETE,
                "Input": {"create": {"name": input_name, "data": json_input_data}},
            }
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
        data={
            "name": output_name,
            "data": json.dumps(output_data),
            "referencedByOutputExecId": node_exec_id,
        }
    )


async def update_execution_status(
    node_exec_id: str, status: ExecutionStatus, execution_data: BlockInput | None = None
) -> ExecutionResult:
    if status == ExecutionStatus.QUEUED and execution_data is None:
        raise ValueError("Execution data must be provided when queuing an execution.")

    now = datetime.now(tz=timezone.utc)
    data = {
        **({"executionStatus": status}),
        **({"queuedTime": now} if status == ExecutionStatus.QUEUED else {}),
        **({"startedTime": now} if status == ExecutionStatus.RUNNING else {}),
        **({"endedTime": now} if status == ExecutionStatus.FAILED else {}),
        **({"endedTime": now} if status == ExecutionStatus.COMPLETED else {}),
        **({"executionData": json.dumps(execution_data)} if execution_data else {}),
    }

    res = await AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data=data,  # type: ignore
        include=EXECUTION_RESULT_INCLUDE,  # type: ignore
    )
    if not res:
        raise ValueError(f"Execution {node_exec_id} not found.")

    return ExecutionResult.from_db(res)


async def list_executions(graph_id: str, graph_version: int | None = None) -> list[str]:
    where: AgentGraphExecutionWhereInput = {"agentGraphId": graph_id}
    if graph_version is not None:
        where["agentGraphVersion"] = graph_version
    executions = await AgentGraphExecution.prisma().find_many(where=where)
    return [execution.id for execution in executions]


async def get_execution_results(graph_exec_id: str) -> list[ExecutionResult]:
    executions = await AgentNodeExecution.prisma().find_many(
        where={"agentGraphExecutionId": graph_exec_id},
        include=EXECUTION_RESULT_INCLUDE,  # type: ignore
        order=[
            {"queuedTime": "asc"},
            {"addedTime": "asc"},  # Fallback: Incomplete execs has no queuedTime.
        ],
    )
    res = [ExecutionResult.from_db(execution) for execution in executions]
    return res


LIST_SPLIT = "_$_"
DICT_SPLIT = "_#_"
OBJC_SPLIT = "_@_"


def parse_execution_output(output: BlockData, name: str) -> Any | None:
    # Allow extracting partial output data by name.
    output_name, output_data = output

    if name == output_name:
        return output_data

    if name.startswith(f"{output_name}{LIST_SPLIT}"):
        index = int(name.split(LIST_SPLIT)[1])
        if not isinstance(output_data, list) or len(output_data) <= index:
            return None
        return output_data[int(name.split(LIST_SPLIT)[1])]

    if name.startswith(f"{output_name}{DICT_SPLIT}"):
        index = name.split(DICT_SPLIT)[1]
        if not isinstance(output_data, dict) or index not in output_data:
            return None
        return output_data[index]

    if name.startswith(f"{output_name}{OBJC_SPLIT}"):
        index = name.split(OBJC_SPLIT)[1]
        if isinstance(output_data, object) and hasattr(output_data, index):
            return getattr(output_data, index)
        return None

    return None


def merge_execution_input(data: BlockInput) -> BlockInput:
    """
    Merge all dynamic input pins which described by the following pattern:
    - <input_name>_$_<index> for list input.
    - <input_name>_#_<index> for dict input.
    - <input_name>_@_<index> for object input.
    This function will construct pins with the same name into a single list/dict/object.
    """

    # Merge all input with <input_name>_$_<index> into a single list.
    items = list(data.items())
    list_input: list[Any] = []
    for key, value in items:
        if LIST_SPLIT not in key:
            continue
        name, index = key.split(LIST_SPLIT)
        if not index.isdigit():
            list_input.append((name, value, 0))
        else:
            list_input.append((name, value, int(index)))

    for name, value, _ in sorted(list_input, key=lambda x: x[2]):
        data[name] = data.get(name, [])
        data[name].append(value)

    # Merge all input with <input_name>_#_<index> into a single dict.
    for key, value in items:
        if DICT_SPLIT not in key:
            continue
        name, index = key.split(DICT_SPLIT)
        data[name] = data.get(name, {})
        data[name][index] = value

    # Merge all input with <input_name>_@_<index> into a single object.
    for key, value in items:
        if OBJC_SPLIT not in key:
            continue
        name, index = key.split(OBJC_SPLIT)
        if not isinstance(data[name], object):
            data[name] = type("Object", (object,), data[name])()
        setattr(data[name], index, value)

    return data


async def get_latest_execution(node_id: str, graph_eid: str) -> ExecutionResult | None:
    execution = await AgentNodeExecution.prisma().find_first(
        where={  # type: ignore
            "agentNodeId": node_id,
            "agentGraphExecutionId": graph_eid,
            "executionStatus": {"not": ExecutionStatus.INCOMPLETE},
            "executionData": {"not": None},
        },
        order={"queuedTime": "desc"},
        include=EXECUTION_RESULT_INCLUDE,  # type: ignore
    )
    if not execution:
        return None
    return ExecutionResult.from_db(execution)


async def get_incomplete_executions(
    node_id: str, graph_eid: str
) -> list[ExecutionResult]:
    executions = await AgentNodeExecution.prisma().find_many(
        where={  # type: ignore
            "agentNodeId": node_id,
            "agentGraphExecutionId": graph_eid,
            "executionStatus": ExecutionStatus.INCOMPLETE,
        },
        include=EXECUTION_RESULT_INCLUDE,  # type: ignore
    )
    return [ExecutionResult.from_db(execution) for execution in executions]
