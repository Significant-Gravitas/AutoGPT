from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from multiprocessing import Manager
from typing import Any

from prisma.models import (
    AgentGraphExecution,
    AgentNode,
    AgentNodeExecution,
    AgentNodeExecutionInputOutput,
)
from pydantic import BaseModel

from autogpt_server.util import json


class NodeExecution(BaseModel):
    graph_exec_id: str
    node_exec_id: str
    node_id: str
    data: dict[str, Any]


class ExecutionStatus(str, Enum):
    INCOMPLETE = "INCOMPLETE"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExecutionQueue:
    """
    Queue for managing the execution of agents.
    This will be shared between different processes
    """

    def __init__(self):
        self.queue = Manager().Queue()

    def add(self, execution: NodeExecution) -> NodeExecution:
        self.queue.put(execution)
        return execution

    def get(self) -> NodeExecution:
        return self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()


class ExecutionResult(BaseModel):
    graph_id: str
    graph_exec_id: str
    node_exec_id: str
    node_id: str
    status: ExecutionStatus
    input_data: dict[str, Any]  # 1 input pin should consume exactly 1 data.
    output_data: dict[str, list[Any]]  # but 1 output pin can produce multiple output.
    add_time: datetime
    queue_time: datetime | None
    start_time: datetime | None
    end_time: datetime | None

    @staticmethod
    def from_db(execution: AgentNodeExecution):
        input_data: dict[str, Any] = defaultdict()
        for data in execution.Input or []:
            input_data[data.name] = json.loads(data.data)

        output_data: dict[str, Any] = defaultdict(list)
        for data in execution.Output or []:
            output_data[data.name].append(json.loads(data.data))

        node: AgentNode | None = execution.AgentNode
        
        return ExecutionResult(
            graph_id=node.agentGraphId if node else "",
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


async def create_graph_execution(
    graph_id: str, node_ids: list[str], data: dict[str, Any]
) -> tuple[str, list[ExecutionResult]]:
    """
    Create a new AgentGraphExecution record.
    Returns:
        The id of the AgentGraphExecution and the list of ExecutionResult for each node.
    """
    result = await AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph_id,
            "AgentNodeExecutions": {
                "create": [  # type: ignore
                    {
                        "agentNodeId": node_id,
                        "executionStatus": ExecutionStatus.INCOMPLETE,
                        "Input": {
                            "create": [
                                {"name": name, "data": json.dumps(data)}
                                for name, data in data.items()
                            ]
                        },
                    }
                    for node_id in node_ids
                ]
            },
        },
        include={"AgentNodeExecutions": True},
    )

    return result.id, [
        ExecutionResult.from_db(execution)
        for execution in result.AgentNodeExecutions or []
    ]


async def upsert_execution_input(
    node_id: str,
    graph_exec_id: str,
    input_name: str,
    data: Any,
) -> str:
    """
    Insert AgentNodeExecutionInputOutput record for as one of AgentNodeExecution.Input.
    If there is no AgentNodeExecution that has no `input_name` as input, create new one.

    Returns:
        The id of the created or existing AgentNodeExecution.
    """
    existing_execution = await AgentNodeExecution.prisma().find_first(
        where={  # type: ignore
            "agentNodeId": node_id,
            "agentGraphExecutionId": graph_exec_id,
            "Input": {"every": {"name": {"not": input_name}}},
        },
        order={"addedTime": "asc"},
    )
    json_data = json.dumps(data)

    if existing_execution:
        await AgentNodeExecutionInputOutput.prisma().create(
            data={
                "name": input_name,
                "data": json_data,
                "referencedByInputExecId": existing_execution.id,
            }
        )
        return existing_execution.id

    else:
        result = await AgentNodeExecution.prisma().create(
            data={
                "agentNodeId": node_id,
                "agentGraphExecutionId": graph_exec_id,
                "executionStatus": ExecutionStatus.INCOMPLETE,
                "Input": {"create": {"name": input_name, "data": json_data}},
            }
        )
        return result.id


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


async def update_execution_status(node_exec_id: str, status: ExecutionStatus) -> None:
    now = datetime.now(tz=timezone.utc)
    data = {
        **({"executionStatus": status}),
        **({"queuedTime": now} if status == ExecutionStatus.QUEUED else {}),
        **({"startedTime": now} if status == ExecutionStatus.RUNNING else {}),
        **({"endedTime": now} if status == ExecutionStatus.FAILED else {}),
        **({"endedTime": now} if status == ExecutionStatus.COMPLETED else {}),
    }

    res = await AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data=data,  # type: ignore
    )
    if not res:
        raise ValueError(f"Execution {node_exec_id} not found.")


async def list_executions(graph_id: str) -> list[str]:
    executions = await AgentGraphExecution.prisma().find_many(
        where={"agentGraphId": graph_id},
    )
    return [execution.id for execution in executions]


async def get_execution_results(graph_exec_id: str) -> list[ExecutionResult]:
    executions = await AgentNodeExecution.prisma().find_many(
        where={"agentGraphExecutionId": graph_exec_id},
        include={"Input": True, "Output": True},
        order={"addedTime": "asc"},
    )
    res = [ExecutionResult.from_db(execution) for execution in executions]
    return res


async def get_execution_result(
    graph_exec_id: str, node_exec_id: str
) -> ExecutionResult:
    execution = await AgentNodeExecution.prisma().find_first_or_raise(
        where={"agentGraphExecutionId": graph_exec_id, "id": node_exec_id},
        include={"Input": True, "Output": True, "AgentNode": True},
        order={"addedTime": "asc"},
    )
    res = ExecutionResult.from_db(execution)
    return res


async def get_node_execution_input(node_exec_id: str) -> dict[str, Any]:
    """
    Get execution node input data from the previous node execution result.

    Returns:
        dictionary of input data, key is the input name, value is the input data.
    """
    execution = await AgentNodeExecution.prisma().find_unique_or_raise(
        where={"id": node_exec_id},
        include={
            "Input": True,
            "AgentNode": True,
        },
    )
    if not execution.AgentNode:
        raise ValueError(f"Node {execution.agentNodeId} not found.")

    exec_input = json.loads(execution.AgentNode.constantInput)
    for input_data in execution.Input or []:
        exec_input[input_data.name] = json.loads(input_data.data)

    return merge_execution_input(exec_input)


LIST_SPLIT = "_$_"
DICT_SPLIT = "_#_"
OBJC_SPLIT = "_@_"


def parse_execution_output(output: tuple[str, Any], name: str) -> Any | None:
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


def merge_execution_input(data: dict[str, Any]) -> dict[str, Any]:
    # Merge all input with <input_name>_$_<index> into a single list.
    list_input: list[Any] = []
    for key, value in data.items():
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
    for key, value in data.items():
        if DICT_SPLIT not in key:
            continue
        name, index = key.split(DICT_SPLIT)
        data[name] = data.get(name, {})
        data[name][index] = value

    # Merge all input with <input_name>_@_<index> into a single object.
    for key, value in data.items():
        if OBJC_SPLIT not in key:
            continue
        name, index = key.split(OBJC_SPLIT)
        if not isinstance(data[name], object):
            data[name] = type("Object", (object,), data[name])()
        setattr(data[name], index, value)

    return data
