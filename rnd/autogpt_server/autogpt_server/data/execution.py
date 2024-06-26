import json
from datetime import datetime
from enum import Enum
from multiprocessing import Queue
from typing import Any

from prisma.models import AgentNodeExecution

from autogpt_server.data.db import BaseDbModel


class Execution(BaseDbModel):
    """Data model for an execution of an Agent"""

    run_id: str
    node_id: str
    data: dict[str, Any]


class ExecutionStatus(str, Enum):
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
        self.queue: Queue[Execution] = Queue()

    def add(self, execution: Execution) -> Execution:
        self.queue.put(execution)
        return execution

    def get(self) -> Execution:
        return self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()


class ExecutionResult(BaseDbModel):
    run_id: str
    execution_id: str
    node_id: str
    status: ExecutionStatus
    input_data: dict[str, Any]
    output_name: str
    output_data: Any
    creation_time: datetime
    start_time: datetime | None
    end_time: datetime | None

    @staticmethod
    def from_db(execution: AgentNodeExecution):
        return ExecutionResult(
            run_id=execution.executionId,
            node_id=execution.agentNodeId,
            execution_id=execution.id,
            status=ExecutionStatus(execution.executionStatus),
            input_data=json.loads(execution.inputData or "{}"),
            output_name=execution.outputName or "",
            output_data=json.loads(execution.outputData or "{}"),
            creation_time=execution.creationTime,
            start_time=execution.startTime,
            end_time=execution.endTime,
        )


# --------------------- Model functions --------------------- #


async def enqueue_execution(execution: Execution) -> None:
    await AgentNodeExecution.prisma().create(
        data={
            "id": execution.id,
            "executionId": execution.run_id,
            "agentNodeId": execution.node_id,
            "executionStatus": ExecutionStatus.QUEUED,
            "inputData": json.dumps(execution.data),
            "creationTime": datetime.now(),
        }
    )


async def start_execution(exec_id: str) -> None:
    await AgentNodeExecution.prisma().update(
        where={"id": exec_id},
        data={
            "executionStatus": ExecutionStatus.RUNNING,
            "startTime": datetime.now(),
        },
    )


async def complete_execution(exec_id: str, output: tuple[str, Any]) -> None:
    output_name, output_data = output

    await AgentNodeExecution.prisma().update(
        where={"id": exec_id},
        data={
            "executionStatus": ExecutionStatus.COMPLETED,
            "outputName": output_name,
            "outputData": json.dumps(output_data),
            "endTime": datetime.now(),
        },
    )


async def fail_execution(exec_id: str, error: Exception) -> None:
    await AgentNodeExecution.prisma().update(
        where={"id": exec_id},
        data={
            "executionStatus": ExecutionStatus.FAILED,
            "outputName": "error",
            "outputData": str(error),
            "endTime": datetime.now(),
        },
    )


async def get_executions(run_id: str) -> list[ExecutionResult]:
    executions = await AgentNodeExecution.prisma().find_many(
        where={"executionId": run_id},
        order={"startTime": "asc"},
    )
    res = [ExecutionResult.from_db(execution) for execution in executions]
    return res
