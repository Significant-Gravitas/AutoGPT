import json
import time

from datetime import datetime
from enum import Enum
from multiprocessing import Queue
from prisma.models import AgentNodeExecution

from autogpt_server.data.db import BaseDbModel


class Execution(BaseDbModel):
    """Data model for an execution of an Agent"""

    graph_exec_id: str
    node_id: str
    data: dict[str, str]


class ExecutionStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# TODO: This shared class make api & executor coupled in one machine.
# Replace this with a persistent & remote-hosted queue.
# One very likely candidate would be persisted Redis (Redis Queue).
# It will also open the possibility of using it for other purposes like
# caching, execution engine broker (like Celery), user session management etc.
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

    def get(self) -> Execution | None:
        return self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()


async def add_execution(execution: Execution, queue: ExecutionQueue) -> Execution:
    await AgentNodeExecution.prisma().create(
        data={
            "id": execution.id,
            "executionId": execution.graph_exec_id,
            "agentNodeId": execution.node_id,
            "executionStatus": ExecutionStatus.QUEUED,
            "inputData": json.dumps(execution.data),
            "creationTime": datetime.now(),
        }
    )
    return queue.add(execution)


async def start_execution(node_exec_id: str) -> None:
    await AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={
            "executionStatus": ExecutionStatus.RUNNING,
            "startTime": datetime.now(),
        },
    )


async def complete_execution(node_exec_id: str, output: (str, str)) -> None:
    output_name, output_data = output

    await AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={
            "executionStatus": ExecutionStatus.COMPLETED,
            "outputName": output_name,
            "outputData": output_data,
            "endTime": datetime.now(),
        },
    )


async def fail_execution(node_exec_id: str, error: Exception) -> None:
    await AgentNodeExecution.prisma().update(
        where={"id": node_exec_id},
        data={
            "executionStatus": ExecutionStatus.FAILED,
            "error": str(error),
            "endTime": datetime.now(),
        },
    )
