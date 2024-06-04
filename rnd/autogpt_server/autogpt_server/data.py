import uuid
from multiprocessing import Queue


class Execution:
    """Data model for an execution of an Agent"""

    def __init__(self, execution_id: str, data: str):
        self.execution_id = execution_id
        self.data = data


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

    def add(self, data: str) -> str:
        execution_id = uuid.uuid4()
        self.queue.put(Execution(str(execution_id), data))
        return str(execution_id)

    def get(self) -> Execution | None:
        return self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()
