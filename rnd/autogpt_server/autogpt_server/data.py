import uuid
from typing import Any
from multiprocessing import Queue


class Event:
    """
    Defines an event type for tiggers to send to the 
    executor manager (EM) and for the EM to send to the executors.
    """
    
    def __init__(self, execution_id: str, etype: str, data: Any):
        self.execution_id = execution_id
        self.etype = etype
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
        self.queue: Queue[Event] = Queue()

    def add(self, etype: str, data: Any) -> str:
        execution_id = uuid.uuid4()
        self.queue.put(Event(str(execution_id), etype, data))
        return str(execution_id)

    def get(self) -> Event | None:
        return self.queue.get()

    def empty(self) -> bool:
        return self.queue.empty()
