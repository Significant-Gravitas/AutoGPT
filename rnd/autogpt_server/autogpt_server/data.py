import uuid 

from multiprocessing import Queue

class Execution:
    def __init__(self, execution_id: str, data: str):
        self.execution_id = execution_id
        self.data = data

# TODO: Replace this by a persistent queue.
class ExecutionQueue:
    def __init__(self):
        self.queue = Queue()

    def add(self, data: str) -> str:
        execution_id = uuid.uuid4()
        self.queue.put(Execution(str(execution_id), data))
        return str(execution_id)

    def get(self) -> Execution | None:
        return self.queue.get()
    
    def empty(self) -> bool:
        return self.queue.empty()
    