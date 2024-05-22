import logging
import time

from concurrent.futures import ThreadPoolExecutor

from autogpt_server.data import ExecutionQueue

logger = logging.getLogger(__name__)


# TODO: Replace this by an actual Agent Execution.
def __execute(id: str, data: str) -> None:
    for i in range(5):
        print(f"Executor processing step {i}, execution_id: {id}, data: {data}")
        time.sleep(1)
    print(f"Executor processing completed, execution_id: {id}, data: {data}")


def start_executor(pool_size: int, queue: ExecutionQueue) -> None:
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        while True:
            execution = queue.get()
            if not execution:
                time.sleep(1)
                continue
            executor.submit(__execute, execution.execution_id, execution.data)
