import logging
import time

from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor

from autogpt_server.data import ExecutionQueue

logger = logging.getLogger(__name__)


# TODO: Replace this by an actual Agent Execution.
def __execute(id: str, data: str) -> None:
    logger.warning(f"Executor processing started, execution_id: {id}, data: {data}")
    for i in range(5):
        logger.warning(
            f"Executor processing step {i}, execution_id: {id}, data: {data}"
        )
        time.sleep(1)
    logger.warning(f"Executor processing completed, execution_id: {id}, data: {data}")


def __start_executor(pool_size: int, queue: ExecutionQueue) -> None:
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        while True:
            execution = queue.get()
            if not execution:
                time.sleep(1)
                continue
            executor.submit(__execute, execution.execution_id, execution.data)


def start_executor(pool_size: int, queue: ExecutionQueue) -> None:
    executor_process = Process(target=__start_executor, args=(pool_size, queue))
    executor_process.start()
