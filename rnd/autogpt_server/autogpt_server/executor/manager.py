import asyncio
import logging

from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Any

from autogpt_server.data import db
from autogpt_server.data.block import Block, get_block
from autogpt_server.data.graph import Node, get_node, get_node_input
from autogpt_server.data.execution import (
    Execution,
    ExecutionQueue,
    enqueue_execution,
    complete_execution,
    fail_execution,
    start_execution,
)
from autogpt_server.util.service import AppService, expose

logger = logging.getLogger(__name__)


def get_log_prefix(run_id: str, exec_id: str, block_name: str = "-"):
    return f"[Execution graph-{run_id}|node-{exec_id}|{block_name}]"


def execute_node(loop: asyncio.AbstractEventLoop, data: Execution) -> Execution | None:
    """
    Execute a node in the graph. This will trigger a block execution on a node,
    persist the execution result, and return the subsequent node to be executed.

    Args:
        loop: The event loop to run the async functions.
        data: The execution data for executing the current node.

    Returns:
        The subsequent node to be enqueued, or None if there is no subsequent node.
    """
    run_id = data.run_id
    exec_id = data.id
    exec_data = data.data
    node_id = data.node_id

    asyncio.set_event_loop(loop)
    wait = lambda f: loop.run_until_complete(f)

    node: Optional[Node] = wait(get_node(node_id))
    if not node:
        logger.error(f"Node {node_id} not found.")
        return None

    node_block: Optional[Block] = wait(get_block(node.block_id))
    if not node_block:
        logger.error(f"Block {node.block_id} not found.")
        return None

    # Execute the node
    prefix = get_log_prefix(run_id, exec_id, node_block.name)
    logger.warning(f"{prefix} execute with input:\n`{exec_data}`")
    wait(start_execution(exec_id))

    try:
        output_name, output_data = node_block.execute(exec_data)
        logger.warning(f"{prefix} executed with output [{output_name}]:`{output_data}`")
        wait(complete_execution(exec_id, (output_name, output_data)))
    except Exception as e:
        logger.exception(f"{prefix} failed with error: %s", e)
        wait(fail_execution(exec_id, e))
        raise e

    # Try to enqueue next eligible nodes
    if output_name not in node.output_nodes:
        logger.error(f"{prefix} Output [{output_name}] has no subsequent node.")
        return None

    next_node_id = node.output_nodes[output_name]
    next_node: Optional[Node] = wait(get_node(next_node_id))
    if not next_node:
        logger.error(f"{prefix} Error, next node {next_node_id} not found.")
        return None

    next_node_input: dict[str, Any] = wait(get_node_input(next_node, run_id))
    next_node_block: Block | None = wait(get_block(next_node.block_id))
    if not next_node_block:
        logger.error(f"{prefix} Error, next block {next_node.block_id} not found.")
        return None

    if not set(next_node.input_nodes).issubset(next_node_input):
        logger.warning(
            f"{prefix} Skipped {next_node_id}-{next_node_block.name}, "
            f"missing: {set(next_node.input_nodes) - set(next_node_input)}"
        )
        return None

    if error := next_node_block.input_schema.validate_data(next_node_input):
        logger.warning(
            f"{prefix} Skipped {next_node_id}-{next_node_block.name}, {error}"
        )
        return None

    logger.warning(f"{prefix} Enqueue next node {next_node_id}-{next_node_block.name}")
    return Execution(run_id=run_id, node_id=next_node_id, data=next_node_input)


class Executor:
    loop: asyncio.AbstractEventLoop

    @classmethod
    def on_executor_start(cls):
        cls.loop = asyncio.new_event_loop()
        cls.loop.run_until_complete(db.connect())

    @classmethod
    def on_start_execution(cls, data: Execution) -> Optional[Execution | None]:
        """
        A synchronous version of `execute_node`, to be used in the ProcessPoolExecutor.
        """
        prefix = get_log_prefix(data.run_id, data.id)
        try:
            logger.warning(f"{prefix} Start execution")
            return execute_node(cls.loop, data)
        except Exception as e:
            logger.error(f"{prefix} Error: {e}")


class ExecutionManager(AppService):

    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.queue = ExecutionQueue()

    def run_service(self):
        def on_complete_execution(f: asyncio.Future[Execution | None]):
            exception = f.exception()
            if exception:
                logger.exception("Error during execution!! %s", exception)
                return exception

            execution = f.result()
            if execution:
                return self.__add_execution(execution)

            return None

        with ProcessPoolExecutor(
                max_workers=self.pool_size,
                initializer=Executor.on_executor_start,
        ) as executor:
            logger.warning(f"Execution manager started with {self.pool_size} workers.")
            while True:
                future = executor.submit(
                    Executor.on_start_execution,
                    self.queue.get()
                )
                future.add_done_callback(on_complete_execution)  # type: ignore

    @expose
    def add_execution(self, run_id: str, node_id: str, data: dict[str, Any]) -> str:
        try:
            execution = Execution(run_id=run_id, node_id=node_id, data=data)
            self.__add_execution(execution)
            return execution.id
        except Exception as e:
            raise Exception("Error adding execution ", e)

    def __add_execution(self, execution: Execution) -> Execution:
        self.run_and_wait(enqueue_execution(execution))
        return self.queue.add(execution)
