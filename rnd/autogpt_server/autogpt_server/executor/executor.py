import asyncio
import logging

from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process

from autogpt_server.data import block, db, graph
from autogpt_server.data.execution import (
    Execution,
    ExecutionQueue,
    add_execution,
    complete_execution,
    start_execution,
    fail_execution,
)

logger = logging.getLogger(__name__)


def get_log_prefix(run_id: str, exec_id: str, block_name: str = "-"):
    return f"[Execution graph-{run_id}|node-{exec_id}|{block_name}]"


async def execute_node(data: Execution) -> Execution | None:
    """
    Execute a node in the graph. This will trigger a block execution on a node,
    persist the execution result, and return the subsequent node to be executed.

    Args:
        data: The execution data for executing the current node.

    Returns:
        The subsequent node to be enqueued, or None if there is no subsequent node.
    """
    run_id = data.run_id
    exec_id = data.id
    exec_data = data.data
    node_id = data.node_id

    node = await graph.get_node(node_id)
    if not node:
        logger.error(f"Node {node_id} not found.")
        return None

    node_block = await block.get_block(node.block_id)
    if not node_block:
        logger.error(f"Block {node.block_id} not found.")
        return None

    # Execute the node
    prefix = get_log_prefix(run_id, exec_id, node_block.name)
    logger.warning(f"{prefix} execute with input:\n{exec_data}")
    await start_execution(exec_id)

    try:
        output_name, output_data = await node_block.execute(exec_data)
        logger.warning(f"{prefix} executed with output: `{output_name}`:{output_data}")
        await complete_execution(exec_id, (output_name, output_data))
    except Exception as e:
        logger.exception(f"{prefix} failed with error: %s", e)
        await fail_execution(exec_id, e)
        raise e

    # Try to enqueue next eligible nodes
    if output_name not in node.output_nodes:
        logger.error(f"{prefix} output name `{output_name}` has no subsequent node.")
        return None

    next_node_id = node.output_nodes[output_name]
    next_node = await graph.get_node(next_node_id)
    if not next_node:
        logger.error(f"{prefix} Error, next node {next_node_id} not found.")
        return None

    next_node_input = await graph.get_node_input(next_node, run_id)
    next_node_block = await next_node.block

    if error := next_node_block.input_schema.validate_data(next_node_input):
        logger.warning(
            f"{prefix} Skipped {next_node_id}-{next_node_block.name}, {error}")
        return None

    logger.warning(f"{prefix} Enqueue next node {next_node_id}-{next_node_block.name}")
    return Execution(
        run_id=run_id, node_id=next_node_id, data=next_node_input
    )


def execute_node_sync(data: Execution) -> Optional[Execution | None]:
    """
    A synchronous version of `execute_node`, to be used in the ProcessPoolExecutor.
    """
    prefix = get_log_prefix(data.run_id, data.id)
    try:
        logger.warning(f"{prefix} Start execution")
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(execute_node(data))
    except Exception as e:
        logger.error(f"{prefix} Error: {e}")


def start_executor(pool_size: int, queue: ExecutionQueue) -> None:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(db.connect())
    loop.run_until_complete(block.initialize_blocks())

    def on_complete_execution(f: asyncio.Future[Execution | None]):
        exception = f.exception()
        if exception:
            logger.exception("Error during execution!! %s", exception)
            return exception

        execution = f.result()
        if execution: 
            loop.run_until_complete(add_execution(execution, queue))
            return exception
        
        return None

    logger.warning("Executor started!")

    with ProcessPoolExecutor(
        max_workers=pool_size,
        initializer=db.connect_sync,
    ) as executor:
        while True:
            future = executor.submit(execute_node_sync, queue.get())
            future.add_done_callback(on_complete_execution)  # type: ignore


def start_executor_manager(pool_size: int, queue: ExecutionQueue) -> None:
    executor_process = Process(target=start_executor, args=(pool_size, queue))
    executor_process.start()
