import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Coroutine, TypeVar

from autogpt_server.data import db
from autogpt_server.data.block import Block, get_block
from autogpt_server.data.execution import (
    get_node_execution_input,
    create_graph_execution,
    update_execution_status as execution_update,
    upsert_execution_output,
    upsert_execution_input,
    NodeExecution as Execution,
    ExecutionStatus,
    ExecutionQueue,
)
from autogpt_server.data.graph import Node, get_node, get_graph
from autogpt_server.util.service import AppService, expose

logger = logging.getLogger(__name__)


def get_log_prefix(graph_eid: str, node_eid: str, block_name: str = "-"):
    return f"[ExecutionManager] [graph-{graph_eid}|node-{node_eid}|{block_name}]"


T = TypeVar("T")


def execute_node(loop: asyncio.AbstractEventLoop, data: Execution) -> list[Execution]:
    """
    Execute a node in the graph. This will trigger a block execution on a node,
    persist the execution result, and return the subsequent node to be executed.

    Args:
        loop: The event loop to run the async functions.
        data: The execution data for executing the current node.

    Returns:
        The subsequent node to be enqueued, or None if there is no subsequent node.
    """
    graph_exec_id = data.graph_exec_id
    node_exec_id = data.node_exec_id
    exec_data = data.data
    node_id = data.node_id

    asyncio.set_event_loop(loop)

    def wait(f: Coroutine[T, Any, T]) -> T:
        return loop.run_until_complete(f)

    node = wait(get_node(node_id))
    if not node:
        logger.error(f"Node {node_id} not found.")
        return []

    node_block = wait(get_block(node.block_id))
    if not node_block:
        logger.error(f"Block {node.block_id} not found.")
        return []

    # Execute the node
    prefix = get_log_prefix(graph_exec_id, node_exec_id, node_block.name)
    logger.warning(f"{prefix} execute with input:\n`{exec_data}`")
    wait(execution_update(node_exec_id, ExecutionStatus.RUNNING))

    try:
        output_name, output_data = node_block.execute(exec_data)
        logger.warning(f"{prefix} executed with output [{output_name}]:`{output_data}`")
        wait(execution_update(node_exec_id, ExecutionStatus.COMPLETED))
        wait(upsert_execution_output(node_exec_id, output_name, output_data))
    except Exception as e:
        logger.exception(f"{prefix} failed with error: %s", e)
        wait(execution_update(node_exec_id, ExecutionStatus.FAILED))
        wait(upsert_execution_output(node_exec_id, "error", str(e)))
        raise e

    # Try to enqueue next eligible nodes
    next_node_ids = [nid for name, nid in node.output_nodes if name == output_name]
    if not next_node_ids:
        logger.error(f"{prefix} Output [{output_name}] has no subsequent node.")
        return []

    def validate_node_execution(next_node_id: str):
        next_node = wait(get_node(next_node_id))
        if not next_node:
            logger.error(f"{prefix} Error, next node {next_node_id} not found.")
            return None

        next_node_input_name = next(
            name for name, nid in next_node.input_nodes if nid == node_id
        )
        next_node_exec_id = wait(upsert_execution_input(
            node_id=next_node_id,
            graph_exec_id=graph_exec_id,
            input_name=next_node_input_name,
            data=output_data
        ))

        next_node_input = wait(get_node_execution_input(next_node_exec_id))
        is_valid, validation_resp = wait(validate_exec(next_node, next_node_input))
        if not is_valid:
            logger.warning(f"{prefix} Skipped {next_node_id}: {validation_resp}")
            return None

        logger.warning(f"{prefix} Enqueue next node {next_node_id}-{validation_resp}")
        return Execution(
            graph_exec_id=graph_exec_id,
            node_exec_id=next_node_exec_id,
            node_id=next_node_id,
            data=next_node_input
        )

    executions = []
    for nid in next_node_ids:
        if execution := validate_node_execution(nid):
            executions.append(execution)
    return executions


async def validate_exec(node: Node, data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate the input data for a node execution.

    Args:
        node: The node to execute.
        data: The input data for the node execution.

    Returns:
        A tuple of a boolean indicating if the data is valid, and a message if not.
        Return the executed block name if the data is valid.
    """
    node_block: Block | None = await(get_block(node.block_id))
    if not node_block:
        return False, f"Block for {node.block_id} not found."

    input_fields = node_block.input_schema.get_fields()
    if not input_fields.issubset(data):
        return False, f"Input data missing: {input_fields - set(data)}"

    if error := node_block.input_schema.validate_data(data):
        logger.error("Input value doesn't match schema: %s", error)
        return False, f"Input data doesn't match {node_block.name}: {error}"

    return True, node_block.name


class Executor:
    loop: asyncio.AbstractEventLoop

    @classmethod
    def on_executor_start(cls):
        cls.loop = asyncio.new_event_loop()
        cls.loop.run_until_complete(db.connect())

    @classmethod
    def on_start_execution(cls, data: Execution) -> list[Execution]:
        """
        A synchronous version of `execute_node`, to be used in the ProcessPoolExecutor.
        """
        prefix = get_log_prefix(data.graph_exec_id, data.node_exec_id)
        try:
            logger.warning(f"{prefix} Start execution")
            return execute_node(cls.loop, data)
        except Exception as e:
            logger.exception(f"{prefix} Error: {e}")
            return []


class ExecutionManager(AppService):

    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.queue = ExecutionQueue()

    def run_service(self):
        def on_complete_execution(f: asyncio.Future[list[Execution]]):
            exception = f.exception()
            if exception:
                logger.exception("Error during execution!! %s", exception)
                return exception

            executions = f.result()
            if not executions:
                return None

            return [
                self.add_node_execution(execution)
                for execution in executions
            ]

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
    def add_execution(self, graph_id: str, data: dict[str, Any]) -> dict:
        graph = self.run_and_wait(get_graph(graph_id))
        if not graph:
            raise Exception(f"Graph #{graph_id} not found.")

        # Currently, there is no constraint on the number of root nodes in the graph.
        for node in graph.starting_nodes:
            valid, error = self.run_and_wait(validate_exec(node, data))
            if not valid:
                raise Exception(error)

        graph_exec_id, node_execs = self.run_and_wait(create_graph_execution(
            graph_id=graph_id,
            node_ids=[node.id for node in graph.starting_nodes],
            data=data
        ))

        executions = []
        for node_exec in node_execs:
            self.add_node_execution(
                Execution(
                    graph_exec_id=node_exec.graph_exec_id,
                    node_exec_id=node_exec.node_exec_id,
                    node_id=node_exec.node_id,
                    data=data
                )
            )
            executions.append({
                "id": node_exec.node_exec_id,
                "node_id": node_exec.node_id,
            })

        return {
            "id": graph_exec_id,
            "executions": executions,
        }

    def add_node_execution(self, execution: Execution) -> Execution:
        self.run_and_wait(execution_update(
            execution.node_exec_id,
            ExecutionStatus.QUEUED
        ))
        return self.queue.add(execution)
