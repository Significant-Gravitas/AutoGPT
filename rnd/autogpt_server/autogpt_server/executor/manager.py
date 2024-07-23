import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, Coroutine, Generator, TypeVar

if TYPE_CHECKING:
    from autogpt_server.server.server import AgentServer

from autogpt_server.data import db
from autogpt_server.data.block import Block, BlockData, BlockInput, get_block
from autogpt_server.data.execution import ExecutionQueue, ExecutionStatus
from autogpt_server.data.execution import NodeExecution as Execution
from autogpt_server.data.execution import (
    create_graph_execution,
    get_node_execution_input,
    merge_execution_input,
    parse_execution_output,
    update_execution_status,
    upsert_execution_input,
    upsert_execution_output,
)
from autogpt_server.data.graph import Graph, Link, Node, get_graph, get_node
from autogpt_server.util.service import AppService, expose, get_service_client

logger = logging.getLogger(__name__)


def get_log_prefix(graph_eid: str, node_eid: str, block_name: str = "-"):
    return f"[ExecutionManager] [graph-{graph_eid}|node-{node_eid}|{block_name}]"


T = TypeVar("T")
ExecutionStream = Generator[Execution, None, None]


def execute_node(
    loop: asyncio.AbstractEventLoop, api_client: "AgentServer", data: Execution
) -> ExecutionStream:
    """
    Execute a node in the graph. This will trigger a block execution on a node,
    persist the execution result, and return the subsequent node to be executed.

    Args:
        loop: The event loop to run the async functions.
        api_client: The client to send execution updates to the server.
        data: The execution data for executing the current node.

    Returns:
        The subsequent node to be enqueued, or None if there is no subsequent node.
    """
    graph_exec_id = data.graph_exec_id
    node_exec_id = data.node_exec_id
    node_id = data.node_id

    asyncio.set_event_loop(loop)

    def wait(f: Coroutine[T, Any, T]) -> T:
        return loop.run_until_complete(f)

    def update_execution(status: ExecutionStatus):
        api_client.send_execution_update(
            wait(update_execution_status(node_exec_id, status)).model_dump()
        )

    node = wait(get_node(node_id))
    if not node:
        logger.error(f"Node {node_id} not found.")
        return

    node_block = get_block(node.block_id)  # type: ignore
    if not node_block:
        logger.error(f"Block {node.block_id} not found.")
        return

    # Sanity check: validate the execution input.
    prefix = get_log_prefix(graph_exec_id, node_exec_id, node_block.name)
    exec_data, error = validate_exec(node, data.data, resolve_input=False)
    if not exec_data:
        logger.error(f"{prefix} Skip execution, input validation error: {error}")
        return

    # Execute the node
    logger.warning(f"{prefix} execute with input:\n`{exec_data}`")
    update_execution(ExecutionStatus.RUNNING)

    try:
        for output_name, output_data in node_block.execute(exec_data):
            logger.warning(f"{prefix} Executed, output [{output_name}]:`{output_data}`")
            wait(upsert_execution_output(node_exec_id, output_name, output_data))
            update_execution(ExecutionStatus.COMPLETED)

            for execution in enqueue_next_nodes(
                api_client=api_client,
                loop=loop,
                node=node,
                output=(output_name, output_data),
                graph_exec_id=graph_exec_id,
                prefix=prefix,
            ):
                yield execution
    except Exception as e:
        error_msg = f"{e.__class__.__name__}: {e}"
        logger.exception(f"{prefix} failed with error. `%s`", error_msg)
        wait(upsert_execution_output(node_exec_id, "error", error_msg))
        update_execution(ExecutionStatus.FAILED)

        raise e


def enqueue_next_nodes(
    api_client: "AgentServer",
    loop: asyncio.AbstractEventLoop,
    node: Node,
    output: BlockData,
    graph_exec_id: str,
    prefix: str,
) -> list[Execution]:
    def wait(f: Coroutine[T, Any, T]) -> T:
        return loop.run_until_complete(f)

    def execution_update(node_exec_id: str, status: ExecutionStatus):
        api_client.send_execution_update(
            wait(update_execution_status(node_exec_id, status)).model_dump()
        )

    def update_execution_result(node_link: Link) -> Execution | None:
        next_output_name = node_link.source_name
        next_input_name = node_link.sink_name
        next_node_id = node_link.sink_id

        next_data = parse_execution_output(output, next_output_name)
        if next_data is None:
            return

        next_node = wait(get_node(next_node_id))
        if not next_node:
            logger.error(f"{prefix} Error, next node {next_node_id} not found.")
            return

        next_node_exec_id = wait(
            upsert_execution_input(
                node_id=next_node_id,
                graph_exec_id=graph_exec_id,
                input_name=next_input_name,
                data=next_data,
            )
        )

        next_node_input = wait(get_node_execution_input(next_node_exec_id))
        next_node_input, validation_msg = validate_exec(next_node, next_node_input)
        suffix = f"{next_output_name}~{next_input_name}#{next_node_id}:{validation_msg}"

        if not next_node_input:
            logger.warning(f"{prefix} Skipped queueing {suffix}")
            return

        # Input is complete, enqueue the execution.
        logger.warning(f"{prefix} Enqueued {suffix}")
        execution_update(next_node_exec_id, ExecutionStatus.QUEUED)
        return Execution(
            graph_exec_id=graph_exec_id,
            node_exec_id=next_node_exec_id,
            node_id=next_node.id,
            data=next_node_input,
        )

    return [
        execution
        for link in node.output_links
        if (execution := update_execution_result(link))
    ]


def validate_exec(
    node: Node,
    data: BlockInput,
    resolve_input: bool = True,
) -> tuple[BlockInput | None, str]:
    """
    Validate the input data for a node execution.

    Args:
        node: The node to execute.
        data: The input data for the node execution.
        resolve_input: Whether to resolve dynamic pins into dict/list/object.

    Returns:
        A tuple of the validated data and the block name.
        If the data is invalid, the first element will be None, and the second element
        will be an error message.
        If the data is valid, the first element will be the resolved input data, and
        the second element will be the block name.
    """
    node_block: Block | None = get_block(node.block_id)  # type: ignore
    if not node_block:
        return None, f"Block for {node.block_id} not found."

    error_prefix = f"Input data missing for {node_block.name}:"

    # Input data (without default values) should contain all required fields.
    input_fields_from_nodes = {link.sink_name for link in node.input_links}
    if not input_fields_from_nodes.issubset(data):
        return None, f"{error_prefix} {input_fields_from_nodes - set(data)}"

    # Merge input data with default values and resolve dynamic dict/list/object pins.
    data = {**node.input_default, **data}
    if resolve_input:
        data = merge_execution_input(data)

    # Input data post-merge should contain all required fields from the schema.
    input_fields_from_schema = node_block.input_schema.get_required_fields()
    if not input_fields_from_schema.issubset(data):
        return None, f"{error_prefix} {input_fields_from_schema - set(data)}"

    # Last validation: Validate the input values against the schema.
    if error := node_block.input_schema.validate_data(data):  # type: ignore
        error_message = f"Input data doesn't match {node_block.name}: {error}"
        logger.error(error_message)
        return None, error_message

    return data, node_block.name


def get_agent_server_client() -> "AgentServer":
    from autogpt_server.server.server import AgentServer

    return get_service_client(AgentServer)


class Executor:
    loop: asyncio.AbstractEventLoop

    @classmethod
    def on_executor_start(cls):
        cls.loop = asyncio.new_event_loop()
        cls.loop.run_until_complete(db.connect())
        cls.agent_server_client = get_agent_server_client()

    @classmethod
    def on_start_execution(cls, q: ExecutionQueue, data: Execution) -> bool:
        prefix = get_log_prefix(data.graph_exec_id, data.node_exec_id)
        try:
            logger.warning(f"{prefix} Start execution")
            for execution in execute_node(cls.loop, cls.agent_server_client, data):
                q.add(execution)
            return True
        except Exception as e:
            logger.exception(f"{prefix} Error: {e}")
            return False


class ExecutionManager(AppService):
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.queue = ExecutionQueue()

    def run_service(self):
        with ProcessPoolExecutor(
            max_workers=self.pool_size,
            initializer=Executor.on_executor_start,
        ) as executor:
            logger.warning(f"Execution manager started with {self.pool_size} workers.")
            while True:
                executor.submit(
                    Executor.on_start_execution,
                    self.queue,
                    self.queue.get(),
                )

    @property
    def agent_server_client(self) -> "AgentServer":
        return get_agent_server_client()

    @expose
    def add_execution(self, graph_id: str, data: BlockInput) -> dict[Any, Any]:
        graph: Graph | None = self.run_and_wait(get_graph(graph_id))
        if not graph:
            raise Exception(f"Graph #{graph_id} not found.")

        nodes_input = []
        for node in graph.starting_nodes:
            input_data, error = validate_exec(node, data)
            if not input_data:
                raise Exception(error)
            else:
                nodes_input.append((node.id, input_data))

        graph_exec_id, node_execs = self.run_and_wait(
            create_graph_execution(
                graph_id=graph_id,
                graph_version=graph.version,
                nodes_input=nodes_input,
            )
        )
        executions: list[BlockInput] = []
        for node_exec in node_execs:
            self.add_node_execution(
                Execution(
                    graph_exec_id=node_exec.graph_exec_id,
                    node_exec_id=node_exec.node_exec_id,
                    node_id=node_exec.node_id,
                    data=node_exec.input_data,
                )
            )

            executions.append(
                {
                    "id": node_exec.node_exec_id,
                    "node_id": node_exec.node_id,
                }
            )

        return {
            "id": graph_exec_id,
            "executions": executions,
        }

    def add_node_execution(self, execution: Execution) -> Execution:
        res = self.run_and_wait(
            update_execution_status(execution.node_exec_id, ExecutionStatus.QUEUED)
        )
        self.agent_server_client.send_execution_update(res.model_dump())
        return self.queue.add(execution)
