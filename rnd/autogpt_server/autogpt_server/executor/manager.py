import asyncio
import logging
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Coroutine, Generator, TypeVar

if TYPE_CHECKING:
    from autogpt_server.server.server import AgentServer

from autogpt_server.blocks.basic import InputBlock
from autogpt_server.data import db
from autogpt_server.data.block import Block, BlockData, BlockInput, get_block
from autogpt_server.data.execution import (
    ExecutionQueue,
    ExecutionStatus,
    GraphExecution,
    NodeExecution,
    create_graph_execution,
    get_incomplete_executions,
    get_latest_execution,
    merge_execution_input,
    parse_execution_output,
    update_execution_status,
    upsert_execution_input,
    upsert_execution_output,
)
from autogpt_server.data.graph import Graph, Link, Node, get_graph, get_node
from autogpt_server.util.service import AppService, expose, get_service_client
from autogpt_server.util.settings import Config

logger = logging.getLogger(__name__)


def get_log_prefix(graph_eid: str, node_eid: str, block_name: str = "-"):
    return f"[ExecutionManager][graph-eid-{graph_eid}|node-eid-{node_eid}|{block_name}]"


T = TypeVar("T")
ExecutionStream = Generator[NodeExecution, None, None]


def execute_node(
    loop: asyncio.AbstractEventLoop, api_client: "AgentServer", data: NodeExecution
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
        exec_update = wait(update_execution_status(node_exec_id, status))
        api_client.send_execution_update(exec_update.model_dump())

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

            for execution in _enqueue_next_nodes(
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


@contextmanager
def synchronized(api_client: "AgentServer", key: Any):
    api_client.acquire_lock(key)
    try:
        yield
    finally:
        api_client.release_lock(key)


def _enqueue_next_nodes(
    api_client: "AgentServer",
    loop: asyncio.AbstractEventLoop,
    node: Node,
    output: BlockData,
    graph_exec_id: str,
    prefix: str,
) -> list[NodeExecution]:
    def wait(f: Coroutine[T, Any, T]) -> T:
        return loop.run_until_complete(f)

    def add_enqueued_execution(
        node_exec_id: str, node_id: str, data: BlockInput
    ) -> NodeExecution:
        exec_update = wait(
            update_execution_status(node_exec_id, ExecutionStatus.QUEUED, data)
        )
        api_client.send_execution_update(exec_update.model_dump())
        return NodeExecution(
            graph_exec_id=graph_exec_id,
            node_exec_id=node_exec_id,
            node_id=node_id,
            data=data,
        )

    def register_next_executions(node_link: Link) -> list[NodeExecution]:
        enqueued_executions = []
        next_output_name = node_link.source_name
        next_input_name = node_link.sink_name
        next_node_id = node_link.sink_id

        next_data = parse_execution_output(output, next_output_name)
        if next_data is None:
            return enqueued_executions

        next_node = wait(get_node(next_node_id))
        if not next_node:
            logger.error(f"{prefix} Error, next node {next_node_id} not found.")
            return enqueued_executions

        # Multiple node can register the same next node, we need this to be atomic
        # To avoid same execution to be enqueued multiple times,
        # Or the same input to be consumed multiple times.
        with synchronized(api_client, ("upsert_input", next_node_id, graph_exec_id)):
            # Add output data to the earliest incomplete execution, or create a new one.
            next_node_exec_id, next_node_input = wait(
                upsert_execution_input(
                    node_id=next_node_id,
                    graph_exec_id=graph_exec_id,
                    input_name=next_input_name,
                    input_data=next_data,
                )
            )

            # Complete missing static input pins data using the last execution input.
            static_link_names = {
                link.sink_name
                for link in next_node.input_links
                if link.is_static and link.sink_name not in next_node_input
            }
            if static_link_names and (
                latest_execution := wait(
                    get_latest_execution(next_node_id, graph_exec_id)
                )
            ):
                for name in static_link_names:
                    next_node_input[name] = latest_execution.input_data.get(name)

            # Validate the input data for the next node.
            next_node_input, validation_msg = validate_exec(next_node, next_node_input)
            suffix = f"{next_output_name}>{next_input_name}~{next_node_exec_id}:{validation_msg}"

            # Incomplete input data, skip queueing the execution.
            if not next_node_input:
                logger.warning(f"{prefix} Skipped queueing {suffix}")
                return enqueued_executions

            # Input is complete, enqueue the execution.
            logger.warning(f"{prefix} Enqueued {suffix}")
            enqueued_executions.append(
                add_enqueued_execution(next_node_exec_id, next_node_id, next_node_input)
            )

            # Next execution stops here if the link is not static.
            if not node_link.is_static:
                return enqueued_executions

            # If link is static, there could be some incomplete executions waiting for it.
            # Load and complete the input missing input data, and try to re-enqueue them.
            for iexec in wait(get_incomplete_executions(next_node_id, graph_exec_id)):
                idata = iexec.input_data
                ineid = iexec.node_exec_id

                static_link_names = {
                    link.sink_name
                    for link in next_node.input_links
                    if link.is_static and link.sink_name not in idata
                }
                for input_name in static_link_names:
                    idata[input_name] = next_node_input[input_name]

                idata, msg = validate_exec(next_node, idata)
                suffix = f"{next_output_name}>{next_input_name}~{ineid}:{msg}"
                if not idata:
                    logger.warning(f"{prefix} Re-enqueueing skipped: {suffix}")
                    continue
                logger.warning(f"{prefix} Re-enqueued {suffix}")
                enqueued_executions.append(
                    add_enqueued_execution(iexec.node_exec_id, next_node_id, idata)
                )
            return enqueued_executions

    return [
        execution
        for link in node.output_links
        for execution in register_next_executions(link)
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
    """
    This class contains event handlers for the process pool executor events.

    The main events are:
        on_node_executor_start: Initialize the process that executes the node.
        on_node_execution: Execution logic for a node.

        on_graph_executor_start: Initialize the process that executes the graph.
        on_graph_execution: Execution logic for a graph.

    The execution flow:
        1. Graph execution request is added to the queue.
        2. Graph executor loop picks the request from the queue.
        3. Graph executor loop submits the graph execution request to the executor pool.
      [on_graph_execution]
        4. Graph executor initialize the node execution queue.
        5. Graph executor adds the starting nodes to the node execution queue.
        6. Graph executor waits for all nodes to be executed.
      [on_node_execution]
        7. Node executor picks the node execution request from the queue.
        8. Node executor executes the node.
        9. Node executor enqueues the next executed nodes to the node execution queue.
    """

    @classmethod
    def on_node_executor_start(cls):
        cls.loop = asyncio.new_event_loop()
        cls.loop.run_until_complete(db.connect())
        cls.agent_server_client = get_agent_server_client()

    @classmethod
    def on_node_execution(cls, q: ExecutionQueue[NodeExecution], data: NodeExecution):
        prefix = get_log_prefix(data.graph_exec_id, data.node_exec_id)
        try:
            logger.warning(f"{prefix} Start node execution")
            for execution in execute_node(cls.loop, cls.agent_server_client, data):
                q.add(execution)
            logger.warning(f"{prefix} Finished node execution")
        except Exception as e:
            logger.exception(f"{prefix} Failed node execution: {e}")

    @classmethod
    def on_graph_executor_start(cls):
        cls.pool_size = Config().num_node_workers
        cls.executor = ProcessPoolExecutor(
            max_workers=cls.pool_size,
            initializer=cls.on_node_executor_start,
        )
        logger.warning(f"Graph executor started with max-{cls.pool_size} node workers.")

    @classmethod
    def on_graph_execution(cls, graph_data: GraphExecution):
        prefix = get_log_prefix(graph_data.graph_exec_id, "*")
        logger.warning(f"{prefix} Start graph execution")

        try:
            queue = ExecutionQueue[NodeExecution]()
            for node_exec in graph_data.start_node_execs:
                queue.add(node_exec)

            futures: dict[str, Future] = {}
            while not queue.empty():
                execution = queue.get()

                # Avoid parallel execution of the same node.
                fut = futures.get(execution.node_id)
                if fut and not fut.done():
                    cls.wait_future(fut)
                    logger.warning(f"{prefix} Re-enqueueing {execution.node_id}")
                    queue.add(execution)
                    continue

                futures[execution.node_id] = cls.executor.submit(
                    cls.on_node_execution, queue, execution
                )

                # Avoid terminating graph execution when some nodes are still running.
                while queue.empty() and futures:
                    for node_id, future in list(futures.items()):
                        if future.done():
                            del futures[node_id]
                        elif queue.empty():
                            cls.wait_future(future)

            logger.warning(f"{prefix} Finished graph execution")
        except Exception as e:
            logger.exception(f"{prefix} Failed graph execution: {e}")

    @classmethod
    def wait_future(cls, future: Future):
        try:
            future.result(timeout=3)
        except TimeoutError:
            # Avoid being blocked by long-running node, by not waiting its completion.
            pass


class ExecutionManager(AppService):
    def __init__(self):
        self.pool_size = Config().num_graph_workers
        self.queue = ExecutionQueue[GraphExecution]()

    def run_service(self):
        with ProcessPoolExecutor(
            max_workers=self.pool_size,
            initializer=Executor.on_graph_executor_start,
        ) as executor:
            logger.warning(
                f"Execution manager started with max-{self.pool_size} graph workers."
            )
            while True:
                executor.submit(Executor.on_graph_execution, self.queue.get())

    @property
    def agent_server_client(self) -> "AgentServer":
        return get_agent_server_client()

    @expose
    def add_execution(
        self, graph_id: str, data: BlockInput, user_id: str
    ) -> dict[Any, Any]:
        graph: Graph | None = self.run_and_wait(get_graph(graph_id, user_id=user_id))
        if not graph:
            raise Exception(f"Graph #{graph_id} not found.")
        graph.validate_graph(for_run=True)

        nodes_input = []
        for node in graph.starting_nodes:
            if isinstance(get_block(node.block_id), InputBlock):
                input_data = {"input": data}
            else:
                input_data = {}

            input_data, error = validate_exec(node, input_data)
            if not input_data:
                raise Exception(error)
            else:
                nodes_input.append((node.id, input_data))

        graph_exec_id, node_execs = self.run_and_wait(
            create_graph_execution(
                graph_id=graph_id,
                graph_version=graph.version,
                nodes_input=nodes_input,
                user_id=user_id,
            )
        )

        starting_node_execs = []
        for node_exec in node_execs:
            starting_node_execs.append(
                NodeExecution(
                    graph_exec_id=node_exec.graph_exec_id,
                    node_exec_id=node_exec.node_exec_id,
                    node_id=node_exec.node_id,
                    data=node_exec.input_data,
                )
            )
            exec_update = self.run_and_wait(
                update_execution_status(
                    node_exec.node_exec_id, ExecutionStatus.QUEUED, node_exec.input_data
                )
            )
            self.agent_server_client.send_execution_update(exec_update.model_dump())

        graph_exec = GraphExecution(
            graph_exec_id=graph_exec_id,
            start_node_execs=starting_node_execs,
        )
        self.queue.add(graph_exec)

        return {"id": graph_exec_id}
