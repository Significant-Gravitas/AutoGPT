import atexit
import logging
import multiprocessing
import os
import signal
import sys
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing.pool import AsyncResult, Pool
from typing import TYPE_CHECKING, Any, Generator, Optional, TypeVar, cast

from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
from redis.lock import Lock as RedisLock

from backend.blocks.io import AgentOutputBlock
from backend.data.model import (
    CredentialsMetaInput,
    GraphExecutionStats,
    NodeExecutionStats,
)
from backend.data.notifications import (
    AgentRunData,
    LowBalanceData,
    NotificationEventModel,
    NotificationType,
)
from backend.data.rabbitmq import SyncRabbitMQ
from backend.executor.utils import create_execution_queue_config
from backend.notifications.notifications import queue_notification
from backend.util.exceptions import InsufficientBalanceError

if TYPE_CHECKING:
    from backend.executor import DatabaseManagerClient

from autogpt_libs.utils.cache import thread_cached
from prometheus_client import Gauge, start_http_server

from backend.blocks.agent import AgentExecutorBlock
from backend.data import redis
from backend.data.block import BlockData, BlockInput, BlockSchema, get_block
from backend.data.credit import UsageTransactionMetadata
from backend.data.execution import (
    ExecutionQueue,
    ExecutionStatus,
    GraphExecution,
    GraphExecutionEntry,
    NodeExecutionEntry,
    NodeExecutionResult,
)
from backend.data.graph import Link, Node
from backend.executor.utils import (
    GRAPH_EXECUTION_CANCEL_QUEUE_NAME,
    GRAPH_EXECUTION_QUEUE_NAME,
    CancelExecutionEvent,
    block_usage_cost,
    execution_usage_cost,
    get_execution_event_bus,
    get_execution_queue,
    parse_execution_output,
    validate_exec,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util import json
from backend.util.decorator import error_logged, time_measured
from backend.util.file import clean_exec_files
from backend.util.logging import TruncatedLogger, configure_logging
from backend.util.process import AppProcess, set_service_name
from backend.util.retry import continuous_retry, func_retry
from backend.util.service import get_service_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

active_runs_gauge = Gauge(
    "execution_manager_active_runs", "Number of active graph runs"
)
pool_size_gauge = Gauge(
    "execution_manager_pool_size", "Maximum number of graph workers"
)
utilization_gauge = Gauge(
    "execution_manager_utilization_ratio",
    "Ratio of active graph runs to max graph workers",
)


class LogMetadata(TruncatedLogger):
    def __init__(
        self,
        user_id: str,
        graph_eid: str,
        graph_id: str,
        node_eid: str,
        node_id: str,
        block_name: str,
        max_length: int = 1000,
    ):
        metadata = {
            "component": "ExecutionManager",
            "user_id": user_id,
            "graph_eid": graph_eid,
            "graph_id": graph_id,
            "node_eid": node_eid,
            "node_id": node_id,
            "block_name": block_name,
        }
        prefix = f"[ExecutionManager|uid:{user_id}|gid:{graph_id}|nid:{node_id}]|geid:{graph_eid}|neid:{node_eid}|{block_name}]"
        super().__init__(
            logger,
            max_length=max_length,
            prefix=prefix,
            metadata=metadata,
        )


T = TypeVar("T")
ExecutionStream = Generator[NodeExecutionEntry, None, None]


def execute_node(
    db_client: "DatabaseManagerClient",
    creds_manager: IntegrationCredentialsManager,
    data: NodeExecutionEntry,
    execution_stats: NodeExecutionStats | None = None,
    node_credentials_input_map: Optional[
        dict[str, dict[str, CredentialsMetaInput]]
    ] = None,
) -> ExecutionStream:
    """
    Execute a node in the graph. This will trigger a block execution on a node,
    persist the execution result, and return the subsequent node to be executed.

    Args:
        db_client: The client to send execution updates to the server.
        creds_manager: The manager to acquire and release credentials.
        data: The execution data for executing the current node.
        execution_stats: The execution statistics to be updated.

    Returns:
        The subsequent node to be enqueued, or None if there is no subsequent node.
    """
    user_id = data.user_id
    graph_exec_id = data.graph_exec_id
    graph_id = data.graph_id
    node_exec_id = data.node_exec_id
    node_id = data.node_id

    def update_execution_status(status: ExecutionStatus) -> NodeExecutionResult:
        """Sets status and fetches+broadcasts the latest state of the node execution"""
        exec_update = db_client.update_node_execution_status(node_exec_id, status)
        send_execution_update(exec_update)
        return exec_update

    node = db_client.get_node(node_id)

    node_block = node.block

    def push_output(output_name: str, output_data: Any) -> None:
        db_client.upsert_execution_output(
            node_exec_id=node_exec_id,
            output_name=output_name,
            output_data=output_data,
        )

    log_metadata = LogMetadata(
        user_id=user_id,
        graph_eid=graph_exec_id,
        graph_id=graph_id,
        node_eid=node_exec_id,
        node_id=node_id,
        block_name=node_block.name,
    )

    # Sanity check: validate the execution input.
    input_data, error = validate_exec(node, data.inputs, resolve_input=False)
    if input_data is None:
        log_metadata.error(f"Skip execution, input validation error: {error}")
        push_output("error", error)
        update_execution_status(ExecutionStatus.FAILED)
        return

    # Re-shape the input data for agent block.
    # AgentExecutorBlock specially separate the node input_data & its input_default.
    if isinstance(node_block, AgentExecutorBlock):
        _input_data = AgentExecutorBlock.Input(**node.input_default)
        _input_data.inputs = input_data
        if node_credentials_input_map:
            _input_data.node_credentials_input_map = node_credentials_input_map
        input_data = _input_data.model_dump()
    data.inputs = input_data

    # Execute the node
    input_data_str = json.dumps(input_data)
    input_size = len(input_data_str)
    log_metadata.debug("Executed node with input", input=input_data_str)
    update_execution_status(ExecutionStatus.RUNNING)

    # Inject extra execution arguments for the blocks via kwargs
    extra_exec_kwargs: dict = {
        "graph_id": graph_id,
        "node_id": node_id,
        "graph_exec_id": graph_exec_id,
        "node_exec_id": node_exec_id,
        "user_id": user_id,
    }

    # Last-minute fetch credentials + acquire a system-wide read-write lock to prevent
    # changes during execution. ⚠️ This means a set of credentials can only be used by
    # one (running) block at a time; simultaneous execution of blocks using same
    # credentials is not supported.
    creds_lock = None
    input_model = cast(type[BlockSchema], node_block.input_schema)
    for field_name, input_type in input_model.get_credentials_fields().items():
        credentials_meta = input_type(**input_data[field_name])
        credentials, creds_lock = creds_manager.acquire(user_id, credentials_meta.id)
        extra_exec_kwargs[field_name] = credentials

    output_size = 0
    try:
        outputs: dict[str, Any] = {}
        for output_name, output_data in node_block.execute(
            input_data, **extra_exec_kwargs
        ):
            output_data = json.convert_pydantic_to_json(output_data)
            output_size += len(json.dumps(output_data))
            log_metadata.debug("Node produced output", **{output_name: output_data})
            push_output(output_name, output_data)
            outputs[output_name] = output_data
            for execution in _enqueue_next_nodes(
                db_client=db_client,
                node=node,
                output=(output_name, output_data),
                user_id=user_id,
                graph_exec_id=graph_exec_id,
                graph_id=graph_id,
                log_metadata=log_metadata,
                node_credentials_input_map=node_credentials_input_map,
            ):
                yield execution

        update_execution_status(ExecutionStatus.COMPLETED)

    except Exception as e:
        error_msg = str(e)
        push_output("error", error_msg)
        update_execution_status(ExecutionStatus.FAILED)

        for execution in _enqueue_next_nodes(
            db_client=db_client,
            node=node,
            output=("error", error_msg),
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            log_metadata=log_metadata,
            node_credentials_input_map=node_credentials_input_map,
        ):
            yield execution

        raise e
    finally:
        # Ensure credentials are released even if execution fails
        if creds_lock and creds_lock.locked() and creds_lock.owned():
            try:
                creds_lock.release()
            except Exception as e:
                log_metadata.error(f"Failed to release credentials lock: {e}")

        # Update execution stats
        if execution_stats is not None:
            execution_stats = execution_stats.model_copy(
                update=node_block.execution_stats.model_dump()
            )
            execution_stats.input_size = input_size
            execution_stats.output_size = output_size


def _enqueue_next_nodes(
    db_client: "DatabaseManagerClient",
    node: Node,
    output: BlockData,
    user_id: str,
    graph_exec_id: str,
    graph_id: str,
    log_metadata: LogMetadata,
    node_credentials_input_map: Optional[dict[str, dict[str, CredentialsMetaInput]]],
) -> list[NodeExecutionEntry]:
    def add_enqueued_execution(
        node_exec_id: str, node_id: str, block_id: str, data: BlockInput
    ) -> NodeExecutionEntry:
        exec_update = db_client.update_node_execution_status(
            node_exec_id, ExecutionStatus.QUEUED, data
        )
        send_execution_update(exec_update)
        return NodeExecutionEntry(
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            node_exec_id=node_exec_id,
            node_id=node_id,
            block_id=block_id,
            inputs=data,
        )

    def register_next_executions(node_link: Link) -> list[NodeExecutionEntry]:
        try:
            return _register_next_executions(node_link)
        except Exception as e:
            log_metadata.exception(f"Failed to register next executions: {e}")
            return []

    def _register_next_executions(node_link: Link) -> list[NodeExecutionEntry]:
        enqueued_executions = []
        next_output_name = node_link.source_name
        next_input_name = node_link.sink_name
        next_node_id = node_link.sink_id

        next_data = parse_execution_output(output, next_output_name)
        if next_data is None:
            return enqueued_executions

        next_node = db_client.get_node(next_node_id)

        # Multiple node can register the same next node, we need this to be atomic
        # To avoid same execution to be enqueued multiple times,
        # Or the same input to be consumed multiple times.
        with synchronized(f"upsert_input-{next_node_id}-{graph_exec_id}"):
            # Add output data to the earliest incomplete execution, or create a new one.
            next_node_exec_id, next_node_input = db_client.upsert_execution_input(
                node_id=next_node_id,
                graph_exec_id=graph_exec_id,
                input_name=next_input_name,
                input_data=next_data,
            )

            # Complete missing static input pins data using the last execution input.
            static_link_names = {
                link.sink_name
                for link in next_node.input_links
                if link.is_static and link.sink_name not in next_node_input
            }
            if static_link_names and (
                latest_execution := db_client.get_latest_node_execution(
                    next_node_id, graph_exec_id
                )
            ):
                for name in static_link_names:
                    next_node_input[name] = latest_execution.input_data.get(name)

            # Apply node credentials overrides
            node_credentials = None
            if node_credentials_input_map and (
                node_credentials := node_credentials_input_map.get(next_node.id)
            ):
                next_node_input.update(
                    {k: v.model_dump() for k, v in node_credentials.items()}
                )

            # Validate the input data for the next node.
            next_node_input, validation_msg = validate_exec(next_node, next_node_input)
            suffix = f"{next_output_name}>{next_input_name}~{next_node_exec_id}:{validation_msg}"

            # Incomplete input data, skip queueing the execution.
            if not next_node_input:
                log_metadata.warning(f"Skipped queueing {suffix}")
                return enqueued_executions

            # Input is complete, enqueue the execution.
            log_metadata.info(f"Enqueued {suffix}")
            enqueued_executions.append(
                add_enqueued_execution(
                    node_exec_id=next_node_exec_id,
                    node_id=next_node_id,
                    block_id=next_node.block_id,
                    data=next_node_input,
                )
            )

            # Next execution stops here if the link is not static.
            if not node_link.is_static:
                return enqueued_executions

            # If link is static, there could be some incomplete executions waiting for it.
            # Load and complete the input missing input data, and try to re-enqueue them.
            for iexec in db_client.get_node_executions(
                node_id=next_node_id,
                graph_exec_id=graph_exec_id,
                statuses=[ExecutionStatus.INCOMPLETE],
            ):
                idata = iexec.input_data
                ineid = iexec.node_exec_id

                static_link_names = {
                    link.sink_name
                    for link in next_node.input_links
                    if link.is_static and link.sink_name not in idata
                }
                for input_name in static_link_names:
                    idata[input_name] = next_node_input[input_name]

                # Apply node credentials overrides
                if node_credentials:
                    idata.update(
                        {k: v.model_dump() for k, v in node_credentials.items()}
                    )

                idata, msg = validate_exec(next_node, idata)
                suffix = f"{next_output_name}>{next_input_name}~{ineid}:{msg}"
                if not idata:
                    log_metadata.info(f"Enqueueing static-link skipped: {suffix}")
                    continue
                log_metadata.info(f"Enqueueing static-link execution {suffix}")
                enqueued_executions.append(
                    add_enqueued_execution(
                        node_exec_id=iexec.node_exec_id,
                        node_id=next_node_id,
                        block_id=next_node.block_id,
                        data=idata,
                    )
                )
            return enqueued_executions

    return [
        execution
        for link in node.output_links
        for execution in register_next_executions(link)
    ]


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
    @func_retry
    def on_node_executor_start(cls):
        configure_logging()
        set_service_name("NodeExecutor")
        redis.connect()
        cls.pid = os.getpid()
        cls.db_client = get_db_client()
        cls.creds_manager = IntegrationCredentialsManager()

        # Set up shutdown handlers
        cls.shutdown_lock = threading.Lock()
        atexit.register(cls.on_node_executor_stop)
        signal.signal(signal.SIGTERM, lambda _, __: cls.on_node_executor_sigterm())
        signal.signal(signal.SIGINT, lambda _, __: cls.on_node_executor_sigterm())

    @classmethod
    def on_node_executor_stop(cls, log=logger.info):
        if not cls.shutdown_lock.acquire(blocking=False):
            return  # already shutting down

        log(f"[on_node_executor_stop {cls.pid}] ⏳ Releasing locks...")
        cls.creds_manager.release_all_locks()
        log(f"[on_node_executor_stop {cls.pid}] ⏳ Disconnecting Redis...")
        redis.disconnect()
        log(f"[on_node_executor_stop {cls.pid}] ⏳ Disconnecting DB manager...")
        cls.db_client.close()
        log(f"[on_node_executor_stop {cls.pid}] ✅ Finished NodeExec cleanup")
        sys.exit(0)

    @classmethod
    def on_node_executor_sigterm(cls):
        llprint(f"[on_node_executor_sigterm {cls.pid}] ⚠️ NodeExec SIGTERM received")
        cls.on_node_executor_stop(log=llprint)

    @classmethod
    @error_logged
    def on_node_execution(
        cls,
        q: ExecutionQueue[NodeExecutionEntry],
        node_exec: NodeExecutionEntry,
        node_credentials_input_map: Optional[
            dict[str, dict[str, CredentialsMetaInput]]
        ] = None,
    ) -> NodeExecutionStats:
        log_metadata = LogMetadata(
            user_id=node_exec.user_id,
            graph_eid=node_exec.graph_exec_id,
            graph_id=node_exec.graph_id,
            node_eid=node_exec.node_exec_id,
            node_id=node_exec.node_id,
            block_name="-",
        )

        execution_stats = NodeExecutionStats()
        timing_info, _ = cls._on_node_execution(
            q, node_exec, log_metadata, execution_stats, node_credentials_input_map
        )
        execution_stats.walltime = timing_info.wall_time
        execution_stats.cputime = timing_info.cpu_time

        if isinstance(execution_stats.error, Exception):
            execution_stats.error = str(execution_stats.error)
        cls.db_client.update_node_execution_stats(
            node_exec.node_exec_id, execution_stats
        )
        return execution_stats

    @classmethod
    @time_measured
    def _on_node_execution(
        cls,
        q: ExecutionQueue[NodeExecutionEntry],
        node_exec: NodeExecutionEntry,
        log_metadata: LogMetadata,
        stats: NodeExecutionStats | None = None,
        node_credentials_input_map: Optional[
            dict[str, dict[str, CredentialsMetaInput]]
        ] = None,
    ):
        try:
            log_metadata.info(f"Start node execution {node_exec.node_exec_id}")
            for execution in execute_node(
                db_client=cls.db_client,
                creds_manager=cls.creds_manager,
                data=node_exec,
                execution_stats=stats,
                node_credentials_input_map=node_credentials_input_map,
            ):
                q.add(execution)
            log_metadata.info(f"Finished node execution {node_exec.node_exec_id}")
        except Exception as e:
            # Avoid user error being marked as an actual error.
            if isinstance(e, ValueError):
                log_metadata.info(
                    f"Failed node execution {node_exec.node_exec_id}: {e}"
                )
            else:
                log_metadata.exception(
                    f"Failed node execution {node_exec.node_exec_id}: {e}"
                )

            if stats is not None:
                stats.error = e

    @classmethod
    @func_retry
    def on_graph_executor_start(cls):
        configure_logging()
        set_service_name("GraphExecutor")

        cls.db_client = get_db_client()
        cls.pool_size = settings.config.num_node_workers
        cls.pid = os.getpid()
        cls._init_node_executor_pool()
        logger.info(f"GraphExec {cls.pid} started with {cls.pool_size} node workers")

    @classmethod
    def _init_node_executor_pool(cls):
        cls.executor = Pool(
            processes=cls.pool_size,
            initializer=cls.on_node_executor_start,
        )

    @classmethod
    @error_logged
    def on_graph_execution(
        cls, graph_exec: GraphExecutionEntry, cancel: threading.Event
    ):
        log_metadata = LogMetadata(
            user_id=graph_exec.user_id,
            graph_eid=graph_exec.graph_exec_id,
            graph_id=graph_exec.graph_id,
            node_id="*",
            node_eid="*",
            block_name="-",
        )

        exec_meta = cls.db_client.get_graph_execution_meta(
            user_id=graph_exec.user_id,
            execution_id=graph_exec.graph_exec_id,
        )
        if exec_meta is None:
            log_metadata.warning(
                f"Skipped graph execution #{graph_exec.graph_exec_id}, the graph execution is not found."
            )
            return

        if exec_meta.status == ExecutionStatus.QUEUED:
            log_metadata.info(f"⚙️ Starting graph execution #{graph_exec.graph_exec_id}")
            exec_meta.status = ExecutionStatus.RUNNING
            send_execution_update(
                cls.db_client.update_graph_execution_start_time(
                    graph_exec.graph_exec_id
                )
            )
        elif exec_meta.status == ExecutionStatus.RUNNING:
            log_metadata.info(
                f"⚙️ Graph execution #{graph_exec.graph_exec_id} is already running, continuing where it left off."
            )
        else:
            log_metadata.warning(
                f"Skipped graph execution {graph_exec.graph_exec_id}, the graph execution status is `{exec_meta.status}`."
            )
            return

        timing_info, (exec_stats, status, error) = cls._on_graph_execution(
            graph_exec=graph_exec,
            cancel=cancel,
            log_metadata=log_metadata,
            execution_stats=(
                exec_meta.stats.to_db() if exec_meta.stats else GraphExecutionStats()
            ),
        )
        exec_stats.walltime += timing_info.wall_time
        exec_stats.cputime += timing_info.cpu_time
        exec_stats.error = str(error) if error else exec_stats.error

        if graph_exec_result := cls.db_client.update_graph_execution_stats(
            graph_exec_id=graph_exec.graph_exec_id,
            status=status,
            stats=exec_stats,
        ):
            send_execution_update(graph_exec_result)

        cls._handle_agent_run_notif(graph_exec, exec_stats)

    @classmethod
    def _charge_usage(
        cls,
        node_exec: NodeExecutionEntry,
        execution_count: int,
        execution_stats: GraphExecutionStats,
    ):
        block = get_block(node_exec.block_id)
        if not block:
            logger.error(f"Block {node_exec.block_id} not found.")
            return

        cost, matching_filter = block_usage_cost(
            block=block, input_data=node_exec.inputs
        )
        if cost > 0:
            cls.db_client.spend_credits(
                user_id=node_exec.user_id,
                cost=cost,
                metadata=UsageTransactionMetadata(
                    graph_exec_id=node_exec.graph_exec_id,
                    graph_id=node_exec.graph_id,
                    node_exec_id=node_exec.node_exec_id,
                    node_id=node_exec.node_id,
                    block_id=node_exec.block_id,
                    block=block.name,
                    input=matching_filter,
                    reason=f"Ran block {node_exec.block_id} {block.name}",
                ),
            )
            execution_stats.cost += cost

        cost, usage_count = execution_usage_cost(execution_count)
        if cost > 0:
            cls.db_client.spend_credits(
                user_id=node_exec.user_id,
                cost=cost,
                metadata=UsageTransactionMetadata(
                    graph_exec_id=node_exec.graph_exec_id,
                    graph_id=node_exec.graph_id,
                    input={
                        "execution_count": usage_count,
                        "charge": "Execution Cost",
                    },
                    reason=f"Execution Cost for {usage_count} blocks of ex_id:{node_exec.graph_exec_id} g_id:{node_exec.graph_id}",
                ),
            )
            execution_stats.cost += cost

    @classmethod
    @time_measured
    def _on_graph_execution(
        cls,
        graph_exec: GraphExecutionEntry,
        cancel: threading.Event,
        log_metadata: LogMetadata,
        execution_stats: GraphExecutionStats,
    ) -> tuple[GraphExecutionStats, ExecutionStatus, Exception | None]:
        """
        Returns:
            dict: The execution statistics of the graph execution.
            ExecutionStatus: The final status of the graph execution.
            Exception | None: The error that occurred during the execution, if any.
        """
        execution_status = ExecutionStatus.RUNNING
        error = None
        finished = False

        def cancel_handler():
            nonlocal execution_status

            while not cancel.is_set():
                cancel.wait(1)
            if finished:
                return
            execution_status = ExecutionStatus.TERMINATED
            cls.executor.terminate()
            log_metadata.info(f"Terminated graph execution {graph_exec.graph_exec_id}")
            cls._init_node_executor_pool()

        cancel_thread = threading.Thread(target=cancel_handler)
        cancel_thread.start()

        try:
            if cls.db_client.get_credits(graph_exec.user_id) <= 0:
                raise InsufficientBalanceError(
                    user_id=graph_exec.user_id,
                    message="You have no credits left to run an agent.",
                    balance=0,
                    amount=1,
                )

            queue = ExecutionQueue[NodeExecutionEntry]()
            for node_exec in cls.db_client.get_node_executions(
                graph_exec.graph_exec_id,
                statuses=[ExecutionStatus.RUNNING, ExecutionStatus.QUEUED],
            ):
                queue.add(node_exec.to_node_execution_entry())

            running_executions: dict[str, AsyncResult] = {}

            def make_exec_callback(exec_data: NodeExecutionEntry):
                def callback(result: object):
                    running_executions.pop(exec_data.node_id)

                    if not isinstance(result, NodeExecutionStats):
                        return

                    nonlocal execution_stats
                    execution_stats.node_count += 1
                    execution_stats.nodes_cputime += result.cputime
                    execution_stats.nodes_walltime += result.walltime
                    if (err := result.error) and isinstance(err, Exception):
                        execution_stats.node_error_count += 1

                    if _graph_exec := cls.db_client.update_graph_execution_stats(
                        graph_exec_id=exec_data.graph_exec_id,
                        status=execution_status,
                        stats=execution_stats,
                    ):
                        send_execution_update(_graph_exec)
                    else:
                        logger.error(
                            "Callback for "
                            f"finished node execution #{exec_data.node_exec_id} "
                            "could not update execution stats "
                            f"for graph execution #{exec_data.graph_exec_id}; "
                            f"triggered while graph exec status = {execution_status}"
                        )

                return callback

            while not queue.empty():
                if cancel.is_set():
                    execution_status = ExecutionStatus.TERMINATED
                    return execution_stats, execution_status, error

                queued_node_exec = queue.get()

                # Avoid parallel execution of the same node.
                execution = running_executions.get(queued_node_exec.node_id)
                if execution and not execution.ready():
                    # TODO (performance improvement):
                    #   Wait for the completion of the same node execution is blocking.
                    #   To improve this we need a separate queue for each node.
                    #   Re-enqueueing the data back to the queue will disrupt the order.
                    execution.wait()

                log_metadata.debug(
                    f"Dispatching node execution {queued_node_exec.node_exec_id} "
                    f"for node {queued_node_exec.node_id}",
                )

                try:
                    cls._charge_usage(
                        node_exec=queued_node_exec,
                        execution_count=increment_execution_count(graph_exec.user_id),
                        execution_stats=execution_stats,
                    )
                except InsufficientBalanceError as error:
                    node_exec_id = queued_node_exec.node_exec_id
                    cls.db_client.upsert_execution_output(
                        node_exec_id=node_exec_id,
                        output_name="error",
                        output_data=str(error),
                    )

                    execution_status = ExecutionStatus.FAILED
                    exec_update = cls.db_client.update_node_execution_status(
                        node_exec_id, execution_status
                    )
                    send_execution_update(exec_update)

                    cls._handle_low_balance_notif(
                        graph_exec.user_id,
                        graph_exec.graph_id,
                        execution_stats,
                        error,
                    )
                    raise

                # Add credentials input overrides
                node_id = queued_node_exec.node_id
                if (node_creds_map := graph_exec.node_credentials_input_map) and (
                    node_field_creds_map := node_creds_map.get(node_id)
                ):
                    queued_node_exec.inputs.update(
                        {
                            field_name: creds_meta.model_dump()
                            for field_name, creds_meta in node_field_creds_map.items()
                        }
                    )

                # Initiate node execution
                running_executions[queued_node_exec.node_id] = cls.executor.apply_async(
                    cls.on_node_execution,
                    (queue, queued_node_exec, node_creds_map),
                    callback=make_exec_callback(queued_node_exec),
                )

                # Avoid terminating graph execution when some nodes are still running.
                while queue.empty() and running_executions:
                    log_metadata.debug(
                        f"Queue empty; running nodes: {list(running_executions.keys())}"
                    )
                    for node_id, execution in list(running_executions.items()):
                        if cancel.is_set():
                            execution_status = ExecutionStatus.TERMINATED
                            return execution_stats, execution_status, error

                        if not queue.empty():
                            break  # yield to parent loop to execute new queue items

                        log_metadata.debug(f"Waiting on execution of node {node_id}")
                        execution.wait(3)

            log_metadata.info(f"Finished graph execution {graph_exec.graph_exec_id}")
            execution_status = ExecutionStatus.COMPLETED

        except Exception as e:
            error = e
            log_metadata.error(
                f"Failed graph execution {graph_exec.graph_exec_id}: {error}"
            )
            execution_status = ExecutionStatus.FAILED

        finally:
            if not cancel.is_set():
                finished = True
                cancel.set()
            cancel_thread.join()
            clean_exec_files(graph_exec.graph_exec_id)
            return execution_stats, execution_status, error

    @classmethod
    def _handle_agent_run_notif(
        cls,
        graph_exec: GraphExecutionEntry,
        exec_stats: GraphExecutionStats,
    ):
        metadata = cls.db_client.get_graph_metadata(
            graph_exec.graph_id, graph_exec.graph_version
        )
        outputs = cls.db_client.get_node_executions(
            graph_exec.graph_exec_id,
            block_ids=[AgentOutputBlock().id],
        )

        named_outputs = [
            {
                key: value[0] if key == "name" else value
                for key, value in output.output_data.items()
            }
            for output in outputs
        ]

        queue_notification(
            NotificationEventModel(
                user_id=graph_exec.user_id,
                type=NotificationType.AGENT_RUN,
                data=AgentRunData(
                    outputs=named_outputs,
                    agent_name=metadata.name if metadata else "Unknown Agent",
                    credits_used=exec_stats.cost,
                    execution_time=exec_stats.walltime,
                    graph_id=graph_exec.graph_id,
                    node_count=exec_stats.node_count,
                ),
            )
        )

    @classmethod
    def _handle_low_balance_notif(
        cls,
        user_id: str,
        graph_id: str,
        exec_stats: GraphExecutionStats,
        e: InsufficientBalanceError,
    ):
        shortfall = e.balance - e.amount
        metadata = cls.db_client.get_graph_metadata(graph_id)
        base_url = (
            settings.config.frontend_base_url or settings.config.platform_base_url
        )
        queue_notification(
            NotificationEventModel(
                user_id=user_id,
                type=NotificationType.LOW_BALANCE,
                data=LowBalanceData(
                    current_balance=exec_stats.cost,
                    billing_page_link=f"{base_url}/profile/credits",
                    shortfall=shortfall,
                    agent_name=metadata.name if metadata else "Unknown Agent",
                ),
            )
        )


class ExecutionManager(AppProcess):
    def __init__(self):
        super().__init__()
        self.pool_size = settings.config.num_graph_workers
        self.running = True
        self.active_graph_runs: dict[str, tuple[Future, threading.Event]] = {}

    def run(self):
        pool_size_gauge.set(self.pool_size)
        active_runs_gauge.set(0)
        utilization_gauge.set(0)

        self.metrics_server = threading.Thread(
            target=start_http_server,
            args=(settings.config.execution_manager_port,),
            daemon=True,
        )
        self.metrics_server.start()
        logger.info(f"[{self.service_name}] Starting execution manager...")
        self._run()

    def _run(self):
        logger.info(f"[{self.service_name}] ⏳ Spawn max-{self.pool_size} workers...")
        self.executor = ProcessPoolExecutor(
            max_workers=self.pool_size,
            initializer=Executor.on_graph_executor_start,
        )

        logger.info(f"[{self.service_name}] ⏳ Connecting to Redis...")
        redis.connect()

        threading.Thread(
            target=lambda: self._consume_execution_cancel(),
            daemon=True,
        ).start()

        self._consume_execution_run()

    @continuous_retry()
    def _consume_execution_cancel(self):
        cancel_client = SyncRabbitMQ(create_execution_queue_config())
        cancel_client.connect()
        cancel_channel = cancel_client.get_channel()
        logger.info(f"[{self.service_name}] ⏳ Starting cancel message consumer...")
        cancel_channel.basic_consume(
            queue=GRAPH_EXECUTION_CANCEL_QUEUE_NAME,
            on_message_callback=self._handle_cancel_message,
            auto_ack=True,
        )
        cancel_channel.start_consuming()
        raise RuntimeError(f"❌ cancel message consumer is stopped: {cancel_channel}")

    @continuous_retry()
    def _consume_execution_run(self):
        run_client = SyncRabbitMQ(create_execution_queue_config())
        run_client.connect()
        run_channel = run_client.get_channel()
        run_channel.basic_qos(prefetch_count=self.pool_size)
        run_channel.basic_consume(
            queue=GRAPH_EXECUTION_QUEUE_NAME,
            on_message_callback=self._handle_run_message,
            auto_ack=False,
        )
        logger.info(f"[{self.service_name}] ⏳ Starting to consume run messages...")
        run_channel.start_consuming()
        raise RuntimeError(f"❌ run message consumer is stopped: {run_channel}")

    def _handle_cancel_message(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ):
        """
        Called whenever we receive a CANCEL message from the queue.
        (With auto_ack=True, message is considered 'acked' automatically.)
        """
        try:
            request = CancelExecutionEvent.model_validate_json(body)
            graph_exec_id = request.graph_exec_id
            if not graph_exec_id:
                logger.warning(
                    f"[{self.service_name}] Cancel message missing 'graph_exec_id'"
                )
                return
            if graph_exec_id not in self.active_graph_runs:
                logger.debug(
                    f"[{self.service_name}] Cancel received for {graph_exec_id} but not active."
                )
                return

            _, cancel_event = self.active_graph_runs[graph_exec_id]
            logger.info(f"[{self.service_name}] Received cancel for {graph_exec_id}")
            if not cancel_event.is_set():
                cancel_event.set()
            else:
                logger.debug(
                    f"[{self.service_name}] Cancel already set for {graph_exec_id}"
                )

        except Exception as e:
            logger.exception(f"Error handling cancel message: {e}")

    def _handle_run_message(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ):
        delivery_tag = method.delivery_tag
        try:
            graph_exec_entry = GraphExecutionEntry.model_validate_json(body)
        except Exception as e:
            logger.error(f"[{self.service_name}] Could not parse run message: {e}")
            channel.basic_nack(delivery_tag, requeue=False)
            return

        graph_exec_id = graph_exec_entry.graph_exec_id
        logger.info(
            f"[{self.service_name}] Received RUN for graph_exec_id={graph_exec_id}"
        )
        if graph_exec_id in self.active_graph_runs:
            logger.warning(
                f"[{self.service_name}] Graph {graph_exec_id} already running; rejecting duplicate run."
            )
            channel.basic_nack(delivery_tag, requeue=False)
            return

        cancel_event = multiprocessing.Manager().Event()
        future = self.executor.submit(
            Executor.on_graph_execution, graph_exec_entry, cancel_event
        )
        self.active_graph_runs[graph_exec_id] = (future, cancel_event)
        active_runs_gauge.set(len(self.active_graph_runs))
        utilization_gauge.set(len(self.active_graph_runs) / self.pool_size)

        def _on_run_done(f: Future):
            logger.info(f"[{self.service_name}] Run completed for {graph_exec_id}")
            try:
                self.active_graph_runs.pop(graph_exec_id, None)
                active_runs_gauge.set(len(self.active_graph_runs))
                utilization_gauge.set(len(self.active_graph_runs) / self.pool_size)
                if f.exception():
                    logger.error(
                        f"[{self.service_name}] Execution for {graph_exec_id} failed: {f.exception()}"
                    )
                    channel.connection.add_callback_threadsafe(
                        lambda: channel.basic_nack(delivery_tag, requeue=False)
                    )
                else:
                    channel.connection.add_callback_threadsafe(
                        lambda: channel.basic_ack(delivery_tag)
                    )
            except Exception as e:
                logger.error(f"[{self.service_name}] Error acknowledging message: {e}")

        future.add_done_callback(_on_run_done)

    def cleanup(self):
        super().cleanup()
        self._on_cleanup()

    def _on_cleanup(self, log=logger.info):
        prefix = f"[{self.service_name}][on_graph_executor_stop {os.getpid()}]"
        log(f"{prefix} ⏳ Shutting down service loop...")
        self.running = False

        log(f"{prefix} ⏳ Shutting down RabbitMQ channel...")
        get_execution_queue().get_channel().stop_consuming()

        if hasattr(self, "executor"):
            log(f"{prefix} ⏳ Shutting down GraphExec pool...")
            self.executor.shutdown(cancel_futures=True, wait=False)

        log(f"{prefix} ⏳ Disconnecting Redis...")
        redis.disconnect()

        log(f"{prefix} ✅ Finished GraphExec cleanup")
        sys.exit(0)


# ------- UTILITIES ------- #


@thread_cached
def get_db_client() -> "DatabaseManagerClient":
    from backend.executor import DatabaseManagerClient

    # Disable health check for the service client to avoid breaking process initializer.
    return get_service_client(DatabaseManagerClient, health_check=False)


def send_execution_update(entry: GraphExecution | NodeExecutionResult | None):
    if entry is None:
        return
    return get_execution_event_bus().publish(entry)


@contextmanager
def synchronized(key: str, timeout: int = 60):
    lock: RedisLock = redis.get_redis().lock(f"lock:{key}", timeout=timeout)
    try:
        lock.acquire()
        yield
    finally:
        if lock.locked() and lock.owned():
            lock.release()


def increment_execution_count(user_id: str) -> int:
    """
    Increment the execution count for a given user,
    this will be used to charge the user for the execution cost.
    """
    r = redis.get_redis()
    k = f"uec:{user_id}"  # User Execution Count global key
    counter = cast(int, r.incr(k))
    if counter == 1:
        r.expire(k, settings.config.execution_counter_expiration_time)
    return counter


def llprint(message: str):
    """
    Low-level print/log helper function for use in signal handlers.
    Regular log/print statements are not allowed in signal handlers.
    """
    os.write(sys.stdout.fileno(), (message + "\n").encode())
