import asyncio
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
from prometheus_client import Gauge, start_http_server
from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.io import AgentOutputBlock
from backend.data import redis_client as redis
from backend.data.block import (
    BlockInput,
    BlockOutput,
    BlockOutputEntry,
    BlockSchema,
    get_block,
)
from backend.data.credit import UsageTransactionMetadata
from backend.data.dynamic_fields import parse_execution_output
from backend.data.execution import (
    ExecutionQueue,
    ExecutionStatus,
    GraphExecution,
    GraphExecutionEntry,
    NodeExecutionEntry,
    NodeExecutionResult,
    NodesInputMasks,
    UserContext,
)
from backend.data.graph import Link, Node
from backend.data.model import GraphExecutionStats, NodeExecutionStats
from backend.data.notifications import (
    AgentRunData,
    LowBalanceData,
    NotificationEventModel,
    NotificationType,
    ZeroBalanceData,
)
from backend.data.rabbitmq import SyncRabbitMQ
from backend.executor.activity_status_generator import (
    generate_activity_status_for_execution,
)
from backend.executor.utils import (
    GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
    GRAPH_EXECUTION_CANCEL_QUEUE_NAME,
    GRAPH_EXECUTION_QUEUE_NAME,
    CancelExecutionEvent,
    ExecutionOutputEntry,
    LogMetadata,
    NodeExecutionProgress,
    block_usage_cost,
    create_execution_queue_config,
    execution_usage_cost,
    validate_exec,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.notifications.notifications import queue_notification
from backend.server.v2.AutoMod.manager import automod_manager
from backend.util import json
from backend.util.clients import (
    get_async_execution_event_bus,
    get_database_manager_async_client,
    get_database_manager_client,
    get_execution_event_bus,
    get_notification_manager_client,
)
from backend.util.decorator import (
    async_error_logged,
    async_time_measured,
    error_logged,
    time_measured,
)
from backend.util.exceptions import InsufficientBalanceError, ModerationError
from backend.util.file import clean_exec_files
from backend.util.logging import TruncatedLogger, configure_logging
from backend.util.metrics import DiscordChannel
from backend.util.process import AppProcess, set_service_name
from backend.util.retry import continuous_retry, func_retry
from backend.util.settings import Settings

from .cluster_lock import ClusterLock

if TYPE_CHECKING:
    from backend.executor import DatabaseManagerAsyncClient, DatabaseManagerClient


_logger = logging.getLogger(__name__)
logger = TruncatedLogger(_logger, prefix="[GraphExecutor]")
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


# Thread-local storage for ExecutionProcessor instances
_tls = threading.local()


def init_worker():
    """Initialize ExecutionProcessor instance in thread-local storage"""
    _tls.processor = ExecutionProcessor()
    _tls.processor.on_graph_executor_start()


def execute_graph(
    graph_exec_entry: "GraphExecutionEntry",
    cancel_event: threading.Event,
    cluster_lock: ClusterLock,
):
    """Execute graph using thread-local ExecutionProcessor instance"""
    return _tls.processor.on_graph_execution(
        graph_exec_entry, cancel_event, cluster_lock
    )


T = TypeVar("T")


async def execute_node(
    node: Node,
    creds_manager: IntegrationCredentialsManager,
    data: NodeExecutionEntry,
    execution_stats: NodeExecutionStats | None = None,
    nodes_input_masks: Optional[NodesInputMasks] = None,
) -> BlockOutput:
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
    node_block = node.block

    log_metadata = LogMetadata(
        logger=_logger,
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
        yield "error", error
        return

    # Re-shape the input data for agent block.
    # AgentExecutorBlock specially separate the node input_data & its input_default.
    if isinstance(node_block, AgentExecutorBlock):
        _input_data = AgentExecutorBlock.Input(**node.input_default)
        _input_data.inputs = input_data
        if nodes_input_masks:
            _input_data.nodes_input_masks = nodes_input_masks
        input_data = _input_data.model_dump()
    data.inputs = input_data

    # Execute the node
    input_data_str = json.dumps(input_data)
    input_size = len(input_data_str)
    log_metadata.debug("Executed node with input", input=input_data_str)

    # Inject extra execution arguments for the blocks via kwargs
    extra_exec_kwargs: dict = {
        "graph_id": graph_id,
        "node_id": node_id,
        "graph_exec_id": graph_exec_id,
        "node_exec_id": node_exec_id,
        "user_id": user_id,
    }

    # Add user context from NodeExecutionEntry
    extra_exec_kwargs["user_context"] = data.user_context

    # Last-minute fetch credentials + acquire a system-wide read-write lock to prevent
    # changes during execution. ⚠️ This means a set of credentials can only be used by
    # one (running) block at a time; simultaneous execution of blocks using same
    # credentials is not supported.
    creds_lock = None
    input_model = cast(type[BlockSchema], node_block.input_schema)
    for field_name, input_type in input_model.get_credentials_fields().items():
        credentials_meta = input_type(**input_data[field_name])
        credentials, creds_lock = await creds_manager.acquire(
            user_id, credentials_meta.id
        )
        extra_exec_kwargs[field_name] = credentials

    output_size = 0
    try:
        async for output_name, output_data in node_block.execute(
            input_data, **extra_exec_kwargs
        ):
            output_data = json.convert_pydantic_to_json(output_data)
            output_size += len(json.dumps(output_data))
            log_metadata.debug("Node produced output", **{output_name: output_data})
            yield output_name, output_data
    finally:
        # Ensure credentials are released even if execution fails
        if creds_lock and (await creds_lock.locked()) and (await creds_lock.owned()):
            try:
                await creds_lock.release()
            except Exception as e:
                log_metadata.error(f"Failed to release credentials lock: {e}")

        # Update execution stats
        if execution_stats is not None:
            execution_stats += node_block.execution_stats
            execution_stats.input_size = input_size
            execution_stats.output_size = output_size


async def _enqueue_next_nodes(
    db_client: "DatabaseManagerAsyncClient",
    node: Node,
    output: BlockOutputEntry,
    user_id: str,
    graph_exec_id: str,
    graph_id: str,
    log_metadata: LogMetadata,
    nodes_input_masks: Optional[NodesInputMasks],
    user_context: UserContext,
) -> list[NodeExecutionEntry]:
    async def add_enqueued_execution(
        node_exec_id: str, node_id: str, block_id: str, data: BlockInput
    ) -> NodeExecutionEntry:
        await async_update_node_execution_status(
            db_client=db_client,
            exec_id=node_exec_id,
            status=ExecutionStatus.QUEUED,
            execution_data=data,
        )
        return NodeExecutionEntry(
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            node_exec_id=node_exec_id,
            node_id=node_id,
            block_id=block_id,
            inputs=data,
            user_context=user_context,
        )

    async def register_next_executions(node_link: Link) -> list[NodeExecutionEntry]:
        try:
            return await _register_next_executions(node_link)
        except Exception as e:
            log_metadata.exception(f"Failed to register next executions: {e}")
            return []

    async def _register_next_executions(node_link: Link) -> list[NodeExecutionEntry]:
        enqueued_executions = []
        next_output_name = node_link.source_name
        next_input_name = node_link.sink_name
        next_node_id = node_link.sink_id

        output_name, _ = output
        next_data = parse_execution_output(output, next_output_name)
        if next_data is None and output_name != next_output_name:
            return enqueued_executions
        next_node = await db_client.get_node(next_node_id)

        # Multiple node can register the same next node, we need this to be atomic
        # To avoid same execution to be enqueued multiple times,
        # Or the same input to be consumed multiple times.
        async with synchronized(f"upsert_input-{next_node_id}-{graph_exec_id}"):
            # Add output data to the earliest incomplete execution, or create a new one.
            next_node_exec_id, next_node_input = await db_client.upsert_execution_input(
                node_id=next_node_id,
                graph_exec_id=graph_exec_id,
                input_name=next_input_name,
                input_data=next_data,
            )
            await async_update_node_execution_status(
                db_client=db_client,
                exec_id=next_node_exec_id,
                status=ExecutionStatus.INCOMPLETE,
            )

            # Complete missing static input pins data using the last execution input.
            static_link_names = {
                link.sink_name
                for link in next_node.input_links
                if link.is_static and link.sink_name not in next_node_input
            }
            if static_link_names and (
                latest_execution := await db_client.get_latest_node_execution(
                    next_node_id, graph_exec_id
                )
            ):
                for name in static_link_names:
                    next_node_input[name] = latest_execution.input_data.get(name)

            # Apply node input overrides
            node_input_mask = None
            if nodes_input_masks and (
                node_input_mask := nodes_input_masks.get(next_node.id)
            ):
                next_node_input.update(node_input_mask)

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
                await add_enqueued_execution(
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
            for iexec in await db_client.get_node_executions(
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

                # Apply node input overrides
                if node_input_mask:
                    idata.update(node_input_mask)

                idata, msg = validate_exec(next_node, idata)
                suffix = f"{next_output_name}>{next_input_name}~{ineid}:{msg}"
                if not idata:
                    log_metadata.info(f"Enqueueing static-link skipped: {suffix}")
                    continue
                log_metadata.info(f"Enqueueing static-link execution {suffix}")
                enqueued_executions.append(
                    await add_enqueued_execution(
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
        for execution in await register_next_executions(link)
    ]


class ExecutionProcessor:
    """
    This class contains event handlers for the process pool executor events.

    The main events are:
        on_graph_executor_start: Initialize the process that executes the graph.
        on_graph_execution: Execution logic for a graph.
        on_node_execution: Execution logic for a node.

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

    @async_error_logged(swallow=True)
    async def on_node_execution(
        self,
        node_exec: NodeExecutionEntry,
        node_exec_progress: NodeExecutionProgress,
        nodes_input_masks: Optional[NodesInputMasks],
        graph_stats_pair: tuple[GraphExecutionStats, threading.Lock],
    ) -> NodeExecutionStats:
        log_metadata = LogMetadata(
            logger=_logger,
            user_id=node_exec.user_id,
            graph_eid=node_exec.graph_exec_id,
            graph_id=node_exec.graph_id,
            node_eid=node_exec.node_exec_id,
            node_id=node_exec.node_id,
            block_name=b.name if (b := get_block(node_exec.block_id)) else "-",
        )
        db_client = get_db_async_client()
        node = await db_client.get_node(node_exec.node_id)
        execution_stats = NodeExecutionStats()

        timing_info, status = await self._on_node_execution(
            node=node,
            node_exec=node_exec,
            node_exec_progress=node_exec_progress,
            stats=execution_stats,
            db_client=db_client,
            log_metadata=log_metadata,
            nodes_input_masks=nodes_input_masks,
        )
        if isinstance(status, BaseException):
            raise status

        execution_stats.walltime = timing_info.wall_time
        execution_stats.cputime = timing_info.cpu_time

        graph_stats, graph_stats_lock = graph_stats_pair
        with graph_stats_lock:
            graph_stats.node_count += 1 + execution_stats.extra_steps
            graph_stats.nodes_cputime += execution_stats.cputime
            graph_stats.nodes_walltime += execution_stats.walltime
            graph_stats.cost += execution_stats.extra_cost
            if isinstance(execution_stats.error, Exception):
                graph_stats.node_error_count += 1

        node_error = execution_stats.error
        node_stats = execution_stats.model_dump()
        if node_error and not isinstance(node_error, str):
            node_stats["error"] = str(node_error) or node_stats.__class__.__name__

        await async_update_node_execution_status(
            db_client=db_client,
            exec_id=node_exec.node_exec_id,
            status=status,
            stats=node_stats,
        )
        await async_update_graph_execution_state(
            db_client=db_client,
            graph_exec_id=node_exec.graph_exec_id,
            stats=graph_stats,
        )

        return execution_stats

    @async_time_measured
    async def _on_node_execution(
        self,
        node: Node,
        node_exec: NodeExecutionEntry,
        node_exec_progress: NodeExecutionProgress,
        stats: NodeExecutionStats,
        db_client: "DatabaseManagerAsyncClient",
        log_metadata: LogMetadata,
        nodes_input_masks: Optional[NodesInputMasks] = None,
    ) -> ExecutionStatus:
        status = ExecutionStatus.RUNNING

        async def persist_output(output_name: str, output_data: Any) -> None:
            await db_client.upsert_execution_output(
                node_exec_id=node_exec.node_exec_id,
                output_name=output_name,
                output_data=output_data,
            )
            if exec_update := await db_client.get_node_execution(
                node_exec.node_exec_id
            ):
                await send_async_execution_update(exec_update)

            node_exec_progress.add_output(
                ExecutionOutputEntry(
                    node=node,
                    node_exec_id=node_exec.node_exec_id,
                    data=(output_name, output_data),
                )
            )

        try:
            log_metadata.info(f"Start node execution {node_exec.node_exec_id}")
            await async_update_node_execution_status(
                db_client=db_client,
                exec_id=node_exec.node_exec_id,
                status=ExecutionStatus.RUNNING,
            )

            async for output_name, output_data in execute_node(
                node=node,
                creds_manager=self.creds_manager,
                data=node_exec,
                execution_stats=stats,
                nodes_input_masks=nodes_input_masks,
            ):
                await persist_output(output_name, output_data)

            log_metadata.info(f"Finished node execution {node_exec.node_exec_id}")
            status = ExecutionStatus.COMPLETED

        except BaseException as e:
            stats.error = e

            if isinstance(e, ValueError):
                # Avoid user error being marked as an actual error.
                log_metadata.info(
                    f"Expected failure on node execution {node_exec.node_exec_id}: {type(e).__name__} - {e}"
                )
                status = ExecutionStatus.FAILED
            elif isinstance(e, Exception):
                # If the exception is not a ValueError, it is unexpected.
                log_metadata.exception(
                    f"Unexpected failure on node execution {node_exec.node_exec_id}: {type(e).__name__} - {e}"
                )
                status = ExecutionStatus.FAILED
            else:
                # CancelledError or SystemExit
                log_metadata.warning(
                    f"Interruption error on node execution {node_exec.node_exec_id}: {type(e).__name__}"
                )
                status = ExecutionStatus.TERMINATED

        finally:
            if status == ExecutionStatus.FAILED and stats.error is not None:
                await persist_output(
                    "error", str(stats.error) or type(stats.error).__name__
                )

        return status

    @func_retry
    def on_graph_executor_start(self):
        configure_logging()
        set_service_name("GraphExecutor")
        self.tid = threading.get_ident()
        self.creds_manager = IntegrationCredentialsManager()
        self.node_execution_loop = asyncio.new_event_loop()
        self.node_evaluation_loop = asyncio.new_event_loop()
        self.node_execution_thread = threading.Thread(
            target=self.node_execution_loop.run_forever, daemon=True
        )
        self.node_evaluation_thread = threading.Thread(
            target=self.node_evaluation_loop.run_forever, daemon=True
        )
        self.node_execution_thread.start()
        self.node_evaluation_thread.start()
        logger.info(f"[GraphExecutor] {self.tid} started")

    @error_logged(swallow=False)
    def on_graph_execution(
        self,
        graph_exec: GraphExecutionEntry,
        cancel: threading.Event,
        cluster_lock: ClusterLock,
    ):
        log_metadata = LogMetadata(
            logger=_logger,
            user_id=graph_exec.user_id,
            graph_eid=graph_exec.graph_exec_id,
            graph_id=graph_exec.graph_id,
            node_id="*",
            node_eid="*",
            block_name="-",
        )
        db_client = get_db_client()

        exec_meta = db_client.get_graph_execution_meta(
            user_id=graph_exec.user_id,
            execution_id=graph_exec.graph_exec_id,
        )
        if exec_meta is None:
            log_metadata.warning(
                f"Skipped graph execution #{graph_exec.graph_exec_id}, the graph execution is not found."
            )
            return

        if exec_meta.status in [ExecutionStatus.QUEUED, ExecutionStatus.INCOMPLETE]:
            log_metadata.info(f"⚙️ Starting graph execution #{graph_exec.graph_exec_id}")
            exec_meta.status = ExecutionStatus.RUNNING
            send_execution_update(
                db_client.update_graph_execution_start_time(graph_exec.graph_exec_id)
            )
        elif exec_meta.status == ExecutionStatus.RUNNING:
            log_metadata.info(
                f"⚙️ Graph execution #{graph_exec.graph_exec_id} is already running, continuing where it left off."
            )
        elif exec_meta.status == ExecutionStatus.FAILED:
            exec_meta.status = ExecutionStatus.RUNNING
            log_metadata.info(
                f"⚙️ Graph execution #{graph_exec.graph_exec_id} was disturbed, continuing where it left off."
            )
            update_graph_execution_state(
                db_client=db_client,
                graph_exec_id=graph_exec.graph_exec_id,
                status=ExecutionStatus.RUNNING,
            )
        else:
            log_metadata.warning(
                f"Skipped graph execution {graph_exec.graph_exec_id}, the graph execution status is `{exec_meta.status}`."
            )
            return

        if exec_meta.stats is None:
            exec_stats = GraphExecutionStats()
        else:
            exec_stats = exec_meta.stats.to_db()

        timing_info, status = self._on_graph_execution(
            graph_exec=graph_exec,
            cancel=cancel,
            log_metadata=log_metadata,
            execution_stats=exec_stats,
            cluster_lock=cluster_lock,
        )
        exec_stats.walltime += timing_info.wall_time
        exec_stats.cputime += timing_info.cpu_time

        try:
            # Failure handling
            if isinstance(status, BaseException):
                raise status
            exec_meta.status = status

            # Activity status handling
            activity_status = asyncio.run_coroutine_threadsafe(
                generate_activity_status_for_execution(
                    graph_exec_id=graph_exec.graph_exec_id,
                    graph_id=graph_exec.graph_id,
                    graph_version=graph_exec.graph_version,
                    execution_stats=exec_stats,
                    db_client=get_db_async_client(),
                    user_id=graph_exec.user_id,
                    execution_status=status,
                ),
                self.node_execution_loop,
            ).result(timeout=60.0)
            if activity_status is not None:
                exec_stats.activity_status = activity_status
                log_metadata.info(f"Generated activity status: {activity_status}")
            else:
                log_metadata.debug(
                    "Activity status generation disabled, not setting field"
                )

            # Communication handling
            self._handle_agent_run_notif(db_client, graph_exec, exec_stats)

        finally:
            update_graph_execution_state(
                db_client=db_client,
                graph_exec_id=graph_exec.graph_exec_id,
                status=exec_meta.status,
                stats=exec_stats,
            )

    def _charge_usage(
        self,
        node_exec: NodeExecutionEntry,
        execution_count: int,
    ) -> tuple[int, int]:
        total_cost = 0
        remaining_balance = 0
        db_client = get_db_client()
        block = get_block(node_exec.block_id)
        if not block:
            logger.error(f"Block {node_exec.block_id} not found.")
            return total_cost, 0

        cost, matching_filter = block_usage_cost(
            block=block, input_data=node_exec.inputs
        )
        if cost > 0:
            remaining_balance = db_client.spend_credits(
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
            total_cost += cost

        cost, usage_count = execution_usage_cost(execution_count)
        if cost > 0:
            remaining_balance = db_client.spend_credits(
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
            total_cost += cost

        return total_cost, remaining_balance

    @time_measured
    def _on_graph_execution(
        self,
        graph_exec: GraphExecutionEntry,
        cancel: threading.Event,
        log_metadata: LogMetadata,
        execution_stats: GraphExecutionStats,
        cluster_lock: ClusterLock,
    ) -> ExecutionStatus:
        """
        Returns:
            dict: The execution statistics of the graph execution.
            ExecutionStatus: The final status of the graph execution.
            Exception | None: The error that occurred during the execution, if any.
        """
        execution_status: ExecutionStatus = ExecutionStatus.RUNNING
        error: Exception | None = None
        db_client = get_db_client()
        execution_stats_lock = threading.Lock()

        # State holders ----------------------------------------------------
        running_node_execution: dict[str, NodeExecutionProgress] = defaultdict(
            NodeExecutionProgress
        )
        running_node_evaluation: dict[str, Future] = {}
        execution_queue = ExecutionQueue[NodeExecutionEntry]()

        try:
            if db_client.get_credits(graph_exec.user_id) <= 0:
                raise InsufficientBalanceError(
                    user_id=graph_exec.user_id,
                    message="You have no credits left to run an agent.",
                    balance=0,
                    amount=1,
                )

            # Input moderation
            try:
                if moderation_error := asyncio.run_coroutine_threadsafe(
                    automod_manager.moderate_graph_execution_inputs(
                        db_client=get_db_async_client(),
                        graph_exec=graph_exec,
                    ),
                    self.node_evaluation_loop,
                ).result(timeout=30.0):
                    raise moderation_error
            except asyncio.TimeoutError:
                log_metadata.warning(
                    f"Input moderation timed out for graph execution {graph_exec.graph_exec_id}, bypassing moderation and continuing execution"
                )
                # Continue execution without moderation

            # ------------------------------------------------------------
            # Pre‑populate queue ---------------------------------------
            # ------------------------------------------------------------
            for node_exec in db_client.get_node_executions(
                graph_exec.graph_exec_id,
                statuses=[
                    ExecutionStatus.RUNNING,
                    ExecutionStatus.QUEUED,
                    ExecutionStatus.TERMINATED,
                ],
            ):
                node_entry = node_exec.to_node_execution_entry(graph_exec.user_context)
                execution_queue.add(node_entry)

            # ------------------------------------------------------------
            # Main dispatch / polling loop -----------------------------
            # ------------------------------------------------------------
            while not execution_queue.empty():
                if cancel.is_set():
                    break

                queued_node_exec = execution_queue.get()

                log_metadata.debug(
                    f"Dispatching node execution {queued_node_exec.node_exec_id} "
                    f"for node {queued_node_exec.node_id}",
                )

                # Charge usage (may raise) ------------------------------
                try:
                    cost, remaining_balance = self._charge_usage(
                        node_exec=queued_node_exec,
                        execution_count=increment_execution_count(graph_exec.user_id),
                    )
                    with execution_stats_lock:
                        execution_stats.cost += cost
                    # Check if we crossed the low balance threshold
                    self._handle_low_balance(
                        db_client=db_client,
                        user_id=graph_exec.user_id,
                        current_balance=remaining_balance,
                        transaction_cost=cost,
                    )
                except InsufficientBalanceError as balance_error:
                    error = balance_error  # Set error to trigger FAILED status
                    node_exec_id = queued_node_exec.node_exec_id
                    db_client.upsert_execution_output(
                        node_exec_id=node_exec_id,
                        output_name="error",
                        output_data=str(error),
                    )
                    update_node_execution_status(
                        db_client=db_client,
                        exec_id=node_exec_id,
                        status=ExecutionStatus.FAILED,
                    )

                    self._handle_insufficient_funds_notif(
                        db_client,
                        graph_exec.user_id,
                        graph_exec.graph_id,
                        error,
                    )
                    # Gracefully stop the execution loop
                    break

                # Add input overrides -----------------------------
                node_id = queued_node_exec.node_id
                if (nodes_input_masks := graph_exec.nodes_input_masks) and (
                    node_input_mask := nodes_input_masks.get(node_id)
                ):
                    queued_node_exec.inputs.update(node_input_mask)

                # Kick off async node execution -------------------------
                node_execution_task = asyncio.run_coroutine_threadsafe(
                    self.on_node_execution(
                        node_exec=queued_node_exec,
                        node_exec_progress=running_node_execution[node_id],
                        nodes_input_masks=nodes_input_masks,
                        graph_stats_pair=(
                            execution_stats,
                            execution_stats_lock,
                        ),
                    ),
                    self.node_execution_loop,
                )
                running_node_execution[node_id].add_task(
                    node_exec_id=queued_node_exec.node_exec_id,
                    task=node_execution_task,
                )

                # Poll until queue refills or all inflight work done ----
                while execution_queue.empty() and (
                    running_node_execution or running_node_evaluation
                ):
                    if cancel.is_set():
                        break

                    # --------------------------------------------------
                    # Handle inflight evaluations ---------------------
                    # --------------------------------------------------
                    node_output_found = False
                    for node_id, inflight_exec in list(running_node_execution.items()):
                        if cancel.is_set():
                            break

                        # node evaluation future -----------------
                        if inflight_eval := running_node_evaluation.get(node_id):
                            if not inflight_eval.done():
                                continue
                            try:
                                inflight_eval.result(timeout=0)
                                running_node_evaluation.pop(node_id)
                            except Exception as e:
                                log_metadata.error(f"Node eval #{node_id} failed: {e}")

                        # node execution future ---------------------------
                        if inflight_exec.is_done():
                            running_node_execution.pop(node_id)
                            continue

                        if output := inflight_exec.pop_output():
                            node_output_found = True
                            running_node_evaluation[node_id] = (
                                asyncio.run_coroutine_threadsafe(
                                    self._process_node_output(
                                        output=output,
                                        node_id=node_id,
                                        graph_exec=graph_exec,
                                        log_metadata=log_metadata,
                                        nodes_input_masks=nodes_input_masks,
                                        execution_queue=execution_queue,
                                    ),
                                    self.node_evaluation_loop,
                                )
                            )
                    if (
                        not node_output_found
                        and execution_queue.empty()
                        and (running_node_execution or running_node_evaluation)
                    ):
                        cluster_lock.refresh()
                        time.sleep(0.1)

            # loop done --------------------------------------------------

            # Output moderation
            try:
                if moderation_error := asyncio.run_coroutine_threadsafe(
                    automod_manager.moderate_graph_execution_outputs(
                        db_client=get_db_async_client(),
                        graph_exec_id=graph_exec.graph_exec_id,
                        user_id=graph_exec.user_id,
                        graph_id=graph_exec.graph_id,
                    ),
                    self.node_evaluation_loop,
                ).result(timeout=30.0):
                    raise moderation_error
            except asyncio.TimeoutError:
                log_metadata.warning(
                    f"Output moderation timed out for graph execution {graph_exec.graph_exec_id}, bypassing moderation and continuing execution"
                )
                # Continue execution without moderation

            # Determine final execution status based on whether there was an error or termination
            if cancel.is_set():
                execution_status = ExecutionStatus.TERMINATED
            elif error is not None:
                execution_status = ExecutionStatus.FAILED
            else:
                execution_status = ExecutionStatus.COMPLETED

            if error:
                execution_stats.error = str(error) or type(error).__name__

            return execution_status

        except BaseException as e:
            error = (
                e
                if isinstance(e, Exception)
                else Exception(f"{e.__class__.__name__}: {e}")
            )

            known_errors = (InsufficientBalanceError, ModerationError)
            if isinstance(error, known_errors):
                execution_stats.error = str(error)
                return ExecutionStatus.FAILED

            execution_status = ExecutionStatus.FAILED
            log_metadata.exception(
                f"Failed graph execution {graph_exec.graph_exec_id}: {error}"
            )
            raise

        finally:
            self._cleanup_graph_execution(
                execution_queue=execution_queue,
                running_node_execution=running_node_execution,
                running_node_evaluation=running_node_evaluation,
                execution_status=execution_status,
                error=error,
                graph_exec_id=graph_exec.graph_exec_id,
                log_metadata=log_metadata,
                db_client=db_client,
            )

    @error_logged(swallow=True)
    def _cleanup_graph_execution(
        self,
        execution_queue: ExecutionQueue[NodeExecutionEntry],
        running_node_execution: dict[str, "NodeExecutionProgress"],
        running_node_evaluation: dict[str, Future],
        execution_status: ExecutionStatus,
        error: Exception | None,
        graph_exec_id: str,
        log_metadata: LogMetadata,
        db_client: "DatabaseManagerClient",
    ) -> None:
        """
        Clean up running node executions and evaluations when graph execution ends.
        This method is decorated with @error_logged(swallow=True) to ensure cleanup
        never fails in the finally block.
        """
        # Cancel and wait for all node executions to complete
        for node_id, inflight_exec in running_node_execution.items():
            if inflight_exec.is_done():
                continue
            log_metadata.info(f"Stopping node execution {node_id}")
            inflight_exec.stop()

        for node_id, inflight_exec in running_node_execution.items():
            try:
                inflight_exec.wait_for_done(timeout=3600.0)
            except TimeoutError:
                log_metadata.exception(
                    f"Node execution #{node_id} did not stop in time, "
                    "it may be stuck or taking too long."
                )

        # Wait the remaining inflight evaluations to finish
        for node_id, inflight_eval in running_node_evaluation.items():
            try:
                inflight_eval.result(timeout=3600.0)
            except TimeoutError:
                log_metadata.exception(
                    f"Node evaluation #{node_id} did not stop in time, "
                    "it may be stuck or taking too long."
                )

        while queued_execution := execution_queue.get_or_none():
            update_node_execution_status(
                db_client=db_client,
                exec_id=queued_execution.node_exec_id,
                status=execution_status,
                stats={"error": str(error)} if error else None,
            )

        clean_exec_files(graph_exec_id)

    @async_error_logged(swallow=True)
    async def _process_node_output(
        self,
        output: ExecutionOutputEntry,
        node_id: str,
        graph_exec: GraphExecutionEntry,
        log_metadata: LogMetadata,
        nodes_input_masks: Optional[NodesInputMasks],
        execution_queue: ExecutionQueue[NodeExecutionEntry],
    ) -> None:
        """Process a node's output, update its status, and enqueue next nodes.

        Args:
            output: The execution output entry to process
            node_id: The ID of the node that produced the output
            graph_exec: The graph execution entry
            log_metadata: Logger metadata for consistent logging
            nodes_input_masks: Optional map of node input overrides
            execution_queue: Queue to add next executions to
        """
        db_client = get_db_async_client()

        log_metadata.debug(f"Enqueue nodes for {node_id}: {output}")

        for next_execution in await _enqueue_next_nodes(
            db_client=db_client,
            node=output.node,
            output=output.data,
            user_id=graph_exec.user_id,
            graph_exec_id=graph_exec.graph_exec_id,
            graph_id=graph_exec.graph_id,
            log_metadata=log_metadata,
            nodes_input_masks=nodes_input_masks,
            user_context=graph_exec.user_context,
        ):
            execution_queue.add(next_execution)

    def _handle_agent_run_notif(
        self,
        db_client: "DatabaseManagerClient",
        graph_exec: GraphExecutionEntry,
        exec_stats: GraphExecutionStats,
    ):
        metadata = db_client.get_graph_metadata(
            graph_exec.graph_id, graph_exec.graph_version
        )
        outputs = db_client.get_node_executions(
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

    def _handle_insufficient_funds_notif(
        self,
        db_client: "DatabaseManagerClient",
        user_id: str,
        graph_id: str,
        e: InsufficientBalanceError,
    ):
        shortfall = abs(e.amount) - e.balance
        metadata = db_client.get_graph_metadata(graph_id)
        base_url = (
            settings.config.frontend_base_url or settings.config.platform_base_url
        )

        queue_notification(
            NotificationEventModel(
                user_id=user_id,
                type=NotificationType.ZERO_BALANCE,
                data=ZeroBalanceData(
                    current_balance=e.balance,
                    billing_page_link=f"{base_url}/profile/credits",
                    shortfall=shortfall,
                    agent_name=metadata.name if metadata else "Unknown Agent",
                ),
            )
        )

        try:
            user_email = db_client.get_user_email_by_id(user_id)

            alert_message = (
                f"❌ **Insufficient Funds Alert**\n"
                f"User: {user_email or user_id}\n"
                f"Agent: {metadata.name if metadata else 'Unknown Agent'}\n"
                f"Current balance: ${e.balance/100:.2f}\n"
                f"Attempted cost: ${abs(e.amount)/100:.2f}\n"
                f"Shortfall: ${abs(shortfall)/100:.2f}\n"
                f"[View User Details]({base_url}/admin/spending?search={user_email})"
            )

            get_notification_manager_client().discord_system_alert(
                alert_message, DiscordChannel.PRODUCT
            )
        except Exception as alert_error:
            logger.error(
                f"Failed to send insufficient funds Discord alert: {alert_error}"
            )

    def _handle_low_balance(
        self,
        db_client: "DatabaseManagerClient",
        user_id: str,
        current_balance: int,
        transaction_cost: int,
    ):
        """Check and handle low balance scenarios after a transaction"""
        LOW_BALANCE_THRESHOLD = settings.config.low_balance_threshold

        balance_before = current_balance + transaction_cost

        if (
            current_balance < LOW_BALANCE_THRESHOLD
            and balance_before >= LOW_BALANCE_THRESHOLD
        ):
            base_url = (
                settings.config.frontend_base_url or settings.config.platform_base_url
            )
            queue_notification(
                NotificationEventModel(
                    user_id=user_id,
                    type=NotificationType.LOW_BALANCE,
                    data=LowBalanceData(
                        current_balance=current_balance,
                        billing_page_link=f"{base_url}/profile/credits",
                    ),
                )
            )

            try:
                user_email = db_client.get_user_email_by_id(user_id)
                alert_message = (
                    f"⚠️ **Low Balance Alert**\n"
                    f"User: {user_email or user_id}\n"
                    f"Balance dropped below ${LOW_BALANCE_THRESHOLD/100:.2f}\n"
                    f"Current balance: ${current_balance/100:.2f}\n"
                    f"Transaction cost: ${transaction_cost/100:.2f}\n"
                    f"[View User Details]({base_url}/admin/spending?search={user_email})"
                )
                get_notification_manager_client().discord_system_alert(
                    alert_message, DiscordChannel.PRODUCT
                )
            except Exception as e:
                logger.error(f"Failed to send low balance Discord alert: {e}")


class ExecutionManager(AppProcess):
    def __init__(self):
        super().__init__()
        self.pool_size = settings.config.num_graph_workers
        self.active_graph_runs: dict[str, tuple[Future, threading.Event]] = {}
        self.executor_id = str(uuid.uuid4())

        self._executor = None
        self._stop_consuming = None

        self._cancel_thread = None
        self._cancel_client = None
        self._run_thread = None
        self._run_client = None

        self._execution_locks = {}

    @property
    def cancel_thread(self) -> threading.Thread:
        if self._cancel_thread is None:
            self._cancel_thread = threading.Thread(
                target=lambda: self._consume_execution_cancel(),
                daemon=True,
            )
        return self._cancel_thread

    @property
    def run_thread(self) -> threading.Thread:
        if self._run_thread is None:
            self._run_thread = threading.Thread(
                target=lambda: self._consume_execution_run(),
                daemon=True,
            )
        return self._run_thread

    @property
    def stop_consuming(self) -> threading.Event:
        if self._stop_consuming is None:
            self._stop_consuming = threading.Event()
        return self._stop_consuming

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.pool_size,
                initializer=init_worker,
            )
        return self._executor

    @property
    def cancel_client(self) -> SyncRabbitMQ:
        if self._cancel_client is None:
            self._cancel_client = SyncRabbitMQ(create_execution_queue_config())
        return self._cancel_client

    @property
    def run_client(self) -> SyncRabbitMQ:
        if self._run_client is None:
            self._run_client = SyncRabbitMQ(create_execution_queue_config())
        return self._run_client

    def run(self):
        logger.info(f"[{self.service_name}] ⏳ Spawn max-{self.pool_size} workers...")

        pool_size_gauge.set(self.pool_size)
        self._update_prompt_metrics()
        start_http_server(settings.config.execution_manager_port)

        self.cancel_thread.start()
        self.run_thread.start()

        while True:
            time.sleep(1e5)

    @continuous_retry()
    def _consume_execution_cancel(self):
        if self.stop_consuming.is_set() and not self.active_graph_runs:
            logger.info(
                f"[{self.service_name}] Stop reconnecting cancel consumer since the service is cleaned up."
            )
            return

        # Check if channel is closed and force reconnection if needed
        if not self.cancel_client.is_ready:
            self.cancel_client.disconnect()
        self.cancel_client.connect()
        cancel_channel = self.cancel_client.get_channel()
        cancel_channel.basic_consume(
            queue=GRAPH_EXECUTION_CANCEL_QUEUE_NAME,
            on_message_callback=self._handle_cancel_message,
            auto_ack=True,
        )
        logger.info(f"[{self.service_name}] ⏳ Starting cancel message consumer...")
        cancel_channel.start_consuming()
        if not self.stop_consuming.is_set() or self.active_graph_runs:
            raise RuntimeError(
                f"[{self.service_name}] ❌ cancel message consumer is stopped: {cancel_channel}"
            )
        logger.info(
            f"[{self.service_name}] ✅ Cancel message consumer stopped gracefully"
        )

    @continuous_retry()
    def _consume_execution_run(self):
        # Long-running executions are handled by:
        # 1. Long consumer timeout (x-consumer-timeout) allows long running agent
        # 2. Enhanced connection settings (5 retries, 1s delay) for quick reconnection
        # 3. Process monitoring ensures failed executors release messages back to queue
        if self.stop_consuming.is_set():
            logger.info(
                f"[{self.service_name}] Stop reconnecting execution consumer since the service is cleaned up."
            )
            return

        # Check if channel is closed and force reconnection if needed
        if not self.run_client.is_ready:
            self.run_client.disconnect()
        self.run_client.connect()
        run_channel = self.run_client.get_channel()
        run_channel.basic_qos(prefetch_count=self.pool_size)

        # Configure consumer for long-running graph executions
        # auto_ack=False: Don't acknowledge messages until execution completes (prevents data loss)
        run_channel.basic_consume(
            queue=GRAPH_EXECUTION_QUEUE_NAME,
            on_message_callback=self._handle_run_message,
            auto_ack=False,
            consumer_tag="graph_execution_consumer",
        )
        run_channel.confirm_delivery()
        logger.info(f"[{self.service_name}] ⏳ Starting to consume run messages...")
        run_channel.start_consuming()
        if not self.stop_consuming.is_set():
            raise RuntimeError(
                f"[{self.service_name}] ❌ run message consumer is stopped: {run_channel}"
            )
        logger.info(f"[{self.service_name}] ✅ Run message consumer stopped gracefully")

    @error_logged(swallow=True)
    def _handle_cancel_message(
        self,
        _channel: BlockingChannel,
        _method: Basic.Deliver,
        _properties: BasicProperties,
        body: bytes,
    ):
        """
        Called whenever we receive a CANCEL message from the queue.
        (With auto_ack=True, message is considered 'acked' automatically.)
        """
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

    def _handle_run_message(
        self,
        _channel: BlockingChannel,
        method: Basic.Deliver,
        _properties: BasicProperties,
        body: bytes,
    ):
        delivery_tag = method.delivery_tag

        @func_retry
        def _ack_message(reject: bool, requeue: bool):
            """Acknowledge or reject the message based on execution status."""

            # Connection can be lost, so always get a fresh channel
            channel = self.run_client.get_channel()
            if reject:
                channel.connection.add_callback_threadsafe(
                    lambda: channel.basic_nack(delivery_tag, requeue=requeue)
                )
            else:
                channel.connection.add_callback_threadsafe(
                    lambda: channel.basic_ack(delivery_tag)
                )

        # Check if we're shutting down - reject new messages but keep connection alive
        if self.stop_consuming.is_set():
            logger.info(
                f"[{self.service_name}] Rejecting new execution during shutdown"
            )
            _ack_message(reject=True, requeue=True)
            return

        # Check if we can accept more runs
        self._cleanup_completed_runs()
        if len(self.active_graph_runs) >= self.pool_size:
            _ack_message(reject=True, requeue=True)
            return

        try:
            graph_exec_entry = GraphExecutionEntry.model_validate_json(body)
        except Exception as e:
            logger.error(
                f"[{self.service_name}] Could not parse run message: {e}, body={body}"
            )
            _ack_message(reject=True, requeue=False)
            return

        graph_exec_id = graph_exec_entry.graph_exec_id
        logger.info(
            f"[{self.service_name}] Received RUN for graph_exec_id={graph_exec_id}"
        )

        # Check for local duplicate execution first
        if graph_exec_id in self.active_graph_runs:
            logger.warning(
                f"[{self.service_name}] Graph {graph_exec_id} already running locally; rejecting duplicate."
            )
            _ack_message(reject=True, requeue=True)
            return

        # Try to acquire cluster-wide execution lock
        cluster_lock = ClusterLock(
            redis=redis.get_redis(),
            key=f"exec_lock:{graph_exec_id}",
            owner_id=self.executor_id,
            timeout=settings.config.cluster_lock_timeout,
        )
        current_owner = cluster_lock.try_acquire()
        if current_owner != self.executor_id:
            # Either someone else has it or Redis is unavailable
            if current_owner is not None:
                logger.warning(
                    f"[{self.service_name}] Graph {graph_exec_id} already running on pod {current_owner}"
                )
            else:
                logger.warning(
                    f"[{self.service_name}] Could not acquire lock for {graph_exec_id} - Redis unavailable"
                )
            _ack_message(reject=True, requeue=True)
            return
        self._execution_locks[graph_exec_id] = cluster_lock

        logger.info(
            f"[{self.service_name}] Acquired cluster lock for {graph_exec_id} with executor {self.executor_id}"
        )

        cancel_event = threading.Event()

        future = self.executor.submit(
            execute_graph, graph_exec_entry, cancel_event, cluster_lock
        )
        self.active_graph_runs[graph_exec_id] = (future, cancel_event)
        self._update_prompt_metrics()

        def _on_run_done(f: Future):
            logger.info(f"[{self.service_name}] Run completed for {graph_exec_id}")
            try:
                if exec_error := f.exception():
                    logger.error(
                        f"[{self.service_name}] Execution for {graph_exec_id} failed: {type(exec_error)} {exec_error}"
                    )
                    _ack_message(reject=True, requeue=True)
                else:
                    _ack_message(reject=False, requeue=False)
            except BaseException as e:
                logger.exception(
                    f"[{self.service_name}] Error in run completion callback: {e}"
                )
            finally:
                # Release the cluster-wide execution lock
                if graph_exec_id in self._execution_locks:
                    self._execution_locks[graph_exec_id].release()
                    del self._execution_locks[graph_exec_id]
                self._cleanup_completed_runs()

        future.add_done_callback(_on_run_done)

    def _cleanup_completed_runs(self) -> list[str]:
        """Remove completed futures from active_graph_runs and update metrics"""
        completed_runs = []
        for graph_exec_id, (future, _) in self.active_graph_runs.items():
            if future.done():
                completed_runs.append(graph_exec_id)

        for geid in completed_runs:
            logger.info(f"[{self.service_name}] ✅ Cleaned up completed run {geid}")
            self.active_graph_runs.pop(geid, None)

        self._update_prompt_metrics()
        return completed_runs

    def _update_prompt_metrics(self):
        active_count = len(self.active_graph_runs)
        active_runs_gauge.set(active_count)
        if self.stop_consuming.is_set():
            utilization_gauge.set(1.0)
        else:
            utilization_gauge.set(active_count / self.pool_size)

    def _stop_message_consumers(
        self, thread: threading.Thread, client: SyncRabbitMQ, prefix: str
    ):
        try:
            channel = client.get_channel()
            channel.connection.add_callback_threadsafe(lambda: channel.stop_consuming())

            try:
                thread.join(timeout=300)
            except TimeoutError:
                logger.error(
                    f"{prefix} ⚠️ Run thread did not finish in time, forcing disconnect"
                )

            client.disconnect()
            logger.info(f"{prefix} ✅ Run client disconnected")
        except Exception as e:
            logger.error(f"{prefix} ⚠️ Error disconnecting run client: {type(e)} {e}")

    def cleanup(self):
        """Override cleanup to implement graceful shutdown with active execution waiting."""
        prefix = f"[{self.service_name}][on_graph_executor_stop {os.getpid()}]"
        logger.info(f"{prefix} 🧹 Starting graceful shutdown...")

        # Signal the consumer thread to stop (thread-safe)
        try:
            self.stop_consuming.set()
            run_channel = self.run_client.get_channel()
            run_channel.connection.add_callback_threadsafe(
                lambda: run_channel.stop_consuming()
            )
            logger.info(f"{prefix} ✅ Exec consumer has been signaled to stop")
        except Exception as e:
            logger.error(f"{prefix} ⚠️ Error signaling consumer to stop: {type(e)} {e}")

        # Wait for active executions to complete
        if self.active_graph_runs:
            logger.info(
                f"{prefix} ⏳ Waiting for {len(self.active_graph_runs)} active executions to complete..."
            )

            max_wait = GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS
            wait_interval = 5
            waited = 0

            while waited < max_wait:
                self._cleanup_completed_runs()
                if not self.active_graph_runs:
                    logger.info(f"{prefix} ✅ All active executions completed")
                    break
                else:
                    ids = [k.split("-")[0] for k in self.active_graph_runs.keys()]
                    logger.info(
                        f"{prefix} ⏳ Still waiting for {len(self.active_graph_runs)} executions: {ids}"
                    )

                    for graph_exec_id in self.active_graph_runs:
                        if lock := self._execution_locks.get(graph_exec_id):
                            lock.refresh()

                time.sleep(wait_interval)
                waited += wait_interval

            if self.active_graph_runs:
                logger.error(
                    f"{prefix} ⚠️ {len(self.active_graph_runs)} executions still running after {max_wait}s"
                )
            else:
                logger.info(f"{prefix} ✅ All executions completed gracefully")

        # Shutdown the executor
        try:
            self.executor.shutdown(cancel_futures=True, wait=False)
            logger.info(f"{prefix} ✅ Executor shutdown completed")
        except Exception as e:
            logger.error(f"{prefix} ⚠️ Error during executor shutdown: {type(e)} {e}")

        # Release remaining execution locks
        try:
            for lock in self._execution_locks.values():
                lock.release()
            self._execution_locks.clear()
            logger.info(f"{prefix} ✅ Released execution locks")
        except Exception as e:
            logger.warning(f"{prefix} ⚠️ Failed to release all locks: {e}")

        # Disconnect the run execution consumer
        self._stop_message_consumers(
            self.run_thread,
            self.run_client,
            prefix + " [run-consumer]",
        )
        self._stop_message_consumers(
            self.cancel_thread,
            self.cancel_client,
            prefix + " [cancel-consumer]",
        )

        logger.info(f"{prefix} ✅ Finished GraphExec cleanup")


# ------- UTILITIES ------- #


def get_db_client() -> "DatabaseManagerClient":
    return get_database_manager_client()


def get_db_async_client() -> "DatabaseManagerAsyncClient":
    return get_database_manager_async_client()


@func_retry
async def send_async_execution_update(
    entry: GraphExecution | NodeExecutionResult | None,
) -> None:
    if entry is None:
        return
    await get_async_execution_event_bus().publish(entry)


@func_retry
def send_execution_update(entry: GraphExecution | NodeExecutionResult | None):
    if entry is None:
        return
    return get_execution_event_bus().publish(entry)


async def async_update_node_execution_status(
    db_client: "DatabaseManagerAsyncClient",
    exec_id: str,
    status: ExecutionStatus,
    execution_data: BlockInput | None = None,
    stats: dict[str, Any] | None = None,
) -> NodeExecutionResult:
    """Sets status and fetches+broadcasts the latest state of the node execution"""
    exec_update = await db_client.update_node_execution_status(
        exec_id, status, execution_data, stats
    )
    await send_async_execution_update(exec_update)
    return exec_update


def update_node_execution_status(
    db_client: "DatabaseManagerClient",
    exec_id: str,
    status: ExecutionStatus,
    execution_data: BlockInput | None = None,
    stats: dict[str, Any] | None = None,
) -> NodeExecutionResult:
    """Sets status and fetches+broadcasts the latest state of the node execution"""
    exec_update = db_client.update_node_execution_status(
        exec_id, status, execution_data, stats
    )
    send_execution_update(exec_update)
    return exec_update


async def async_update_graph_execution_state(
    db_client: "DatabaseManagerAsyncClient",
    graph_exec_id: str,
    status: ExecutionStatus | None = None,
    stats: GraphExecutionStats | None = None,
) -> GraphExecution | None:
    """Sets status and fetches+broadcasts the latest state of the graph execution"""
    graph_update = await db_client.update_graph_execution_stats(
        graph_exec_id, status, stats
    )
    if graph_update:
        await send_async_execution_update(graph_update)
    else:
        logger.error(f"Failed to update graph execution stats for {graph_exec_id}")
    return graph_update


def update_graph_execution_state(
    db_client: "DatabaseManagerClient",
    graph_exec_id: str,
    status: ExecutionStatus | None = None,
    stats: GraphExecutionStats | None = None,
) -> GraphExecution | None:
    """Sets status and fetches+broadcasts the latest state of the graph execution"""
    graph_update = db_client.update_graph_execution_stats(graph_exec_id, status, stats)
    if graph_update:
        send_execution_update(graph_update)
    else:
        logger.error(f"Failed to update graph execution stats for {graph_exec_id}")
    return graph_update


@asynccontextmanager
async def synchronized(key: str, timeout: int = settings.config.cluster_lock_timeout):
    r = await redis.get_redis_async()
    lock: AsyncRedisLock = r.lock(f"lock:{key}", timeout=timeout)
    try:
        await lock.acquire()
        yield
    finally:
        if await lock.locked() and await lock.owned():
            try:
                await lock.release()
            except Exception as e:
                logger.warning(f"Failed to release lock for key {key}: {e}")


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
