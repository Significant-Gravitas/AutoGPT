import logging
import threading
import uuid
from concurrent.futures import Executor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar, cast

from backend.data import redis_client as redis
from backend.data.credit import UsageTransactionMetadata
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionMeta,
    NodeExecutionResult,
)
from backend.data.graph import Node
from backend.data.model import GraphExecutionStats
from backend.executor.execution_cache import ExecutionCache
from backend.util.clients import (
    get_database_manager_async_client,
    get_database_manager_client,
    get_execution_event_bus,
)
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.executor import DatabaseManagerAsyncClient, DatabaseManagerClient

settings = Settings()
logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def non_blocking_persist(func: Callable[P, T]) -> Callable[P, None]:
    from functools import wraps

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        # First argument is always self for methods - access through cast for typing
        self = cast("ExecutionDataClient", args[0])
        future = self._executor.submit(func, *args, **kwargs)
        self._pending_tasks.add(future)

    return wrapper


class ExecutionDataClient:
    def __init__(
        self, executor: Executor, graph_exec_id: str, graph_metadata: GraphExecutionMeta
    ):
        self._executor = executor
        self._graph_exec_id = graph_exec_id
        self._cache = ExecutionCache(graph_exec_id, self.db_client_sync)
        self._pending_tasks = set()
        self._graph_metadata = graph_metadata
        self.graph_lock = threading.RLock()

    def finalize_execution(self, timeout: float = 30.0):
        logger.info(f"Flushing db writes for execution {self._graph_exec_id}")
        exceptions = []

        # Wait for all pending database operations to complete
        logger.debug(
            f"Waiting for {len(self._pending_tasks)} pending database operations"
        )
        for future in list(self._pending_tasks):
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Background database operation failed: {e}")
                exceptions.append(e)
            finally:
                self._pending_tasks.discard(future)

        self._cache.finalize()

        if exceptions:
            logger.error(f"Background persistence failed with {len(exceptions)} errors")
            raise RuntimeError(
                f"Background persistence failed with {len(exceptions)} errors: {exceptions}"
            )

    @property
    def db_client_async(self) -> "DatabaseManagerAsyncClient":
        return get_database_manager_async_client()

    @property
    def db_client_sync(self) -> "DatabaseManagerClient":
        return get_database_manager_client()

    @property
    def event_bus(self):
        return get_execution_event_bus()

    async def get_node(self, node_id: str) -> Node:
        return await self.db_client_async.get_node(node_id)

    def spend_credits(
        self,
        user_id: str,
        cost: int,
        metadata: UsageTransactionMetadata,
    ) -> int:
        return self.db_client_sync.spend_credits(
            user_id=user_id, cost=cost, metadata=metadata
        )

    def get_graph_execution_meta(
        self, user_id: str, execution_id: str
    ) -> GraphExecutionMeta | None:
        return self.db_client_sync.get_graph_execution_meta(
            user_id=user_id, execution_id=execution_id
        )

    def get_graph_metadata(
        self, graph_id: str, graph_version: int | None = None
    ) -> Any:
        return self.db_client_sync.get_graph_metadata(graph_id, graph_version)

    def get_credits(self, user_id: str) -> int:
        return self.db_client_sync.get_credits(user_id)

    def get_user_email_by_id(self, user_id: str) -> str | None:
        return self.db_client_sync.get_user_email_by_id(user_id)

    def get_latest_node_execution(self, node_id: str) -> NodeExecutionResult | None:
        return self._cache.get_latest_node_execution(node_id)

    def get_node_execution(self, node_exec_id: str) -> NodeExecutionResult | None:
        return self._cache.get_node_execution(node_exec_id)

    def get_node_executions(
        self,
        *,
        node_id: str | None = None,
        statuses: list[ExecutionStatus] | None = None,
        block_ids: list[str] | None = None,
    ) -> list[NodeExecutionResult]:
        return self._cache.get_node_executions(
            statuses=statuses, block_ids=block_ids, node_id=node_id
        )

    def update_node_status_and_publish(
        self,
        exec_id: str,
        status: ExecutionStatus,
        execution_data: dict | None = None,
        stats: dict[str, Any] | None = None,
    ):
        self._cache.update_node_execution_status(exec_id, status, execution_data, stats)
        self._persist_node_status_to_db(exec_id, status, execution_data, stats)

    def upsert_execution_input(
        self, node_id: str, input_name: str, input_data: Any, block_id: str
    ) -> tuple[str, dict]:
        # Validate input parameters to prevent foreign key constraint errors
        if not node_id or not isinstance(node_id, str):
            raise ValueError(f"Invalid node_id: {node_id}")
        if not self._graph_exec_id or not isinstance(self._graph_exec_id, str):
            raise ValueError(f"Invalid graph_exec_id: {self._graph_exec_id}")
        if not block_id or not isinstance(block_id, str):
            raise ValueError(f"Invalid block_id: {block_id}")

        # UPDATE: Try to find an existing incomplete execution for this node and input
        if result := self._cache.find_incomplete_execution_for_input(
            node_id, input_name
        ):
            exec_id, _ = result
            updated_input_data = self._cache.update_execution_input(
                exec_id, input_name, input_data
            )
            self._persist_add_input_to_db(exec_id, input_name, input_data)
            return exec_id, updated_input_data

        # CREATE: No suitable execution found, create new one
        node_exec_id = str(uuid.uuid4())
        logger.debug(
            f"Creating new execution {node_exec_id} for node {node_id} "
            f"in graph execution {self._graph_exec_id}"
        )

        new_execution = NodeExecutionResult(
            user_id=self._graph_metadata.user_id,
            graph_id=self._graph_metadata.graph_id,
            graph_version=self._graph_metadata.graph_version,
            graph_exec_id=self._graph_exec_id,
            node_exec_id=node_exec_id,
            node_id=node_id,
            block_id=block_id,
            status=ExecutionStatus.INCOMPLETE,
            input_data={input_name: input_data},
            output_data={},
            add_time=datetime.now(timezone.utc),
        )
        self._cache.add_node_execution(node_exec_id, new_execution)
        self._persist_new_node_execution_to_db(
            node_exec_id, node_id, input_name, input_data
        )

        return node_exec_id, {input_name: input_data}

    def upsert_execution_output(
        self, node_exec_id: str, output_name: str, output_data: Any
    ):
        self._cache.upsert_execution_output(node_exec_id, output_name, output_data)
        self._persist_execution_output_to_db(node_exec_id, output_name, output_data)

    def update_graph_stats_and_publish(
        self,
        status: ExecutionStatus | None = None,
        stats: GraphExecutionStats | None = None,
    ) -> None:
        stats_dict = stats.model_dump() if stats else None
        self._cache.update_graph_stats(status=status, stats=stats_dict)
        self._persist_graph_stats_to_db(status=status, stats=stats)

    def update_graph_start_time_and_publish(self) -> None:
        self._cache.update_graph_start_time()
        self._persist_graph_start_time_to_db()

    @non_blocking_persist
    def _persist_node_status_to_db(
        self,
        exec_id: str,
        status: ExecutionStatus,
        execution_data: dict | None = None,
        stats: dict[str, Any] | None = None,
    ):
        exec_update = self.db_client_sync.update_node_execution_status(
            exec_id, status, execution_data, stats
        )
        self.event_bus.publish(exec_update)

    @non_blocking_persist
    def _persist_add_input_to_db(
        self, node_exec_id: str, input_name: str, input_data: Any
    ):
        self.db_client_sync.add_input_to_node_execution(
            node_exec_id=node_exec_id,
            input_name=input_name,
            input_data=input_data,
        )

    @non_blocking_persist
    def _persist_execution_output_to_db(
        self, node_exec_id: str, output_name: str, output_data: Any
    ):
        self.db_client_sync.upsert_execution_output(
            node_exec_id=node_exec_id,
            output_name=output_name,
            output_data=output_data,
        )
        if exec_update := self.get_node_execution(node_exec_id):
            self.event_bus.publish(exec_update)

    @non_blocking_persist
    def _persist_graph_stats_to_db(
        self,
        status: ExecutionStatus | None = None,
        stats: GraphExecutionStats | None = None,
    ):
        graph_update = self.db_client_sync.update_graph_execution_stats(
            self._graph_exec_id, status, stats
        )
        if not graph_update:
            raise RuntimeError(
                f"Failed to update graph execution stats for {self._graph_exec_id}"
            )
        self.event_bus.publish(graph_update)

    @non_blocking_persist
    def _persist_graph_start_time_to_db(self):
        graph_update = self.db_client_sync.update_graph_execution_start_time(
            self._graph_exec_id
        )
        if not graph_update:
            raise RuntimeError(
                f"Failed to update graph execution start time for {self._graph_exec_id}"
            )
        self.event_bus.publish(graph_update)

    async def generate_activity_status(
        self,
        graph_id: str,
        graph_version: int,
        execution_stats: GraphExecutionStats,
        user_id: str,
        execution_status: ExecutionStatus,
    ) -> str | None:
        from backend.executor.activity_status_generator import (
            generate_activity_status_for_execution,
        )

        return await generate_activity_status_for_execution(
            graph_exec_id=self._graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            execution_stats=execution_stats,
            db_client=self.db_client_async,
            user_id=user_id,
            execution_status=execution_status,
        )

    @non_blocking_persist
    def _send_execution_update(self, execution: NodeExecutionResult):
        """Send execution update to event bus."""
        try:
            self.event_bus.publish(execution)
        except Exception as e:
            logger.warning(f"Failed to send execution update: {e}")

    @non_blocking_persist
    def _persist_new_node_execution_to_db(
        self, node_exec_id: str, node_id: str, input_name: str, input_data: Any
    ):
        try:
            self.db_client_sync.create_node_execution(
                node_exec_id=node_exec_id,
                node_id=node_id,
                graph_exec_id=self._graph_exec_id,
                input_name=input_name,
                input_data=input_data,
            )
        except Exception as e:
            logger.error(
                f"Failed to create node execution {node_exec_id} for node {node_id} "
                f"in graph execution {self._graph_exec_id}: {e}"
            )
            raise

    def increment_execution_count(self, user_id: str) -> int:
        r = redis.get_redis()
        k = f"uec:{user_id}"
        counter = cast(int, r.incr(k))
        if counter == 1:
            r.expire(k, settings.config.execution_counter_expiration_time)
        return counter
