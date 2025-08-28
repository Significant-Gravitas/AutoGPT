import logging
import threading
from collections import OrderedDict
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.data.model import GraphExecutionStats, NodeExecutionStats

if TYPE_CHECKING:
    from backend.executor import DatabaseManagerClient

logger = logging.getLogger(__name__)


def with_lock(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


class ExecutionCache:
    def __init__(self, graph_exec_id: str, db_client: "DatabaseManagerClient"):
        self._lock = threading.RLock()
        self._graph_exec_id = graph_exec_id
        self._graph_stats: GraphExecutionStats = GraphExecutionStats()
        self._node_executions: OrderedDict[str, NodeExecutionResult] = OrderedDict()

        for execution in db_client.get_node_executions(self._graph_exec_id):
            self._node_executions[execution.node_exec_id] = execution

    @with_lock
    def get_node_execution(self, node_exec_id: str) -> Optional[NodeExecutionResult]:
        execution = self._node_executions.get(node_exec_id)
        return execution.model_copy(deep=True) if execution else None

    @with_lock
    def get_latest_node_execution(self, node_id: str) -> Optional[NodeExecutionResult]:
        for execution in reversed(self._node_executions.values()):
            if (
                execution.node_id == node_id
                and execution.status != ExecutionStatus.INCOMPLETE
            ):
                return execution.model_copy(deep=True)
        return None

    @with_lock
    def get_node_executions(
        self,
        *,
        statuses: Optional[list] = None,
        block_ids: Optional[list] = None,
        node_id: Optional[str] = None,
    ):
        results = []
        for execution in self._node_executions.values():
            if statuses and execution.status not in statuses:
                continue
            if block_ids and execution.block_id not in block_ids:
                continue
            if node_id and execution.node_id != node_id:
                continue
            results.append(execution.model_copy(deep=True))
        return results

    @with_lock
    def update_node_execution_status(
        self,
        exec_id: str,
        status: ExecutionStatus,
        execution_data: Optional[dict] = None,
        stats: Optional[dict] = None,
    ):
        if exec_id not in self._node_executions:
            raise RuntimeError(f"Execution {exec_id} not found in cache")

        execution = self._node_executions[exec_id]
        execution.status = status

        if execution_data:
            execution.input_data.update(execution_data)

        if stats:
            execution.stats = execution.stats or NodeExecutionStats()
            current_stats = execution.stats.model_dump()
            current_stats.update(stats)
            execution.stats = NodeExecutionStats.model_validate(current_stats)

    @with_lock
    def upsert_execution_output(
        self, node_exec_id: str, output_name: str, output_data: Any
    ) -> NodeExecutionResult:
        if node_exec_id not in self._node_executions:
            raise RuntimeError(f"Execution {node_exec_id} not found in cache")

        execution = self._node_executions[node_exec_id]
        if output_name not in execution.output_data:
            execution.output_data[output_name] = []
        execution.output_data[output_name].append(output_data)

        return execution

    @with_lock
    def update_graph_stats(
        self, status: Optional[ExecutionStatus] = None, stats: Optional[dict] = None
    ):
        if status is not None:
            # We don't cache the graph status since it's not used in execution logic.
            # But adding here for future needs and conssistency with the DB model signature.
            pass

        if stats is not None:
            current_stats = self._graph_stats.model_dump()
            current_stats.update(stats)
            self._graph_stats = GraphExecutionStats.model_validate(current_stats)

    @with_lock
    def update_graph_start_time(self):
        """Update graph start time - this is primarily handled by database persistence.

        The cache doesn't need to store start_time since it's metadata (GraphExecutionMeta),
        not execution statistics (GraphExecutionStats). The actual start_time update
        happens in the database via _persist_graph_start_time_to_db.
        """
        pass

    @with_lock
    def find_incomplete_execution_for_input(
        self, node_id: str, input_name: str
    ) -> tuple[str, NodeExecutionResult] | None:
        for exec_id, execution in self._node_executions.items():
            # Debug logging to understand what's happening
            if execution.node_id == node_id:
                print(
                    f"DEBUG: Found execution {exec_id} for node {node_id}, status={execution.status}, input_name={input_name}, input_data={execution.input_data}"
                )
            if (
                execution.node_id == node_id
                and execution.status == ExecutionStatus.INCOMPLETE
                and input_name not in execution.input_data  # Only if input missing
            ):
                print(f"DEBUG: Returning existing execution {exec_id}")
                return exec_id, execution
        print(
            f"DEBUG: No incomplete execution found for node {node_id}, input {input_name}"
        )
        return None

    @with_lock
    def add_node_execution(
        self, node_exec_id: str, execution: NodeExecutionResult
    ) -> None:
        self._node_executions[node_exec_id] = execution

    @with_lock
    def update_execution_input(
        self, exec_id: str, input_name: str, input_data: Any
    ) -> dict:
        if exec_id not in self._node_executions:
            raise RuntimeError(f"Execution {exec_id} not found in cache")
        execution = self._node_executions[exec_id]
        execution.input_data[input_name] = input_data
        return execution.input_data.copy()

    def finalize(self) -> None:
        with self._lock:
            self._node_executions.clear()
            self._graph_stats = GraphExecutionStats()
