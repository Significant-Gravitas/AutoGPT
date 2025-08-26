import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SimpleExecutorCache:
    def __init__(self):
        self._nodes: Dict[str, Any] = {}
        self._node_executions: Dict[str, List[Any]] = {}
        self._execution_outputs: List[Dict] = []
        self._status_updates: List[Dict] = []
        self._lock = threading.RLock()
        self._cached_graphs: Set[str] = set()

    def cache_node(self, node_id: str, node: Any):
        with self._lock:
            self._nodes[node_id] = node

    def get_node(self, node_id: str) -> Optional[Any]:
        with self._lock:
            return self._nodes.get(node_id)

    def cache_node_executions(self, graph_exec_id: str, executions: List[Any]):
        with self._lock:
            self._node_executions[graph_exec_id] = executions
            self._cached_graphs.add(graph_exec_id)

    def get_node_executions(self, graph_exec_id: str) -> Optional[List[Any]]:
        with self._lock:
            return self._node_executions.get(graph_exec_id)

    def queue_output_update(self, node_exec_id: str, output: Any):
        with self._lock:
            self._execution_outputs.append(
                {
                    "node_exec_id": node_exec_id,
                    "output": output,
                    "timestamp": time.time(),
                }
            )

    def queue_status_update(self, node_exec_id: str, status: Any):
        with self._lock:
            self._status_updates.append(
                {
                    "node_exec_id": node_exec_id,
                    "status": status,
                    "timestamp": time.time(),
                }
            )

    def get_pending_updates(self) -> tuple[List[Dict], List[Dict]]:
        with self._lock:
            outputs = self._execution_outputs.copy()
            statuses = self._status_updates.copy()
            self._execution_outputs.clear()
            self._status_updates.clear()
            return outputs, statuses

    def clear_graph_cache(self, graph_exec_id: str):
        with self._lock:
            if graph_exec_id in self._node_executions:
                del self._node_executions[graph_exec_id]
            self._cached_graphs.discard(graph_exec_id)

    def clear_all(self):
        with self._lock:
            self._nodes.clear()
            self._node_executions.clear()
            self._execution_outputs.clear()
            self._status_updates.clear()
            self._cached_graphs.clear()


_executor_cache: Optional[SimpleExecutorCache] = None


def get_cache() -> SimpleExecutorCache:
    global _executor_cache
    if _executor_cache is None:
        _executor_cache = SimpleExecutorCache()
    return _executor_cache


def clear_cache():
    global _executor_cache
    if _executor_cache:
        _executor_cache.clear_all()
        _executor_cache = None
