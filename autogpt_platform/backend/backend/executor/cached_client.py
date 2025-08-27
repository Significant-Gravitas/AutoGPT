import logging
from typing import Any

from backend.executor.simple_cache import get_cache

logger = logging.getLogger(__name__)


class CachedDatabaseClient:
    def __init__(self, original_client):
        self._client = original_client
        self._cache = get_cache()

    def get_node(self, node_id: str) -> Any:
        cached = self._cache.get_node(node_id)
        if cached:
            return cached

        node = self._client.get_node(node_id)
        if node:
            self._cache.cache_node(node_id, node)
        return node

    def get_node_executions(self, graph_exec_id: str, *args, **kwargs) -> Any:
        if not args and not kwargs:
            cached = self._cache.get_node_executions(graph_exec_id)
            if cached:
                return cached

        executions = self._client.get_node_executions(graph_exec_id, *args, **kwargs)
        if not args and not kwargs:
            self._cache.cache_node_executions(graph_exec_id, executions)
        return executions

    def upsert_execution_output(self, *args, **kwargs) -> Any:
        node_exec_id = kwargs.get("node_exec_id") or (args[0] if args else None)
        output = kwargs.get("output") or (args[1] if len(args) > 1 else None)

        if node_exec_id and output:
            self._cache.queue_output_update(node_exec_id, output)
            return {"success": True}

        return self._client.upsert_execution_output(*args, **kwargs)

    def update_node_execution_status(self, *args, **kwargs) -> Any:
        node_exec_id = kwargs.get("node_exec_id") or (args[0] if args else None)
        status = kwargs.get("status") or (args[1] if len(args) > 1 else None)

        if node_exec_id and status:
            self._cache.queue_status_update(node_exec_id, status)
            return {"success": True}

        return self._client.update_node_execution_status(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


def wrap_client(original_client):
    return CachedDatabaseClient(original_client)
