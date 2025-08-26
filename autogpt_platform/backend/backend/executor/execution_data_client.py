import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from backend.executor.simple_cache import get_cache

if TYPE_CHECKING:
    from backend.data.graph import Node

logger = logging.getLogger(__name__)


class ExecutionDataClient:
    def __init__(self, backend_client):
        self._backend = backend_client
        self._cache = get_cache()

    def get_node(self, node_id: str) -> "Node":
        cached = self._cache.get_node(node_id)
        if cached:
            return cached

        node = self._backend.get_node(node_id)
        if node:
            self._cache.cache_node(node_id, node)
        return node

    def get_node_executions(self, graph_exec_id: str, *args, **kwargs):
        if not args and not kwargs:
            cached = self._cache.get_node_executions(graph_exec_id)
            if cached:
                return cached

        executions = self._backend.get_node_executions(graph_exec_id, *args, **kwargs)
        if not args and not kwargs:
            self._cache.cache_node_executions(graph_exec_id, executions)
        return executions

    def upsert_execution_output(self, *args, **kwargs):
        node_exec_id = kwargs.get("node_exec_id") or (args[0] if args else None)
        output = kwargs.get("output") or (args[1] if len(args) > 1 else None)

        if node_exec_id and output:
            self._cache.queue_output_update(node_exec_id, output)
            return {"success": True}

        return self._backend.upsert_execution_output(*args, **kwargs)

    def update_node_execution_status(self, *args, **kwargs):
        node_exec_id = kwargs.get("node_exec_id") or (args[0] if args else None)
        status = kwargs.get("status") or (args[1] if len(args) > 1 else None)

        if node_exec_id and status:
            self._cache.queue_status_update(node_exec_id, status)
            return {"success": True}

        return self._backend.update_node_execution_status(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backend, name)


class ExecutionDataAsyncClient:
    def __init__(self, backend_client):
        self._backend = backend_client

    async def __getattr__(self, name):
        return getattr(self._backend, name)


def create_execution_data_client():
    from backend.util.clients import get_database_manager_client

    backend = get_database_manager_client()
    return ExecutionDataClient(backend)


def create_execution_data_async_client():
    from backend.util.clients import get_database_manager_async_client

    backend = get_database_manager_async_client()
    return ExecutionDataAsyncClient(backend)


@asynccontextmanager
async def execution_lock(key: str, timeout: int = 60):
    from redis.asyncio.lock import Lock as RedisLock

    from backend.data import redis

    r = await redis.get_redis_async()
    lock: RedisLock = r.lock(f"lock:{key}", timeout=timeout)
    try:
        await lock.acquire()
        yield
    finally:
        if await lock.locked() and await lock.owned():
            await lock.release()


def get_execution_counter() -> object:
    from typing import cast

    from backend.data import redis
    from backend.util import settings

    class ExecutionCounter:
        def increment(self, user_id: str) -> int:
            r = redis.get_redis()
            k = f"uec:{user_id}"
            counter = cast(int, r.incr(k))
            if counter == 1:
                r.expire(k, settings.config.execution_counter_expiration_time)
            return counter

    return ExecutionCounter()
