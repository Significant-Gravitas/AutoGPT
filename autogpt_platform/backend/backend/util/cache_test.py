"""Tests for the @thread_cached decorator.

This module tests the thread-local caching functionality including:
- Basic caching for sync and async functions
- Thread isolation (each thread has its own cache)
- Cache clearing functionality
- Exception handling (exceptions are not cached)
- Argument handling (positional vs keyword arguments)
"""

import asyncio
import gc
import inspect
import logging
import pickle
import threading
import time
import traceback
import uuid
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import Mock

import pytest
import redis
import redis.asyncio.cluster as async_cluster
import redis.asyncio.connection as async_connection
import redis.connection
from redis.cluster import ClusterNode, RedisCluster
from redis.exceptions import AskError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import MaxConnectionsError, MovedError, ReadOnlyError

from backend.util import cache as cache_module
from backend.util import cache_redis
from backend.util.cache import (
    _SCAN_BATCH_SIZE,
    cached,
    clear_thread_cache,
    thread_cached,
)
from backend.util.cache_test_helpers import (
    cache_key,
    callers_in_flight,
    forget_connections,
    hold_cache_read,
)
from backend.util.redis_client_guard import is_open

# Only a step that never comes keeps a wait going this long.
_HANG_SECONDS = 10.0


def _event_loop_factories() -> list[Any]:
    """asyncio's own loop, which workers and tests run, and uvloop's, which
    uvicorn runs the servers on wherever it's installed (not on Windows)."""
    factories = [pytest.param(asyncio.new_event_loop, id="asyncio")]
    try:
        import uvloop
    except ImportError:
        return factories
    return [*factories, pytest.param(uvloop.new_event_loop, id="uvloop")]


_ON_EVERY_KIND_OF_LOOP = pytest.mark.parametrize("new_loop", _event_loop_factories())
NewLoop = Callable[[], asyncio.AbstractEventLoop]


@pytest.fixture(autouse=True)
def no_redis_round_trip_on_an_event_loop(monkeypatch: pytest.MonkeyPatch):
    """Fail a test in which the cache's sync Redis client sends a command from
    a thread running an event loop: every coroutine on that loop then waits
    for Redis's answer, so one shard that stops answering stops them all. A
    coroutine reads the cache through ``cache_redis.get_async_redis``, and
    calls a sync cache method (``cache_clear``, ``cache_delete``) through
    ``asyncio.to_thread``."""
    sent_on_a_loop: list[str] = []
    send = redis.connection.AbstractConnection.send_packed_command

    def guarded_send(self, command, check_health=True):
        # Noted, not raised: the cache would log a raised error and carry on,
        # and the client's connect retries would wait it out on the loop.
        if _runs_an_event_loop():
            sent_on_a_loop.append(_backend_callers())
        return send(self, command, check_health)

    monkeypatch.setattr(
        redis.connection.AbstractConnection, "send_packed_command", guarded_send
    )
    yield
    assert not sent_on_a_loop, (
        "The cache's sync Redis client ran on an event loop's thread, blocking "
        "every coroutine on it until Redis answered:\n" + "\n".join(sent_on_a_loop)
    )


def _runs_an_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _backend_callers() -> str:
    frames = [
        frame
        for frame in traceback.extract_stack()[:-2]
        if "/backend/backend/" in frame.filename
    ]
    return "  " + " <- ".join(
        f"{frame.filename.rsplit('/backend/', 1)[-1]}:{frame.lineno} {frame.name}"
        for frame in reversed(frames[-4:])
    )


class TestThreadCached:
    def test_sync_function_caching(self):
        call_count = 0

        @thread_cached
        def expensive_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        assert expensive_function(1, 2) == 3
        assert call_count == 1

        assert expensive_function(1, 2) == 3
        assert call_count == 1

        assert expensive_function(1, y=2) == 3
        assert call_count == 2

        assert expensive_function(2, 3) == 5
        assert call_count == 3

        assert expensive_function(1) == 1
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_async_function_caching(self):
        call_count = 0

        @thread_cached
        async def expensive_async_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x + y

        assert await expensive_async_function(1, 2) == 3
        assert call_count == 1

        assert await expensive_async_function(1, 2) == 3
        assert call_count == 1

        assert await expensive_async_function(1, y=2) == 3
        assert call_count == 2

        assert await expensive_async_function(2, 3) == 5
        assert call_count == 3

    def test_thread_isolation(self):
        call_count = 0
        results = {}

        @thread_cached
        def thread_specific_function(x: int) -> str:
            nonlocal call_count
            call_count += 1
            return f"{threading.current_thread().name}-{x}"

        def worker(thread_id: int):
            result1 = thread_specific_function(1)
            result2 = thread_specific_function(1)
            result3 = thread_specific_function(2)
            results[thread_id] = (result1, result2, result3)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            for future in futures:
                future.result()

        assert call_count >= 2

        for thread_id, (r1, r2, r3) in results.items():
            assert r1 == r2
            assert r1 != r3

    @pytest.mark.asyncio
    async def test_async_thread_isolation(self):
        call_count = 0
        results = {}

        @thread_cached
        async def async_thread_specific_function(x: int) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"{threading.current_thread().name}-{x}"

        async def async_worker(worker_id: int):
            result1 = await async_thread_specific_function(1)
            result2 = await async_thread_specific_function(1)
            result3 = await async_thread_specific_function(2)
            results[worker_id] = (result1, result2, result3)

        tasks = [async_worker(i) for i in range(3)]
        await asyncio.gather(*tasks)

        for worker_id, (r1, r2, r3) in results.items():
            assert r1 == r2
            assert r1 != r3

    def test_clear_cache_sync(self):
        call_count = 0

        @thread_cached
        def clearable_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert clearable_function(5) == 10
        assert call_count == 1

        assert clearable_function(5) == 10
        assert call_count == 1

        clear_thread_cache(clearable_function)

        assert clearable_function(5) == 10
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_clear_cache_async(self):
        call_count = 0

        @thread_cached
        async def clearable_async_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        assert await clearable_async_function(5) == 10
        assert call_count == 1

        assert await clearable_async_function(5) == 10
        assert call_count == 1

        clear_thread_cache(clearable_async_function)

        assert await clearable_async_function(5) == 10
        assert call_count == 2

    def test_simple_arguments(self):
        call_count = 0

        @thread_cached
        def simple_function(a: str, b: int, c: str = "default") -> str:
            nonlocal call_count
            call_count += 1
            return f"{a}-{b}-{c}"

        # First call with all positional args
        result1 = simple_function("test", 42, "custom")
        assert call_count == 1

        # Same args, all positional - should hit cache
        result2 = simple_function("test", 42, "custom")
        assert call_count == 1
        assert result1 == result2

        # Same values but last arg as keyword - creates different cache key
        result3 = simple_function("test", 42, c="custom")
        assert call_count == 2
        assert result1 == result3  # Same result, different cache entry

        # Different value - new cache entry
        result4 = simple_function("test", 43, "custom")
        assert call_count == 3
        assert result1 != result4

    def test_positional_vs_keyword_args(self):
        """Test that positional and keyword arguments create different cache entries."""
        call_count = 0

        @thread_cached
        def func(a: int, b: int = 10) -> str:
            nonlocal call_count
            call_count += 1
            return f"result-{a}-{b}"

        # All positional
        result1 = func(1, 2)
        assert call_count == 1
        assert result1 == "result-1-2"

        # Same values, but second arg as keyword
        result2 = func(1, b=2)
        assert call_count == 2  # Different cache key!
        assert result2 == "result-1-2"  # Same result

        # Verify both are cached separately
        func(1, 2)  # Uses first cache entry
        assert call_count == 2

        func(1, b=2)  # Uses second cache entry
        assert call_count == 2

    def test_exception_handling(self):
        call_count = 0

        @thread_cached
        def failing_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if x < 0:
                raise ValueError("Negative value")
            return x * 2

        assert failing_function(5) == 10
        assert call_count == 1

        with pytest.raises(ValueError):
            failing_function(-1)
        assert call_count == 2

        with pytest.raises(ValueError):
            failing_function(-1)
        assert call_count == 3

        assert failing_function(5) == 10
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_exception_handling(self):
        call_count = 0

        @thread_cached
        async def async_failing_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if x < 0:
                raise ValueError("Negative value")
            return x * 2

        assert await async_failing_function(5) == 10
        assert call_count == 1

        with pytest.raises(ValueError):
            await async_failing_function(-1)
        assert call_count == 2

        with pytest.raises(ValueError):
            await async_failing_function(-1)
        assert call_count == 3

    def test_sync_caching_performance(self):
        calls = 0

        @thread_cached
        def slow_function(x: int) -> int:
            nonlocal calls
            calls += 1
            time.sleep(0.1)
            return x * 2

        assert slow_function(5) == 10
        assert slow_function(5) == 10
        # The slow body ran once: the second call was a cache hit.
        assert calls == 1

    @pytest.mark.asyncio
    async def test_async_caching_performance(self):
        calls = 0

        @thread_cached
        async def slow_async_function(x: int) -> int:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.1)
            return x * 2

        assert await slow_async_function(5) == 10
        assert await slow_async_function(5) == 10
        # The slow body ran once: the second call was a cache hit.
        assert calls == 1

    def test_with_mock_objects(self):
        mock = Mock(return_value=42)

        @thread_cached
        def function_using_mock(x: int) -> int:
            return mock(x)

        assert function_using_mock(1) == 42
        assert mock.call_count == 1

        assert function_using_mock(1) == 42
        assert mock.call_count == 1

        assert function_using_mock(2) == 42
        assert mock.call_count == 2


class TestCache:
    """Tests for the unified @cache decorator (works for both sync and async)."""

    def test_basic_sync_caching(self):
        """Test basic sync caching functionality."""
        call_count = 0

        @cached(ttl_seconds=300)
        def expensive_sync_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        # First call
        result1 = expensive_sync_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = expensive_sync_function(1, 2)
        assert result2 == 3
        assert call_count == 1

        # Different args - should call function again
        result3 = expensive_sync_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_basic_async_caching(self):
        """Test basic async caching functionality."""
        call_count = 0

        @cached(ttl_seconds=300)
        async def expensive_async_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x + y

        # First call
        result1 = await expensive_async_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = await expensive_async_function(1, 2)
        assert result2 == 3
        assert call_count == 1

        # Different args - should call function again
        result3 = await expensive_async_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_sync_thundering_herd_protection(self):
        """Test that concurrent sync calls don't cause thundering herd."""
        call_count = 0
        results = []

        @cached(ttl_seconds=300)
        def slow_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return x * x

        def worker():
            result = slow_function(5)
            results.append(result)

        # Launch multiple concurrent threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(5)]
            for future in futures:
                future.result()

        # All results should be the same
        assert all(result == 25 for result in results)
        # Only one thread should have executed the expensive operation
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_thundering_herd_protection(self):
        """Test that concurrent async calls don't cause thundering herd."""
        call_count = 0

        @cached(ttl_seconds=300)
        async def slow_async_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return x * x

        # Launch concurrent coroutines
        tasks = [slow_async_function(7) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be the same
        assert all(result == 49 for result in results)
        # Only one coroutine should have executed the expensive operation
        assert call_count == 1

    def test_ttl_functionality(self):
        """Test TTL functionality with sync function."""
        call_count = 0

        @cached(maxsize=10, ttl_seconds=1)  # Short TTL
        def ttl_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 3

        # First call
        result1 = ttl_function(3)
        assert result1 == 9
        assert call_count == 1

        # Second call immediately - should use cache
        result2 = ttl_function(3)
        assert result2 == 9
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Third call after expiration - should call function again
        result3 = ttl_function(3)
        assert result3 == 9
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_ttl_functionality(self):
        """Test TTL functionality with async function."""
        call_count = 0

        @cached(maxsize=10, ttl_seconds=1)  # Short TTL
        async def async_ttl_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 4

        # First call
        result1 = await async_ttl_function(3)
        assert result1 == 12
        assert call_count == 1

        # Second call immediately - should use cache
        result2 = await async_ttl_function(3)
        assert result2 == 12
        assert call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # Third call after expiration - should call function again
        result3 = await async_ttl_function(3)
        assert result3 == 12
        assert call_count == 2

    def test_cache_info(self):
        """Test cache info functionality."""

        @cached(maxsize=10, ttl_seconds=60)
        def info_test_function(x: int) -> int:
            return x * 3

        # Check initial cache info
        info = info_test_function.cache_info()
        assert info["size"] == 0
        assert info["maxsize"] == 10
        assert info["ttl_seconds"] == 60

        # Add an entry
        info_test_function(1)
        info = info_test_function.cache_info()
        assert info["size"] == 1

    def test_cache_clear(self):
        """Test cache clearing functionality."""
        call_count = 0

        @cached(ttl_seconds=300)
        def clearable_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 4

        # First call
        result1 = clearable_function(2)
        assert result1 == 8
        assert call_count == 1

        # Second call - should use cache
        result2 = clearable_function(2)
        assert result2 == 8
        assert call_count == 1

        # Clear cache
        clearable_function.cache_clear()

        # Third call after clear - should call function again
        result3 = clearable_function(2)
        assert result3 == 8
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_cache_clear(self):
        """Test cache clearing functionality with async function."""
        call_count = 0

        @cached(ttl_seconds=300)
        async def async_clearable_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 5

        # First call
        result1 = await async_clearable_function(2)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = await async_clearable_function(2)
        assert result2 == 10
        assert call_count == 1

        # Clear cache
        async_clearable_function.cache_clear()

        # Third call after clear - should call function again
        result3 = await async_clearable_function(2)
        assert result3 == 10
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_returns_results_not_coroutines(self):
        """Test that cached async functions return actual results, not coroutines."""
        call_count = 0

        @cached(ttl_seconds=300)
        async def async_result_function(x: int) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"result_{x}"

        # First call
        result1 = await async_result_function(1)
        assert result1 == "result_1"
        assert isinstance(result1, str)  # Should be string, not coroutine
        assert call_count == 1

        # Second call - should return cached result (string), not coroutine
        result2 = await async_result_function(1)
        assert result2 == "result_1"
        assert isinstance(result2, str)  # Should be string, not coroutine
        assert call_count == 1  # Function should not be called again

        # Verify results are identical
        assert result1 is result2  # Should be same cached object

    def test_cache_delete(self):
        """Test selective cache deletion functionality."""
        call_count = 0

        @cached(ttl_seconds=300)
        def deletable_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 6

        # First call for x=1
        result1 = deletable_function(1)
        assert result1 == 6
        assert call_count == 1

        # First call for x=2
        result2 = deletable_function(2)
        assert result2 == 12
        assert call_count == 2

        # Second calls - should use cache
        assert deletable_function(1) == 6
        assert deletable_function(2) == 12
        assert call_count == 2

        # Delete specific entry for x=1
        was_deleted = deletable_function.cache_delete(1)
        assert was_deleted is True

        # Call with x=1 should execute function again
        result3 = deletable_function(1)
        assert result3 == 6
        assert call_count == 3

        # Call with x=2 should still use cache
        assert deletable_function(2) == 12
        assert call_count == 3

        # Try to delete non-existent entry
        was_deleted = deletable_function.cache_delete(99)
        assert was_deleted is False

    @pytest.mark.asyncio
    async def test_async_cache_delete(self):
        """Test selective cache deletion functionality with async function."""
        call_count = 0

        @cached(ttl_seconds=300)
        async def async_deletable_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 7

        # First call for x=1
        result1 = await async_deletable_function(1)
        assert result1 == 7
        assert call_count == 1

        # First call for x=2
        result2 = await async_deletable_function(2)
        assert result2 == 14
        assert call_count == 2

        # Second calls - should use cache
        assert await async_deletable_function(1) == 7
        assert await async_deletable_function(2) == 14
        assert call_count == 2

        # Delete specific entry for x=1
        was_deleted = async_deletable_function.cache_delete(1)
        assert was_deleted is True

        # Call with x=1 should execute function again
        result3 = await async_deletable_function(1)
        assert result3 == 7
        assert call_count == 3

        # Call with x=2 should still use cache
        assert await async_deletable_function(2) == 14
        assert call_count == 3

        # Try to delete non-existent entry
        was_deleted = async_deletable_function.cache_delete(99)
        assert was_deleted is False


class TestSharedCache:
    """Tests for shared_cache (Redis-backed) functionality."""

    def test_scans_walk_redis_in_large_batches(self, mocker):
        redis = mocker.MagicMock()
        redis.scan_iter.side_effect = lambda *args, **kwargs: iter([])
        mocker.patch("backend.util.cache._get_redis", return_value=redis)

        @cached(ttl_seconds=30, shared_cache=True)
        def batched_scan_function() -> int:
            return 1

        batched_scan_function.cache_info()
        batched_scan_function.cache_clear()

        batch_sizes = [c.kwargs.get("count") for c in redis.scan_iter.call_args_list]
        assert batch_sizes == [_SCAN_BATCH_SIZE, _SCAN_BATCH_SIZE]

    def test_sync_shared_cache_basic(self):
        """Test basic shared cache functionality with sync function."""
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        def shared_sync_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        # Clear any existing cache
        shared_sync_function.cache_clear()

        # First call
        result1 = shared_sync_function(10, 20)
        assert result1 == 30
        assert call_count == 1

        # Second call - should use Redis cache
        result2 = shared_sync_function(10, 20)
        assert result2 == 30
        assert call_count == 1

        # Different args - should call function again
        result3 = shared_sync_function(15, 25)
        assert result3 == 40
        assert call_count == 2

        # Cleanup
        shared_sync_function.cache_clear()

    @pytest.mark.asyncio
    async def test_async_shared_cache_basic(self):
        """Test basic shared cache functionality with async function."""
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def shared_async_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x + y

        # Clear any existing cache
        await asyncio.to_thread(shared_async_function.cache_clear)

        # First call
        result1 = await shared_async_function(10, 20)
        assert result1 == 30
        assert call_count == 1

        # Second call - should use Redis cache
        result2 = await shared_async_function(10, 20)
        assert result2 == 30
        assert call_count == 1

        # Different args - should call function again
        result3 = await shared_async_function(15, 25)
        assert result3 == 40
        assert call_count == 2

        # Cleanup
        await asyncio.to_thread(shared_async_function.cache_clear)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("refresh", [True, False], ids=["refreshed", "kept"])
    @pytest.mark.parametrize("caller", ["sync", "async"])
    async def test_a_hit_refreshes_its_entrys_ttl_only_when_asked_to(
        self, caller: str, refresh: bool
    ):
        """``refresh_ttl_on_get`` reads an entry with GETEX, which gives it its
        whole TTL again; without it, a hit leaves the TTL running down. The
        TTLs leave half an hour between the two answers, so no pause of the
        runner's can expire the entry or blur them."""
        calls = 0

        def answer(key: str) -> str:
            nonlocal calls
            calls += 1
            return f"answer to {key}"

        if caller == "sync":

            @cached(ttl_seconds=3600, shared_cache=True, refresh_ttl_on_get=refresh)
            def ttl_lookup(key: str) -> str:
                return answer(key)

            async def call(key: str) -> str:
                return await asyncio.to_thread(ttl_lookup, key)

        else:

            @cached(ttl_seconds=3600, shared_cache=True, refresh_ttl_on_get=refresh)
            async def ttl_lookup(key: str) -> str:
                return answer(key)

            async def call(key: str) -> str:
                return await ttl_lookup(key)

        key = _unique("ttl")
        redis_key = cache_key("ttl_lookup", key=key)
        redis = await asyncio.to_thread(cache_module._get_redis)
        try:
            assert await call(key) == f"answer to {key}"
            # As if half its TTL had run down.
            await asyncio.to_thread(redis.expire, redis_key, 1800)
            assert await call(key) == f"answer to {key}"
            ttl = cast(int, await asyncio.to_thread(redis.ttl, redis_key))
        finally:
            await asyncio.to_thread(redis.delete, redis_key)

        assert calls == 1
        if refresh:
            assert ttl > 1800
        else:
            assert 0 < ttl <= 1800

    def test_shared_cache_complex_objects(self):
        """Test caching complex objects with shared cache (pickle serialization)."""
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        def complex_object_function(x: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {
                "number": x,
                "squared": x**2,
                "nested": {"list": [1, 2, x], "tuple": (x, x * 2)},
                "string": f"value_{x}",
            }

        # Clear any existing cache
        complex_object_function.cache_clear()

        # First call
        result1 = complex_object_function(5)
        assert result1["number"] == 5
        assert result1["squared"] == 25
        assert result1["nested"]["list"] == [1, 2, 5]
        assert call_count == 1

        # Second call - should use cache
        result2 = complex_object_function(5)
        assert result2 == result1
        assert call_count == 1

        # Cleanup
        complex_object_function.cache_clear()

    def test_shared_cache_info(self):
        """Test cache_info for shared cache."""

        @cached(ttl_seconds=30, shared_cache=True)
        def info_shared_function(x: int) -> int:
            return x * 2

        # Clear any existing cache
        info_shared_function.cache_clear()

        # Check initial info
        info = info_shared_function.cache_info()
        assert info["size"] == 0
        assert info["maxsize"] is None  # Redis manages size
        assert info["ttl_seconds"] == 30

        # Add some entries
        info_shared_function(1)
        info_shared_function(2)
        info_shared_function(3)

        info = info_shared_function.cache_info()
        assert info["size"] == 3

        # Cleanup
        info_shared_function.cache_clear()

    def test_shared_cache_delete(self):
        """Test selective deletion with shared cache."""
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        def delete_shared_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 3

        # Clear any existing cache
        delete_shared_function.cache_clear()

        # Add entries
        delete_shared_function(1)
        delete_shared_function(2)
        delete_shared_function(3)
        assert call_count == 3

        # Verify cached
        delete_shared_function(1)
        delete_shared_function(2)
        assert call_count == 3

        # Delete specific entry
        was_deleted = delete_shared_function.cache_delete(2)
        assert was_deleted is True

        # Entry for x=2 should be gone
        delete_shared_function(2)
        assert call_count == 4

        # Others should still be cached
        delete_shared_function(1)
        delete_shared_function(3)
        assert call_count == 4

        # Try to delete non-existent
        was_deleted = delete_shared_function.cache_delete(99)
        assert was_deleted is False

        # Cleanup
        delete_shared_function.cache_clear()

    @pytest.mark.asyncio
    async def test_async_shared_cache_thundering_herd(self):
        """Test that shared cache prevents thundering herd for async functions."""
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def shared_slow_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * x

        # Clear any existing cache
        await asyncio.to_thread(shared_slow_function.cache_clear)

        # Launch multiple concurrent tasks
        tasks = [shared_slow_function(8) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return same result
        assert all(r == 64 for r in results)
        # Only one should have executed
        assert call_count == 1

        # Cleanup
        await asyncio.to_thread(shared_slow_function.cache_clear)

    def test_shared_cache_clear_pattern(self):
        """Test pattern-based cache clearing (Redis feature)."""

        @cached(ttl_seconds=30, shared_cache=True)
        def pattern_function(category: str, item: int) -> str:
            return f"{category}_{item}"

        # Clear any existing cache
        pattern_function.cache_clear()

        # Add various entries
        pattern_function("fruit", 1)
        pattern_function("fruit", 2)
        pattern_function("vegetable", 1)
        pattern_function("vegetable", 2)

        info = pattern_function.cache_info()
        assert info["size"] == 4

        # Note: Pattern clearing with wildcards requires specific Redis scan
        # implementation. The current code clears by pattern but needs
        # adjustment for partial matching. For now, test full clear.
        pattern_function.cache_clear()
        info = pattern_function.cache_info()
        assert info["size"] == 0

    def test_shared_vs_local_cache_isolation(self):
        """Test that shared and local caches are isolated."""
        shared_count = 0
        local_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        def shared_function(x: int) -> int:
            nonlocal shared_count
            shared_count += 1
            return x * 2

        @cached(ttl_seconds=30, shared_cache=False)
        def local_function(x: int) -> int:
            nonlocal local_count
            local_count += 1
            return x * 2

        # Clear caches
        shared_function.cache_clear()
        local_function.cache_clear()

        # Call both with same args
        shared_result = shared_function(5)
        local_result = local_function(5)

        assert shared_result == local_result == 10
        assert shared_count == 1
        assert local_count == 1

        # Call again - both should use their respective caches
        shared_function(5)
        local_function(5)
        assert shared_count == 1
        assert local_count == 1

        # Clear only shared cache
        shared_function.cache_clear()

        # Shared should recompute, local should still use cache
        shared_function(5)
        local_function(5)
        assert shared_count == 2
        assert local_count == 1

        # Cleanup
        shared_function.cache_clear()
        local_function.cache_clear()

    @pytest.mark.asyncio
    async def test_shared_cache_concurrent_different_keys(self):
        """Test that concurrent calls with different keys work correctly."""
        call_counts = {}

        @cached(ttl_seconds=30, shared_cache=True)
        async def multi_key_function(key: str) -> str:
            if key not in call_counts:
                call_counts[key] = 0
            call_counts[key] += 1
            await asyncio.sleep(0.05)
            return f"result_{key}"

        # Clear cache
        await asyncio.to_thread(multi_key_function.cache_clear)

        # Launch concurrent tasks with different keys
        keys = ["a", "b", "c", "d", "e"]
        tasks = []
        for key in keys:
            # Multiple calls per key
            tasks.extend([multi_key_function(key) for _ in range(3)])

        results = await asyncio.gather(*tasks)

        # Verify results
        for i, key in enumerate(keys):
            expected = f"result_{key}"
            # Each key appears 3 times in results
            key_results = results[i * 3 : (i + 1) * 3]
            assert all(r == expected for r in key_results)

        # Each key should only be computed once
        for key in keys:
            assert call_counts[key] == 1

        # Cleanup
        await asyncio.to_thread(multi_key_function.cache_clear)

    def test_shared_cache_performance_comparison(self):
        """Compare performance of shared vs local cache."""
        import statistics

        shared_times = []
        local_times = []

        @cached(ttl_seconds=30, shared_cache=True)
        def shared_perf_function(x: int) -> int:
            time.sleep(0.01)  # Simulate work
            return x * 2

        @cached(ttl_seconds=30, shared_cache=False)
        def local_perf_function(x: int) -> int:
            time.sleep(0.01)  # Simulate work
            return x * 2

        # Clear caches
        shared_perf_function.cache_clear()
        local_perf_function.cache_clear()

        # Warm up both caches
        for i in range(5):
            shared_perf_function(i)
            local_perf_function(i)

        # Measure cache hit times
        for i in range(5):
            # Shared cache hit
            start = time.time()
            shared_perf_function(i)
            shared_times.append(time.time() - start)

            # Local cache hit
            start = time.time()
            local_perf_function(i)
            local_times.append(time.time() - start)

        # Local cache should be faster (no Redis round-trip)
        avg_shared = statistics.mean(shared_times)
        avg_local = statistics.mean(local_times)

        print(f"Avg shared cache hit time: {avg_shared:.6f}s")
        print(f"Avg local cache hit time: {avg_local:.6f}s")

        # Local should be significantly faster for cache hits
        # Redis adds network latency even for cache hits
        assert avg_local < avg_shared

        # Cleanup
        shared_perf_function.cache_clear()
        local_perf_function.cache_clear()


class TestCacheHMAC:
    """Tests for HMAC integrity verification on Redis-backed cache."""

    def test_hmac_signed_roundtrip(self):
        """Values written to Redis can be read back via HMAC verification."""
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        def hmac_roundtrip_fn(x: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {"value": x, "nested": [1, 2, 3]}

        hmac_roundtrip_fn.cache_clear()

        result1 = hmac_roundtrip_fn(42)
        assert result1 == {"value": 42, "nested": [1, 2, 3]}
        assert call_count == 1

        # Second call should hit cache (HMAC verification passes)
        result2 = hmac_roundtrip_fn(42)
        assert result2 == {"value": 42, "nested": [1, 2, 3]}
        assert call_count == 1

        hmac_roundtrip_fn.cache_clear()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entry", ["unsigned", "tampered", "signed-unreadable"])
    @pytest.mark.parametrize("caller", ["sync", "async"])
    async def test_an_entry_that_does_not_decode_is_computed_again_and_replaced(
        self, caller: str, entry: str
    ):
        """An entry without a valid signature (a legacy one, or one someone
        changed), or whose signed payload doesn't unpickle, counts as a miss:
        the value is computed again and stored in its place."""
        calls = 0

        def answer(key: str) -> str:
            nonlocal calls
            calls += 1
            return f"answer to {key}"

        if caller == "sync":

            @cached(ttl_seconds=30, shared_cache=True)
            def decoded_lookup(key: str) -> str:
                return answer(key)

            async def call(key: str) -> str:
                return await asyncio.to_thread(decoded_lookup, key)

        else:

            @cached(ttl_seconds=30, shared_cache=True)
            async def decoded_lookup(key: str) -> str:
                return answer(key)

            async def call(key: str) -> str:
                return await decoded_lookup(key)

        key = _unique("decode")
        redis_key = cache_key("decoded_lookup", key=key)
        redis = await asyncio.to_thread(cache_module._get_redis)
        try:
            assert await call(key) == f"answer to {key}"
            stored = cast(bytes, await asyncio.to_thread(redis.get, redis_key))
            undecodable = {
                "unsigned": pickle.dumps("not the answer"),
                "tampered": bytes([stored[0] ^ 0xFF]) + stored[1:],
                "signed-unreadable": cache_module._sign_payload(b"not a pickle"),
            }[entry]
            await asyncio.to_thread(redis.set, redis_key, undecodable, ex=30)

            assert await call(key) == f"answer to {key}"
            assert calls == 2
            # The value computed again replaced the entry.
            assert await call(key) == f"answer to {key}"
            assert calls == 2
        finally:
            await asyncio.to_thread(redis.delete, redis_key)


class TestCacheNoneHandling:
    """Tests for the ``cache_none`` parameter on the @cached decorator.

    Sentry bug PRRT_kwDOJKSTjM56RTEu (HIGH): the cache previously could not
    distinguish "no entry" from "entry is None", so any function returning
    ``None`` was effectively re-executed on every call. The fix is a
    sentinel-based check inside the wrappers, plus an opt-out
    ``cache_none=False`` flag for callers that *want* errors to retry.
    """

    @pytest.mark.asyncio
    async def test_async_none_is_cached_by_default(self):
        """With ``cache_none=True`` (default), cached ``None`` is returned
        from the cache instead of triggering re-execution."""
        call_count = 0

        @cached(ttl_seconds=300)
        async def maybe_none(x: int) -> int | None:
            nonlocal call_count
            call_count += 1
            return None

        assert await maybe_none(1) is None
        assert call_count == 1

        # Second call should hit the cache, not re-execute.
        assert await maybe_none(1) is None
        assert call_count == 1

        # Different argument is a different cache key — re-executes.
        assert await maybe_none(2) is None
        assert call_count == 2

    def test_sync_none_is_cached_by_default(self):
        call_count = 0

        @cached(ttl_seconds=300)
        def maybe_none(x: int) -> int | None:
            nonlocal call_count
            call_count += 1
            return None

        assert maybe_none(1) is None
        assert maybe_none(1) is None
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_cache_none_false_skips_storing_none(self):
        """``cache_none=False`` skips storing ``None`` so transient errors
        are retried on the next call instead of poisoning the cache."""
        call_count = 0
        results: list[int | None] = [None, None, 42]

        @cached(ttl_seconds=300, cache_none=False)
        async def maybe_none(x: int) -> int | None:
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        # First call: returns None, NOT stored.
        assert await maybe_none(1) is None
        assert call_count == 1

        # Second call with same key: re-executes (None wasn't cached).
        assert await maybe_none(1) is None
        assert call_count == 2

        # Third call: returns 42, this time it IS stored.
        assert await maybe_none(1) == 42
        assert call_count == 3

        # Fourth call: cache hit on the stored 42.
        assert await maybe_none(1) == 42
        assert call_count == 3

    def test_sync_cache_none_false_skips_storing_none(self):
        call_count = 0
        results: list[int | None] = [None, 99]

        @cached(ttl_seconds=300, cache_none=False)
        def maybe_none(x: int) -> int | None:
            nonlocal call_count
            result = results[call_count]
            call_count += 1
            return result

        assert maybe_none(1) is None
        assert call_count == 1

        # None was not stored — re-executes.
        assert maybe_none(1) == 99
        assert call_count == 2

        # 99 IS stored — no re-execution.
        assert maybe_none(1) == 99
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_shared_cache_none_is_cached_by_default(self):
        """Shared (Redis) cache also properly returns cached ``None`` values."""
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def maybe_none_redis(x: int) -> int | None:
            nonlocal call_count
            call_count += 1
            return None

        await asyncio.to_thread(maybe_none_redis.cache_clear)

        assert await maybe_none_redis(1) is None
        assert call_count == 1

        assert await maybe_none_redis(1) is None
        assert call_count == 1

        await asyncio.to_thread(maybe_none_redis.cache_clear)


class TestCacheKeyBinding:
    """The service RPC calls cached functions by keyword while invalidation
    call sites pass arguments positionally; both must reach one entry."""

    @pytest.mark.parametrize("shared_cache", [False, True])
    def test_delete_reaches_an_entry_cached_by_another_call_style(
        self, shared_cache: bool
    ):
        call_count = 0

        @cached(ttl_seconds=30, shared_cache=shared_cache)
        def lookup(user_id: str, include_org: bool = False) -> str:
            nonlocal call_count
            call_count += 1
            return f"{user_id}:{include_org}"

        lookup.cache_clear()
        lookup(user_id="u1")
        lookup("u1")
        lookup("u1", False)
        assert call_count == 1

        assert lookup.cache_delete("u1") is True
        lookup(user_id="u1")
        assert call_count == 2

        assert lookup.cache_delete(user_id="u1", include_org=False) is True
        lookup("u1")
        assert call_count == 3

        lookup("u1", include_org=True)
        assert call_count == 4
        lookup.cache_clear()

    @pytest.mark.parametrize("shared_cache", [False, True])
    def test_a_call_that_does_not_bind_raises_instead_of_reading_the_cache(
        self, shared_cache: bool
    ):
        @cached(ttl_seconds=30, shared_cache=shared_cache)
        def lookup(user_id: str, /) -> str:
            return user_id

        lookup.cache_clear()
        assert lookup("u1") == "u1"

        # Keyed by name, this call would match the entry ``lookup("u1")`` stored.
        by_name: dict[str, Any] = {"user_id": "u1"}
        with pytest.raises(TypeError):
            lookup(**by_name)
        with pytest.raises(TypeError):
            lookup.cache_delete(**by_name)
        lookup.cache_clear()


async def _stop_and_join(
    loop: asyncio.AbstractEventLoop, worker: threading.Thread
) -> None:
    """Stop the loop ``worker`` runs and close it once the thread has ended,
    whatever its ``shutdown_asyncgens()`` does on the way."""
    try:
        closed = asyncio.run_coroutine_threadsafe(loop.shutdown_asyncgens(), loop)
        async with asyncio.timeout(_HANG_SECONDS):
            await asyncio.wrap_future(closed)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        await asyncio.to_thread(worker.join, _HANG_SECONDS)
        if not worker.is_alive():
            loop.close()


def _unique(name: str) -> str:
    # The shared cache is the real Redis, which later runs share.
    return f"{name}-{uuid.uuid4().hex}"


def _a_key_off_the_first_node(function_name: str) -> str:
    """An argument whose cache key lives on a node other than the one the
    client starts from, when the cluster has more than one."""
    redis = cache_module._get_redis()
    first_port = cache_module.settings.config.redis_port
    key = _unique("routed")
    for _ in range(200):
        if (
            redis.get_node_from_key(cache_key(function_name, key=key)).port
            != first_port
        ):
            break
        key = _unique("routed")
    return key


class TestSharedCacheOnAnEventLoop:
    """A coroutine waiting on Redis is one coroutine waiting: the loop runs
    the others, a read Redis doesn't answer costs a call the cache's bound,
    and a slow computation holds up only the callers of its own key."""

    @pytest.mark.asyncio
    async def test_the_loop_runs_other_coroutines_while_redis_answers_one(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        @cached(ttl_seconds=30, shared_cache=True)
        async def held_lookup(question: str) -> str:
            return f"answer to {question}"

        question = _unique("loop")
        # Held until released below, so no bound may end the wait first.
        monkeypatch.setattr(cache_module, "REDIS_OP_TIMEOUT_S", 3 * _HANG_SECONDS)
        held = hold_cache_read(
            monkeypatch, cache_key("held_lookup", question=question), _HANG_SECONDS
        )
        call = asyncio.create_task(held_lookup(question))
        try:
            async with asyncio.timeout(2 * _HANG_SECONDS):
                await held.started()
            # This coroutine ran while the read waited on Redis.
            assert not held.answered
            held.release()
            async with asyncio.timeout(_HANG_SECONDS):
                assert await call == f"answer to {question}"
        finally:
            held.release()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(call, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_a_read_redis_does_not_answer_in_time_is_computed_instead(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def unanswered_lookup(question: str) -> str:
            nonlocal calls
            calls += 1
            return f"answer to {question}"

        question = _unique("unanswered")
        held = hold_cache_read(
            monkeypatch,
            cache_key("unanswered_lookup", question=question),
            _HANG_SECONDS,
        )
        # The loop's client reads the slot map under the usual bound first, so
        # only the held read is in the short one.
        await cache_module._get_async_redis()
        # Redis never answers this read, so the bound decides when the call
        # stops waiting for it, not whether.
        monkeypatch.setattr(cache_module, "REDIS_OP_TIMEOUT_S", 0.05)
        try:
            async with asyncio.timeout(2 * _HANG_SECONDS):
                assert await unanswered_lookup(question) == f"answer to {question}"
            assert not held.answered
            assert calls == 1
            # One wait, not three: no second read, and no write to wait on.
            assert held.commands == ["GET"]
        finally:
            held.release()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shared_cache", [False, True], ids=["memory", "redis"])
    async def test_a_slow_miss_holds_up_only_callers_of_its_own_key(
        self, shared_cache: bool
    ):
        computing, finish = asyncio.Event(), asyncio.Event()

        @cached(ttl_seconds=30, shared_cache=shared_cache)
        async def lookup(key: str) -> str:
            if key.startswith("slow"):
                computing.set()
                await finish.wait()
            return key

        slow_key, fast_key = _unique("slow"), _unique("fast")
        slow = asyncio.create_task(lookup(slow_key))
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await computing.wait()
                assert await lookup(fast_key) == fast_key
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                assert await slow == slow_key

    @pytest.mark.asyncio
    @_ON_EVERY_KIND_OF_LOOP
    async def test_another_event_loop_reads_the_cache_through_a_client_of_its_own(
        self, new_loop: NewLoop
    ):
        calls = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            nonlocal calls
            calls += 1
            return key

        key = _unique("loops")
        assert await lookup(key) == key
        # A loop on a thread of its own, as each copilot worker runs one.
        other = new_loop()
        worker = threading.Thread(target=other.run_forever, daemon=True)
        worker.start()
        read = asyncio.run_coroutine_threadsafe(lookup(key), other)
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                assert await asyncio.wrap_future(read) == key
        finally:
            read.cancel()
            await _stop_and_join(other, worker)
        # The other loop read the entry this one stored.
        assert calls == 1

    @pytest.mark.asyncio
    @_ON_EVERY_KIND_OF_LOOP
    async def test_closing_a_loop_closes_its_client(self, new_loop: NewLoop):
        """However a loop is closed, its client is closed first: here as a
        retired direct-block-execution worker's is, with ``loop.close()`` and
        no ``shutdown_asyncgens()``. Nothing could close it after that."""

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        async def read_and_get_client():
            await lookup(_unique("retired"))
            return await cache_module._get_async_redis()

        def retire_a_loop_after_one_read() -> bool:
            loop = new_loop()

            def serve():
                loop.run_forever()
                loop.close()

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            read = asyncio.run_coroutine_threadsafe(read_and_get_client(), loop)
            try:
                client = read.result(_HANG_SECONDS)
            finally:
                read.cancel()
                loop.call_soon_threadsafe(loop.stop)
                worker.join(_HANG_SECONDS)
            assert loop.is_closed()
            return is_open(client)

        left_open = [
            await asyncio.to_thread(retire_a_loop_after_one_read) for _ in range(3)
        ]

        assert left_open == [False] * 3

    @pytest.mark.asyncio
    @_ON_EVERY_KIND_OF_LOOP
    async def test_a_fresh_loops_first_read_goes_straight_to_its_keys_shard(
        self, new_loop: NewLoop, monkeypatch: pytest.MonkeyPatch
    ):
        """redis-py's cluster client sends a read it gets before reading the
        slot map to a random primary, which redirects it, so each loop's client
        reads the map first. A random primary owns the key a third of the time
        on CI's cluster, so several fresh loops read."""

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        redirects: list[Exception] = []
        move_slot = async_cluster.NodesManager.move_slot

        async def recorded_move_slot(self, e):
            redirects.append(e)
            return await move_slot(self, e)

        monkeypatch.setattr(async_cluster.NodesManager, "move_slot", recorded_move_slot)

        def read_on_a_fresh_loop(key: str) -> str:
            loop = new_loop()
            try:
                return loop.run_until_complete(lookup(key))
            finally:
                loop.close()

        for _ in range(8):
            key = await asyncio.to_thread(_a_key_off_the_first_node, "lookup")
            assert await asyncio.to_thread(read_on_a_fresh_loop, key) == key
        assert redirects == []

    @pytest.mark.asyncio
    @_ON_EVERY_KIND_OF_LOOP
    async def test_closing_a_loop_closes_its_client_after_a_connection_error(
        self, new_loop: NewLoop, monkeypatch: pytest.MonkeyPatch
    ):
        """A connection error leaves redis-py's client due to read the slot
        map again, and its ``aclose()`` then skips the client, connections and
        all. Closing the loop must still close them."""

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        key = _unique("connection-lost")
        redis_key = cache_key("lookup", key=key)
        execute = async_cluster.ClusterNode.execute_command

        async def connection_lost(self, *args, **kwargs):
            if args[1:2] == (redis_key,):
                raise RedisConnectionError("connection lost")
            return await execute(self, *args, **kwargs)

        monkeypatch.setattr(
            async_cluster.ClusterNode, "execute_command", connection_lost
        )
        created: list[async_cluster.RedisCluster] = []
        new_client = cache_redis._new_async_redis

        def remember_new_client() -> async_cluster.RedisCluster:
            created.append(new_client())
            return created[-1]

        monkeypatch.setattr(cache_redis, "_new_async_redis", remember_new_client)

        def read_on_a_loop_then_close_it() -> str:
            loop = new_loop()
            try:
                return loop.run_until_complete(lookup(key))
            finally:
                loop.close()

        assert await asyncio.to_thread(read_on_a_loop_then_close_it) == key
        [client] = created
        assert not is_open(client)

    @pytest.mark.asyncio
    async def test_a_write_redis_does_not_answer_in_time_is_given_up(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        calls = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def unstored_lookup(question: str) -> str:
            nonlocal calls
            calls += 1
            # Redis answers the reads; only the write that follows gets a
            # short bound, which decides when the call stops waiting for it.
            monkeypatch.setattr(cache_module, "REDIS_OP_TIMEOUT_S", 0.05)
            return f"answer to {question}"

        question = _unique("unstored")
        held = hold_cache_read(
            monkeypatch,
            cache_key("unstored_lookup", question=question),
            _HANG_SECONDS,
            commands=("SET",),
        )
        try:
            async with asyncio.timeout(2 * _HANG_SECONDS):
                assert await unstored_lookup(question) == f"answer to {question}"
            assert not held.answered
            assert held.commands == ["GET", "GET", "SET"]
            assert calls == 1
        finally:
            held.release()

    @pytest.mark.asyncio
    async def test_a_process_out_of_redis_connections_computes_with_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A burst of commands past the client's connection cap is a moment's
        overload, as a timeout is: a warning, not an error for Sentry."""

        async def out_of_connections():
            raise MaxConnectionsError("Too many connections")

        monkeypatch.setattr(cache_module, "_get_async_redis", out_of_connections)
        calls = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def crowded_lookup(key: str) -> str:
            nonlocal calls
            calls += 1
            return key

        key = _unique("crowded")
        with caplog.at_level(logging.WARNING, logger=cache_module.logger.name):
            assert await crowded_lookup(key) == key

        assert calls == 1
        assert [
            record.levelno
            for record in caplog.records
            if record.name == cache_module.logger.name
        ] == [logging.WARNING]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shared_cache", [False, True], ids=["memory", "redis"])
    async def test_callers_missing_a_key_share_a_result_that_is_not_stored(
        self, shared_cache: bool, monkeypatch: pytest.MonkeyPatch
    ):
        """``cache_none=False`` stores no ``None``. The callers that missed the
        key while it was computed get it all the same, instead of computing
        it again one after another."""
        everyone_in = callers_in_flight(monkeypatch, 3)
        finish = asyncio.Event()
        calls = 0

        @cached(ttl_seconds=30, shared_cache=shared_cache, cache_none=False)
        async def unstored_lookup(key: str) -> None:
            nonlocal calls
            calls += 1
            await finish.wait()

        key = _unique("unstored-none")
        callers = [asyncio.create_task(unstored_lookup(key)) for _ in range(3)]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await everyone_in.wait()
                finish.set()
                assert await asyncio.gather(*callers) == [None] * 3
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert calls == 1

    @pytest.mark.asyncio
    async def test_callers_missing_a_key_share_one_computation_while_redis_is_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Nothing is stored while Redis can't be reached, so without sharing,
        each caller would compute the value in turn. Each gets a copy of its
        own, as from Redis."""

        async def unreachable():
            raise RedisConnectionError("Redis is unreachable")

        monkeypatch.setattr(cache_module, "_get_async_redis", unreachable)
        everyone_in = callers_in_flight(monkeypatch, 3)
        finish = asyncio.Event()
        calls = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def unreachable_lookup(key: str) -> dict[str, str]:
            nonlocal calls
            calls += 1
            await finish.wait()
            return {"answer": key}

        key = _unique("unreachable")
        callers = [asyncio.create_task(unreachable_lookup(key)) for _ in range(3)]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await everyone_in.wait()
                finish.set()
                results = await asyncio.gather(*callers)
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert results == [{"answer": key}] * 3
        assert len({id(result) for result in results}) == 3
        assert calls == 1

    @pytest.mark.asyncio
    async def test_an_error_reaches_every_caller_that_waited_for_its_computation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        everyone_in = callers_in_flight(monkeypatch, 3)
        finish = asyncio.Event()
        calls = 0

        @cached(ttl_seconds=30)
        async def failing_lookup(key: str) -> str:
            nonlocal calls
            calls += 1
            await finish.wait()
            raise ValueError(f"no answer to {key}")

        key = _unique("failing")
        callers = [asyncio.create_task(failing_lookup(key)) for _ in range(3)]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await everyone_in.wait()
                finish.set()
                results = await asyncio.gather(*callers, return_exceptions=True)
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert [type(result) for result in results] == [ValueError] * 3
        assert calls == 1

        # An error isn't stored: the next caller computes it again.
        with pytest.raises(ValueError):
            await failing_lookup(key)
        assert calls == 2

    @pytest.mark.asyncio
    async def test_a_caller_waiting_on_a_cancelled_computation_computes_it_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        everyone_in = callers_in_flight(monkeypatch, 2)
        computing, finish = asyncio.Event(), asyncio.Event()
        calls = 0

        @cached(ttl_seconds=30)
        async def lookup(key: str) -> str:
            nonlocal calls
            calls += 1
            computing.set()
            await finish.wait()
            return key

        key = _unique("handed-over")
        leader = asyncio.create_task(lookup(key))
        callers = [leader]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await computing.wait()
                follower = asyncio.create_task(lookup(key))
                callers.append(follower)
                await everyone_in.wait()
                computing.clear()
                leader.cancel()
                # The waiting caller computes it now.
                await computing.wait()
                finish.set()
                assert await follower == key
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert leader.cancelled()
        assert calls == 2

    @pytest.mark.asyncio
    async def test_a_caller_cancelled_along_with_its_keys_computation_stays_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """As when one task group cancels both callers: the waiting one must
        not take the computation over."""
        everyone_in = callers_in_flight(monkeypatch, 2)
        computing, finish = asyncio.Event(), asyncio.Event()
        calls = 0

        @cached(ttl_seconds=30)
        async def lookup(key: str) -> str:
            nonlocal calls
            calls += 1
            computing.set()
            await finish.wait()
            return key

        key = _unique("cancelled-together")
        leader = asyncio.create_task(lookup(key))
        callers = [leader]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await computing.wait()
                follower = asyncio.create_task(lookup(key))
                callers.append(follower)
                await everyone_in.wait()
                # The computing caller first, so the waiting one finds the
                # computation cancelled when its own cancellation reaches it.
                leader.cancel()
                follower.cancel()
                finish.set()
                results = await asyncio.gather(*callers, return_exceptions=True)
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert [type(result) for result in results] == [asyncio.CancelledError] * 2
        assert calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ending", ["returns", "raises", "is_cancelled"])
    async def test_a_keys_computation_is_forgotten_once_it_ends(self, ending: str):
        """One entry per key ever computed would grow without end."""
        in_flight = cache_module._InFlight(cache_module.MAX_CONCURRENT_MISSES)
        computing, finish = asyncio.Event(), asyncio.Event()

        async def compute() -> str:
            computing.set()
            await finish.wait()
            if ending == "raises":
                raise ValueError("no answer")
            return "answer"

        def caller() -> asyncio.Task[str]:
            return asyncio.create_task(
                in_flight.run("key", compute, cache_module._the_same)
            )

        leader = caller()
        follower: asyncio.Task[str] | None = None
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await computing.wait()
                follower = caller()
                await asyncio.sleep(0)
                # It waits for the leader's computation: started, and suspended.
                waiting = follower.get_coro()
                assert waiting is not None
                assert inspect.getcoroutinestate(waiting) == inspect.CORO_SUSPENDED
                follower.cancel()
                if ending == "is_cancelled":
                    leader.cancel()
                finish.set()
                await asyncio.gather(leader, follower, return_exceptions=True)
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(
                    leader, *([follower] if follower else []), return_exceptions=True
                )
        assert follower.cancelled()
        assert in_flight._flights == {}

    @pytest.mark.asyncio
    @_ON_EVERY_KIND_OF_LOOP
    async def test_a_fresh_loops_concurrent_first_reads_share_one_client(
        self, new_loop: NewLoop, monkeypatch: pytest.MonkeyPatch
    ):
        """The first callers on a new loop all wait for the client the first
        of them made, rather than each making one and leaving the rest open."""
        created: list[async_cluster.RedisCluster] = []
        new_client = cache_redis._new_async_redis

        def remember_new_client() -> async_cluster.RedisCluster:
            created.append(new_client())
            return created[-1]

        monkeypatch.setattr(cache_redis, "_new_async_redis", remember_new_client)

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        keys = [_unique("first") for _ in range(8)]

        async def first_reads() -> tuple[list[str], list[Any]]:
            values = asyncio.gather(*(lookup(key) for key in keys))
            clients = asyncio.gather(*(cache_module._get_async_redis() for _ in keys))
            return await values, await clients

        def on_a_fresh_loop() -> tuple[list[str], list[Any]]:
            loop = new_loop()
            try:
                return loop.run_until_complete(first_reads())
            finally:
                loop.close()

        values, clients = await asyncio.to_thread(on_a_fresh_loop)
        assert values == keys
        [client] = created
        assert all(each is client for each in clients)
        assert not is_open(client)

    @pytest.mark.asyncio
    async def test_misses_of_different_keys_compute_at_most_the_cap_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A burst of distinct keys can't flood what the function calls: the
        misses past ``max_concurrent_misses`` wait for a turn. In memory, where
        no read suspends, so every miss has reached its turn or its
        computation before the test looks. The store search test covers the
        shared cache."""
        everyone_in = callers_in_flight(monkeypatch, 5)
        two_running, finish = asyncio.Event(), asyncio.Event()
        running = peak = 0

        @cached(ttl_seconds=30, max_concurrent_misses=2)
        async def capped_lookup(key: str) -> str:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            if running == 2:
                two_running.set()
            await finish.wait()
            running -= 1
            return key

        keys = [_unique("capped") for _ in range(5)]
        callers = [asyncio.create_task(capped_lookup(key)) for key in keys]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await everyone_in.wait()
                await two_running.wait()
                # Every caller has missed: two compute, three wait their turn.
                assert running == 2
                finish.set()
                assert await asyncio.gather(*callers) == keys
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert peak == 2

    @pytest.mark.asyncio
    async def test_cancelling_a_call_waiting_on_redis_cancels_it(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A cancellation that reaches a call while Redis holds its read ends
        the call, rather than counting as a cache error to compute through."""
        calls = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def held_lookup(question: str) -> str:
            nonlocal calls
            calls += 1
            return question

        question = _unique("cancelled")
        monkeypatch.setattr(cache_module, "REDIS_OP_TIMEOUT_S", 3 * _HANG_SECONDS)
        held = hold_cache_read(
            monkeypatch, cache_key("held_lookup", question=question), _HANG_SECONDS
        )
        call = asyncio.create_task(held_lookup(question))
        try:
            async with asyncio.timeout(2 * _HANG_SECONDS):
                await held.started()
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                async with asyncio.timeout(_HANG_SECONDS):
                    await call
            assert calls == 0
        finally:
            held.release()

    @pytest.mark.asyncio
    async def test_a_write_that_fails_does_not_fail_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """As during a failover, when a shard answers writes with READONLY."""

        @cached(ttl_seconds=30, shared_cache=True)
        async def unwritable_lookup(question: str) -> str:
            return question

        execute = async_cluster.RedisCluster.execute_command

        async def read_only(self, *args, **kwargs):
            if args and args[0] == "SET":
                raise ReadOnlyError("You can't write against a read only replica.")
            return await execute(self, *args, **kwargs)

        monkeypatch.setattr(async_cluster.RedisCluster, "execute_command", read_only)
        question = _unique("read-only")
        assert await unwritable_lookup(question) == question

    @pytest.mark.asyncio
    @_ON_EVERY_KIND_OF_LOOP
    async def test_a_closed_loop_is_not_kept_alive_by_misses_that_waited_their_turn(
        self, new_loop: NewLoop
    ):
        """A miss that waits for a turn binds the turn's semaphore to its loop.
        The coordinator that holds it lives on the loop, so once the loop
        closes, nothing of the cache keeps it alive."""

        @cached(ttl_seconds=30, max_concurrent_misses=1)
        async def lookup(key: str) -> str:
            # Lets the other miss reach its turn and wait for it.
            await asyncio.sleep(0)
            return key

        async def misses_that_wait_their_turn() -> list[str]:
            return await asyncio.gather(
                lookup(_unique("first")), lookup(_unique("second"))
            )

        def retire_a_loop() -> weakref.ref[asyncio.AbstractEventLoop]:
            loop = new_loop()
            try:
                assert len(loop.run_until_complete(misses_that_wait_their_turn())) == 2
            finally:
                loop.close()
            return weakref.ref(loop)

        retired = await asyncio.to_thread(retire_a_loop)
        gc.collect()
        assert retired() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("turns_before_the_change", [0, 1])
    async def test_a_waiting_caller_gets_the_value_as_computed_whatever_its_caller_does(
        self, turns_before_the_change: int, monkeypatch: pytest.MonkeyPatch
    ):
        """The computing caller gets the value itself and may change it before
        a waiting caller runs. The waiting one still gets the value as it was
        computed, as Redis holds it."""
        everyone_in = callers_in_flight(monkeypatch, 2)
        computing, finish = asyncio.Event(), asyncio.Event()

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> dict[str, str]:
            computing.set()
            await finish.wait()
            return {"answer": key}

        key = _unique("changed")

        async def caller_that_changes_its_value() -> dict[str, str]:
            value = await lookup(key)
            for _ in range(turns_before_the_change):
                await asyncio.sleep(0)
            value["answer"] = "changed by its caller"
            return value

        computing_caller = asyncio.create_task(caller_that_changes_its_value())
        callers = [computing_caller]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await computing.wait()
                waiting_caller = asyncio.create_task(lookup(key))
                callers.append(waiting_caller)
                await everyone_in.wait()
                finish.set()
                assert await waiting_caller == {"answer": key}
                assert await computing_caller == {"answer": "changed by its caller"}
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_equal_arguments_stored_apart_compute_apart(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """``True == 1``, but the shared cache stores ``f(True)`` and ``f(1)``
        under different keys, so concurrent misses of the two compute apart,
        each with its own argument, rather than share one result."""
        everyone_in = callers_in_flight(monkeypatch, 2)
        computing, finish = asyncio.Event(), asyncio.Event()

        @cached(ttl_seconds=30, shared_cache=True)
        async def describe(token: str, value: object) -> str:
            computing.set()
            await finish.wait()
            return type(value).__name__

        token = _unique("equal")
        first = asyncio.create_task(describe(token, True))
        callers = [first]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await computing.wait()
                second = asyncio.create_task(describe(token, 1))
                callers.append(second)
                await everyone_in.wait()
                finish.set()
                assert await asyncio.gather(first, second) == ["bool", "int"]
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_miss_limit_below_one_is_refused_when_decorating(self, limit: int):
        """No miss could ever start computing. Refused at once, rather than
        found out at the first cold call, which would wait for ever."""
        with pytest.raises(ValueError, match="max_concurrent_misses"):
            cached(ttl_seconds=30, max_concurrent_misses=limit)

    @pytest.mark.asyncio
    async def test_closing_a_client_closes_its_startup_pool_when_its_node_pool_fails_to(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """After a connection error, ``aclose()`` skips the client, and only
        the pools' own closes reach its connections: one failing mustn't spare
        the other."""

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        key = _unique("pools")
        redis_key = cache_key("lookup", key=key)
        execute = async_cluster.ClusterNode.execute_command

        async def connection_lost(self, *args, **kwargs):
            if args[1:2] == (redis_key,):
                raise RedisConnectionError("connection lost")
            return await execute(self, *args, **kwargs)

        monkeypatch.setattr(
            async_cluster.ClusterNode, "execute_command", connection_lost
        )
        close_pool = cache_redis._AsyncNodesManager.aclose

        async def nodes_fail_to_close(self, attr: str = "nodes_cache") -> None:
            if attr == "nodes_cache":
                raise RuntimeError("the nodes failed to close")
            await close_pool(self, attr)

        monkeypatch.setattr(
            cache_redis._AsyncNodesManager, "aclose", nodes_fail_to_close
        )

        def startup_connections(client: Any) -> list[bool]:
            return [
                connection.is_connected
                for node in client.nodes_manager.startup_nodes.values()
                for connection in node._connections
            ]

        def read_on_a_loop_then_close_it() -> tuple[list[bool], list[bool], Any]:
            loop = asyncio.new_event_loop()
            try:
                assert loop.run_until_complete(lookup(key)) == key
                client = loop.run_until_complete(cache_module._get_async_redis())
                before = startup_connections(client)
            finally:
                loop.close()
            return before, startup_connections(client), client

        def forget_the_node_pool(client: Any) -> None:
            # Left open on purpose, its connections belong to the closed loop:
            # closing each raises, and redis-py forgets it all the same.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(close_pool(client.nodes_manager, "nodes_cache"))
            except RuntimeError:
                pass
            finally:
                loop.close()

        before, after, client = await asyncio.to_thread(read_on_a_loop_then_close_it)
        await asyncio.to_thread(forget_the_node_pool, client)
        assert any(before)
        assert not any(after)

    @pytest.mark.asyncio
    async def test_a_write_redis_holds_keeps_no_other_key_from_computing(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A computation holds its turn only while it calls the function, not
        while Redis answers the reads and the write around it: a stalled write
        mustn't keep other keys' misses from the database."""
        monkeypatch.setattr(cache_module, "REDIS_OP_TIMEOUT_S", 3 * _HANG_SECONDS)
        computing: dict[str, asyncio.Event] = {}

        @cached(ttl_seconds=30, shared_cache=True, max_concurrent_misses=1)
        async def lookup(key: str) -> str:
            computing[key].set()
            return key

        held_key, other_key = _unique("held-write"), _unique("other")
        computing.update({held_key: asyncio.Event(), other_key: asyncio.Event()})
        held = hold_cache_read(
            monkeypatch,
            cache_key("lookup", key=held_key),
            _HANG_SECONDS,
            commands=("SET",),
        )
        held_call = asyncio.create_task(lookup(held_key))
        callers = [held_call]
        try:
            async with asyncio.timeout(2 * _HANG_SECONDS):
                await held.started()
                callers.append(asyncio.create_task(lookup(other_key)))
                # The only turn is free while Redis holds the first key's write.
                await computing[other_key].wait()
                assert not held.answered
            held.release()
            async with asyncio.timeout(_HANG_SECONDS):
                assert await asyncio.gather(*callers) == [held_key, other_key]
        finally:
            held.release()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_a_waiting_caller_that_is_cancelled_leaves_the_computation_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A waiting caller's cancellation reaches only that caller: the
        computation it waited on finishes for the caller computing it."""
        everyone_in = callers_in_flight(monkeypatch, 2)
        finish = asyncio.Event()
        calls = 0

        @cached(ttl_seconds=30)
        async def lookup(key: str) -> str:
            nonlocal calls
            calls += 1
            await finish.wait()
            return key

        key = _unique("waiter-cancelled")
        callers = [asyncio.create_task(lookup(key)) for _ in range(2)]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await everyone_in.wait()
                callers[1].cancel()
                # Lets the cancellation reach the waiting caller first.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                finish.set()
                results = await asyncio.gather(*callers, return_exceptions=True)
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert results[0] == key
        assert isinstance(results[1], asyncio.CancelledError)
        assert calls == 1

    @pytest.mark.asyncio
    async def test_an_error_no_caller_waited_for_is_not_reported_as_unretrieved(self):
        """A failed computation's error is marked retrieved, so asyncio doesn't
        log "Future exception was never retrieved" (a Sentry event) for each
        failing call nobody waited on."""
        reported: list[str] = []
        loop = asyncio.get_running_loop()
        handler = loop.get_exception_handler()
        loop.set_exception_handler(
            lambda _loop, context: reported.append(context["message"])
        )

        @cached(ttl_seconds=30)
        async def failing(key: str) -> str:
            raise ValueError(key)

        try:
            for _ in range(3):
                with pytest.raises(ValueError):
                    await failing(_unique("alone"))
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(handler)
        assert reported == []

    @pytest.mark.asyncio
    async def test_a_value_stored_meanwhile_is_returned_instead_of_computed(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Between a miss and its computation, another process may store the
        value: the computation's own read of Redis finds it, and the function
        isn't called."""
        calls = 0

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            nonlocal calls
            calls += 1
            return "computed here"

        key = _unique("meanwhile")
        redis_key = cache_key("lookup", key=key)
        execute = async_cluster.RedisCluster.execute_command
        stored = False

        async def stored_elsewhere_after_the_miss(self, *args, **kwargs):
            nonlocal stored
            result = await execute(self, *args, **kwargs)
            if not stored and args[:2] == ("GET", redis_key):
                stored = True
                payload = cache_module._sign_payload(pickle.dumps("stored elsewhere"))
                await execute(self, "SET", redis_key, payload, "EX", 30)
            return result

        monkeypatch.setattr(
            async_cluster.RedisCluster,
            "execute_command",
            stored_elsewhere_after_the_miss,
        )
        assert await lookup(key) == "stored elsewhere"
        assert calls == 0

    @pytest.mark.asyncio
    async def test_a_value_that_cannot_be_pickled_is_shared_with_waiting_callers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        everyone_in = callers_in_flight(monkeypatch, 2)
        finish = asyncio.Event()
        calls = 0

        class Handle:
            def __init__(self) -> None:
                self.callback = lambda: None  # A lambda doesn't pickle.

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> Handle:
            nonlocal calls
            calls += 1
            await finish.wait()
            return Handle()

        key = _unique("unpicklable")
        callers = [asyncio.create_task(lookup(key)) for _ in range(2)]
        try:
            async with asyncio.timeout(_HANG_SECONDS):
                await everyone_in.wait()
                finish.set()
                first, second = await asyncio.gather(*callers)
        finally:
            finish.set()
            async with asyncio.timeout(_HANG_SECONDS):
                await asyncio.gather(*callers, return_exceptions=True)
        assert isinstance(first, Handle) and isinstance(second, Handle)
        assert calls == 1

    @pytest.mark.asyncio
    async def test_an_error_closing_the_client_does_not_keep_its_loop_open(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        real_close = cache_redis._close

        async def closes_then_fails(client: Any) -> None:
            await real_close(client)
            raise RuntimeError("close failed")

        def read_then_close_a_loop() -> bool:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(lookup(_unique("close-fails")))
                monkeypatch.setattr(cache_redis, "_close", closes_then_fails)
                loop.close()
                return loop.is_closed()
            finally:
                monkeypatch.setattr(cache_redis, "_close", real_close)
                if not loop.is_closed():
                    loop.close()

        with caplog.at_level(logging.WARNING, logger=cache_redis.logger.name):
            assert await asyncio.to_thread(read_then_close_a_loop)
        assert "Could not close the cache's Redis client" in caplog.text

    @pytest.mark.asyncio
    @_ON_EVERY_KIND_OF_LOOP
    async def test_a_loop_closes_even_when_closing_its_client_is_interrupted(
        self, new_loop: NewLoop, monkeypatch: pytest.MonkeyPatch
    ):
        """A ``KeyboardInterrupt``, or any other ``BaseException``, while the
        client closes still lets the loop close."""

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        created: list[async_cluster.RedisCluster] = []
        new_client = cache_redis._new_async_redis

        def remember_new_client() -> async_cluster.RedisCluster:
            created.append(new_client())
            return created[-1]

        monkeypatch.setattr(cache_redis, "_new_async_redis", remember_new_client)

        async def interrupted(client: Any) -> None:
            raise KeyboardInterrupt

        def read_then_close_a_loop() -> bool:
            loop = new_loop()
            loop.run_until_complete(lookup(_unique("interrupted")))
            monkeypatch.setattr(cache_redis, "_close", interrupted)
            with pytest.raises(KeyboardInterrupt):
                loop.close()
            return loop.is_closed()

        try:
            assert await asyncio.to_thread(read_then_close_a_loop)
        finally:
            # Its client was never closed, so its connections are forgotten.
            for client in created:
                await asyncio.to_thread(forget_connections, client)

    @pytest.mark.asyncio
    async def test_closing_a_loop_gives_up_on_a_node_that_will_not_disconnect(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The close runs on every loop close in the process, so a node that
        won't disconnect is given up on within the bound."""

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        created: list[async_cluster.RedisCluster] = []
        new_client = cache_redis._new_async_redis

        def remember_new_client() -> async_cluster.RedisCluster:
            created.append(new_client())
            return created[-1]

        monkeypatch.setattr(cache_redis, "_new_async_redis", remember_new_client)
        disconnect = async_cluster.ClusterNode.disconnect
        stuck = threading.Event()

        async def will_not_disconnect(self, *args, **kwargs):
            stuck.set()
            await asyncio.sleep(3 * _HANG_SECONDS)
            return await disconnect(self, *args, **kwargs)

        def read_on_a_new_loop() -> asyncio.AbstractEventLoop:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(lookup(_unique("stuck")))
            return loop

        loop = await asyncio.to_thread(read_on_a_new_loop)
        monkeypatch.setattr(cache_redis, "REDIS_OP_TIMEOUT_S", 0.05)
        monkeypatch.setattr(
            async_cluster.ClusterNode, "disconnect", will_not_disconnect
        )
        closing = threading.Thread(target=loop.close, daemon=True)
        closing.start()
        await asyncio.to_thread(closing.join, _HANG_SECONDS)
        finished = not closing.is_alive()
        monkeypatch.setattr(async_cluster.ClusterNode, "disconnect", disconnect)
        await asyncio.to_thread(closing.join, 4 * _HANG_SECONDS)
        # The node never disconnected, so its connections are forgotten.
        for client in created:
            await asyncio.to_thread(forget_connections, client)
        assert finished and stuck.is_set()


class TestSharedCacheFollowsAShardsRedirect:
    """A shard asked about a key whose slot it doesn't serve names the one
    that does, by the address that shard announces: on CI's cluster, a
    hostname only the cluster's docker network resolves. The clients reach
    every shard at the address ``address_remap`` gives, redirects included.
    Needs a cluster of several shards, as CI's."""

    def test_redis_py_is_the_release_the_redirect_subclasses_were_written_for(self):
        """``cache_redis``'s redirect subclasses override redis-py 8.1's
        ``NodesManager.move_slot`` and ``RedisCluster.get_node``, and
        ``reinitialize_steps=0`` stands in for its MOVED handling. Another
        release may change all three."""
        assert redis.__version__.startswith("8.1."), (
            f"redis-py is {redis.__version__}. Recheck the redirect subclasses in "
            "backend/util/cache_redis.py against it, and drop them if it includes "
            "redis/redis-py#4522; then update this test."
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("redirect", ["MOVED", "ASK"])
    @pytest.mark.parametrize("caller", ["sync", "async"])
    async def test_a_redirect_is_followed_to_the_shards_remapped_address(
        self, caller: str, redirect: str, monkeypatch: pytest.MonkeyPatch
    ):
        sync_redis = await asyncio.to_thread(cache_module._get_redis)
        primaries = sync_redis.get_primaries()
        if len(primaries) < 2:
            pytest.skip("A redirect needs a cluster of several shards, as CI's")
        calls = 0

        def answer(key: str) -> str:
            nonlocal calls
            calls += 1
            return f"answer to {key}"

        @cached(ttl_seconds=30, shared_cache=True)
        def sync_lookup(key: str) -> str:
            return answer(key)

        @cached(ttl_seconds=30, shared_cache=True)
        async def async_lookup(key: str) -> str:
            return answer(key)

        key = _unique("redirected")
        redis_key = cache_key(f"{caller}_lookup", key=key)
        slot = sync_redis.keyslot(redis_key)
        owner = sync_redis.get_node_from_key(redis_key)
        other = next(node for node in primaries if node.name != owner.name)

        def make_stale(client: Any) -> None:
            if redirect == "MOVED":
                # A slot map gone stale, as after a failover or a resharding:
                # the key's next read goes to a shard that doesn't serve it.
                stale = client.get_node(host=other.host, port=other.port)
                monkeypatch.setitem(client.nodes_manager.slots_cache, slot, [stale])
                # Every fifth MOVED makes redis-py read the whole map instead.
                monkeypatch.setattr(client, "reinitialize_counter", 0)

        def read_twice() -> None:
            if caller == "sync":
                assert sync_lookup(key) == f"answer to {key}"
                make_stale(sync_redis)
                assert sync_lookup(key) == f"answer to {key}"
                return

            async def on_a_fresh_loop() -> None:
                assert await async_lookup(key) == f"answer to {key}"
                make_stale(await cache_module._get_async_redis())
                assert await async_lookup(key) == f"answer to {key}"

            # A fresh loop's client holds one connection to each shard.
            # redis-py's async client sends ASKING and the command it allows
            # as two commands, each on the first idle connection, so with
            # several idle the command can miss its ASKING: a redis-py bug of
            # its own, not the remap's (#17090).
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(on_a_fresh_loop())
            finally:
                loop.close()

        redirects = _record_redirects(monkeypatch)
        failed_dials = _record_failed_dials(monkeypatch)
        if redirect == "MOVED":
            await asyncio.to_thread(read_twice)
        else:
            # Mid-resharding, the shard moving the slot answers for a key it
            # doesn't hold with ASK, and the other shard holds it from then.
            # Undone however far it got: a slot left migrating on the shared
            # cluster would redirect every later test's commands for its keys.
            try:
                await asyncio.to_thread(
                    _start_moving_slot, sync_redis, slot, owner, other
                )
                await asyncio.to_thread(read_twice)
            finally:
                await asyncio.to_thread(
                    _stop_moving_slot, sync_redis, slot, owner, other, redis_key
                )

        assert calls == 1
        assert redirects and set(redirects) == {redirect}
        assert failed_dials == []

    @pytest.mark.asyncio
    async def test_a_run_of_moved_redirects_keeps_the_clients_connections(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """redis-py 8.1's async client closes every connection on every fifth
        MOVED, failing the other commands in flight on them: during a
        resharding, a stream of reads failed and were computed instead. The
        cache's client follows each MOVED on its own."""
        sync_redis = await asyncio.to_thread(cache_module._get_redis)
        primaries = sync_redis.get_primaries()
        if len(primaries) < 2:
            pytest.skip("A redirect needs a cluster of several shards, as CI's")
        client = await cache_module._get_async_redis()

        @cached(ttl_seconds=30, shared_cache=True)
        async def lookup(key: str) -> str:
            return key

        keys = [_unique("moved") for _ in range(5)]
        for key in keys:
            assert await lookup(key) == key
        closed: list[Any] = []
        disconnect = async_connection.AbstractConnection.disconnect

        async def recorded_disconnect(self, *args, **kwargs):
            closed.append(self)
            return await disconnect(self, *args, **kwargs)

        monkeypatch.setattr(
            async_connection.AbstractConnection, "disconnect", recorded_disconnect
        )
        monkeypatch.setattr(client, "reinitialize_counter", 0)
        redirects = _record_redirects(monkeypatch)
        for key in keys:
            # Each read goes to a shard that doesn't serve its slot.
            redis_key = cache_key("lookup", key=key)
            owner = sync_redis.get_node_from_key(redis_key)
            other = next(node for node in primaries if node.name != owner.name)
            stale = client.get_node(host=other.host, port=other.port)
            monkeypatch.setitem(
                client.nodes_manager.slots_cache, sync_redis.keyslot(redis_key), [stale]
            )
            assert await lookup(key) == key

        assert closed == []
        assert redirects == ["MOVED"] * 5


def _record_redirects(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The kind of every MOVED or ASK reply a client gets from now on."""
    redirects: list[str] = []
    init = AskError.__init__

    def recorded_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        redirects.append("MOVED" if isinstance(self, MovedError) else "ASK")

    monkeypatch.setattr(AskError, "__init__", recorded_init)
    return redirects


def _record_failed_dials(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Every address a Redis connection, sync or async, fails to reach from
    now on."""
    failed: list[tuple[str, int]] = []
    connect = redis.connection.Connection._connect
    connect_async = async_connection.Connection._connect

    def recorded_connect(self):
        try:
            return connect(self)
        except Exception:
            failed.append((self.host, self.port))
            raise

    async def recorded_connect_async(self):
        try:
            return await connect_async(self)
        except Exception:
            failed.append((self.host, self.port))
            raise

    monkeypatch.setattr(redis.connection.Connection, "_connect", recorded_connect)
    monkeypatch.setattr(async_connection.Connection, "_connect", recorded_connect_async)
    return failed


def _start_moving_slot(
    redis: RedisCluster, slot: int, source: ClusterNode, destination: ClusterNode
) -> None:
    destination_id = redis.cluster_myid(destination)
    source_id = redis.cluster_myid(source)
    redis.cluster_setslot(destination, source_id, slot, "IMPORTING")
    redis.cluster_setslot(source, destination_id, slot, "MIGRATING")


def _stop_moving_slot(
    redis: RedisCluster,
    slot: int,
    source: ClusterNode,
    destination: ClusterNode,
    redis_key: str,
) -> None:
    """Undo ``_start_moving_slot``, however far it got. Each step is tried
    whatever the others did, and their failures are raised together after."""
    failures: list[Exception] = []

    def attempt(step: Callable[[], object]) -> None:
        try:
            step()
        except Exception as e:
            failures.append(e)

    def drop_entry() -> None:
        # The entry went to the importing shard, which serves it only after
        # ASKING. A shard that isn't importing the slot holds none of it.
        with redis.get_redis_connection(destination).pipeline(
            transaction=False
        ) as pipe:
            pipe.execute_command("ASKING")
            pipe.delete(redis_key)
            try:
                pipe.execute()
            except MovedError:
                pass

    def stabilize(node: ClusterNode) -> None:
        redis.execute_command("CLUSTER SETSLOT", slot, "STABLE", target_nodes=node)

    attempt(drop_entry)
    attempt(lambda: stabilize(source))
    attempt(lambda: stabilize(destination))
    if failures:
        raise ExceptionGroup("Could not undo the slot's migration", failures)
