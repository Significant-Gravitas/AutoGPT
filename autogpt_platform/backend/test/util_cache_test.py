"""Tests for the @thread_cached decorator.

This module tests the thread-local caching functionality including:
- Basic caching for sync and async functions
- Thread isolation (each thread has its own cache)
- Cache clearing functionality
- Exception handling (exceptions are not cached)
- Argument handling (positional vs keyword arguments)
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from backend.util.cache import cached, clear_thread_cache, thread_cached


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
        @thread_cached
        def slow_function(x: int) -> int:
            print(f"slow_function called with x={x}")
            time.sleep(0.1)
            return x * 2

        start = time.time()
        result1 = slow_function(5)
        first_call_time = time.time() - start
        print(f"First call took {first_call_time:.4f} seconds")

        start = time.time()
        result2 = slow_function(5)
        second_call_time = time.time() - start
        print(f"Second call took {second_call_time:.4f} seconds")

        assert result1 == result2 == 10
        assert first_call_time > 0.09
        assert second_call_time < 0.01

    @pytest.mark.asyncio
    async def test_async_caching_performance(self):
        @thread_cached
        async def slow_async_function(x: int) -> int:
            print(f"slow_async_function called with x={x}")
            await asyncio.sleep(0.1)
            return x * 2

        start = time.time()
        result1 = await slow_async_function(5)
        first_call_time = time.time() - start
        print(f"First async call took {first_call_time:.4f} seconds")

        start = time.time()
        result2 = await slow_async_function(5)
        second_call_time = time.time() - start
        print(f"Second async call took {second_call_time:.4f} seconds")

        assert result1 == result2 == 10
        assert first_call_time > 0.09
        assert second_call_time < 0.01

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
    """Tests for shared_cache functionality using Redis."""

    @pytest.fixture(autouse=True)
    def setup_redis_mock(self):
        """Mock Redis client for testing."""
        with patch("backend.util.cache._get_redis_client") as mock_redis_func:
            # Configure mock to behave like Redis
            mock_redis = Mock()
            self.mock_redis = mock_redis
            self.redis_storage = {}

            def mock_get(key):
                return self.redis_storage.get(key)

            def mock_getex(key, ex=None):
                # GETEX returns value and optionally refreshes TTL
                return self.redis_storage.get(key)

            def mock_set(key, value):
                self.redis_storage[key] = value
                return True

            def mock_setex(key, ttl, value):
                self.redis_storage[key] = value
                return True

            def mock_exists(key):
                return 1 if key in self.redis_storage else 0

            def mock_delete(key):
                if key in self.redis_storage:
                    del self.redis_storage[key]
                    return 1
                return 0

            def mock_scan_iter(pattern, count=None):
                # Pattern is a string like "cache:*", keys in storage are strings
                prefix = pattern.rstrip("*")
                return [
                    k
                    for k in self.redis_storage.keys()
                    if isinstance(k, str) and k.startswith(prefix)
                ]

            def mock_pipeline():
                pipe = Mock()
                deleted_keys = []

                def pipe_delete(key):
                    deleted_keys.append(key)
                    return pipe

                def pipe_execute():
                    # Actually delete the keys when pipeline executes
                    for key in deleted_keys:
                        self.redis_storage.pop(key, None)
                    deleted_keys.clear()
                    return []

                pipe.delete = Mock(side_effect=pipe_delete)
                pipe.execute = Mock(side_effect=pipe_execute)
                return pipe

            mock_redis.get = Mock(side_effect=mock_get)
            mock_redis.getex = Mock(side_effect=mock_getex)
            mock_redis.set = Mock(side_effect=mock_set)
            mock_redis.setex = Mock(side_effect=mock_setex)
            mock_redis.exists = Mock(side_effect=mock_exists)
            mock_redis.delete = Mock(side_effect=mock_delete)
            mock_redis.scan_iter = Mock(side_effect=mock_scan_iter)
            mock_redis.pipeline = Mock(side_effect=mock_pipeline)

            # Make _get_redis_client return the mock
            mock_redis_func.return_value = mock_redis

            yield mock_redis

            # Cleanup
            self.redis_storage.clear()

    def test_sync_shared_cache_basic(self):
        """Test basic shared cache functionality with sync function."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        def shared_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 10

        # First call - should miss cache
        result1 = shared_function(5)
        assert result1 == 50
        assert call_count == 1
        assert self.mock_redis.get.called
        assert self.mock_redis.setex.called  # setex is used for TTL

        # Second call - should hit cache
        result2 = shared_function(5)
        assert result2 == 50
        assert call_count == 1  # Function not called again

    @pytest.mark.asyncio
    async def test_async_shared_cache_basic(self):
        """Test basic shared cache functionality with async function."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        async def async_shared_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 20

        # First call - should miss cache
        result1 = await async_shared_function(3)
        assert result1 == 60
        assert call_count == 1
        assert self.mock_redis.get.called
        assert self.mock_redis.setex.called  # setex is used for TTL

        # Second call - should hit cache
        result2 = await async_shared_function(3)
        assert result2 == 60
        assert call_count == 1  # Function not called again

    def test_sync_shared_cache_with_ttl(self):
        """Test shared cache with TTL using sync function."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=60)
        def shared_ttl_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 30

        # First call
        result1 = shared_ttl_function(2)
        assert result1 == 60
        assert call_count == 1
        assert self.mock_redis.setex.called

        # Second call - should use cache
        result2 = shared_ttl_function(2)
        assert result2 == 60
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_shared_cache_with_ttl(self):
        """Test shared cache with TTL using async function."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=120)
        async def async_shared_ttl_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 40

        # First call
        result1 = await async_shared_ttl_function(4)
        assert result1 == 160
        assert call_count == 1
        assert self.mock_redis.setex.called

        # Second call - should use cache
        result2 = await async_shared_ttl_function(4)
        assert result2 == 160
        assert call_count == 1

    def test_shared_cache_clear(self):
        """Test clearing shared cache."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        def clearable_shared_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 50

        # First call
        result1 = clearable_shared_function(1)
        assert result1 == 50
        assert call_count == 1

        # Second call - should use cache
        result2 = clearable_shared_function(1)
        assert result2 == 50
        assert call_count == 1

        # Clear cache
        clearable_shared_function.cache_clear()
        assert self.mock_redis.pipeline.called

        # Third call - should execute function again
        result3 = clearable_shared_function(1)
        assert result3 == 50
        assert call_count == 2

    def test_shared_cache_delete(self):
        """Test deleting specific shared cache entry."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        def deletable_shared_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 60

        # First call for x=1
        result1 = deletable_shared_function(1)
        assert result1 == 60
        assert call_count == 1

        # First call for x=2
        result2 = deletable_shared_function(2)
        assert result2 == 120
        assert call_count == 2

        # Delete entry for x=1
        was_deleted = deletable_shared_function.cache_delete(1)
        assert was_deleted is True

        # Call with x=1 should execute function again
        result3 = deletable_shared_function(1)
        assert result3 == 60
        assert call_count == 3

        # Call with x=2 should still use cache
        result4 = deletable_shared_function(2)
        assert result4 == 120
        assert call_count == 3

    def test_shared_cache_error_handling(self):
        """Test that Redis errors are handled gracefully."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        def error_prone_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 70

        # Simulate Redis error
        self.mock_redis.get.side_effect = Exception("Redis connection error")

        # Function should still work
        result = error_prone_function(1)
        assert result == 70
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_shared_cache_error_handling(self):
        """Test that Redis errors are handled gracefully in async functions."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        async def async_error_prone_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 80

        # Simulate Redis error
        self.mock_redis.get.side_effect = Exception("Redis connection error")

        # Function should still work
        result = await async_error_prone_function(1)
        assert result == 80
        assert call_count == 1

    def test_shared_cache_with_complex_types(self):
        """Test shared cache with complex return types (lists, dicts)."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        def complex_return_function(x: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {"value": x, "squared": x * x, "list": [1, 2, 3]}

        # First call
        result1 = complex_return_function(5)
        assert result1 == {"value": 5, "squared": 25, "list": [1, 2, 3]}
        assert call_count == 1

        # Second call - should use cache
        result2 = complex_return_function(5)
        assert result2 == {"value": 5, "squared": 25, "list": [1, 2, 3]}
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_thundering_herd_shared_cache(self):
        """Test thundering herd protection with shared cache."""
        call_count = 0

        @cached(shared_cache=True, ttl_seconds=300)
        async def slow_shared_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * x

        # Launch concurrent coroutines
        tasks = [slow_shared_function(9) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be the same
        assert all(result == 81 for result in results)
        # Only one coroutine should have executed the function
        assert call_count == 1

    def test_shared_cache_info(self):
        """Test cache_info with shared cache."""

        @cached(shared_cache=True, maxsize=100, ttl_seconds=300)
        def info_function(x: int) -> int:
            return x * 90

        # Call the function to populate cache
        info_function(1)

        # Get cache info
        info = info_function.cache_info()
        assert "size" in info
        assert info["maxsize"] is None  # Redis manages its own size
        assert info["ttl_seconds"] == 300
