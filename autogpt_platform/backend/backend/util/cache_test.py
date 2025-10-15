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
from unittest.mock import Mock

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
    """Tests for shared_cache (Redis-backed) functionality."""

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
        shared_async_function.cache_clear()

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
        shared_async_function.cache_clear()

    def test_shared_cache_ttl_refresh(self):
        """Test TTL refresh functionality with shared cache."""
        call_count = 0

        @cached(ttl_seconds=2, shared_cache=True, refresh_ttl_on_get=True)
        def ttl_refresh_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 10

        # Clear any existing cache
        ttl_refresh_function.cache_clear()

        # First call
        result1 = ttl_refresh_function(3)
        assert result1 == 30
        assert call_count == 1

        # Wait 1 second
        time.sleep(1)

        # Second call - should refresh TTL and use cache
        result2 = ttl_refresh_function(3)
        assert result2 == 30
        assert call_count == 1

        # Wait another 1.5 seconds (total 2.5s from first call, 1.5s from second)
        time.sleep(1.5)

        # Third call - TTL should have been refreshed, so still cached
        result3 = ttl_refresh_function(3)
        assert result3 == 30
        assert call_count == 1

        # Wait 2.1 seconds - now it should expire
        time.sleep(2.1)

        # Fourth call - should call function again
        result4 = ttl_refresh_function(3)
        assert result4 == 30
        assert call_count == 2

        # Cleanup
        ttl_refresh_function.cache_clear()

    def test_shared_cache_without_ttl_refresh(self):
        """Test that TTL doesn't refresh when refresh_ttl_on_get=False."""
        call_count = 0

        @cached(ttl_seconds=2, shared_cache=True, refresh_ttl_on_get=False)
        def no_ttl_refresh_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 10

        # Clear any existing cache
        no_ttl_refresh_function.cache_clear()

        # First call
        result1 = no_ttl_refresh_function(4)
        assert result1 == 40
        assert call_count == 1

        # Wait 1 second
        time.sleep(1)

        # Second call - should use cache but NOT refresh TTL
        result2 = no_ttl_refresh_function(4)
        assert result2 == 40
        assert call_count == 1

        # Wait another 1.1 seconds (total 2.1s from first call)
        time.sleep(1.1)

        # Third call - should have expired
        result3 = no_ttl_refresh_function(4)
        assert result3 == 40
        assert call_count == 2

        # Cleanup
        no_ttl_refresh_function.cache_clear()

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
        shared_slow_function.cache_clear()

        # Launch multiple concurrent tasks
        tasks = [shared_slow_function(8) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return same result
        assert all(r == 64 for r in results)
        # Only one should have executed
        assert call_count == 1

        # Cleanup
        shared_slow_function.cache_clear()

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
        multi_key_function.cache_clear()

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
        multi_key_function.cache_clear()

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
