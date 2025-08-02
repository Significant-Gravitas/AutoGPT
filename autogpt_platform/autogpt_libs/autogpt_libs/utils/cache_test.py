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

from autogpt_libs.utils.cache import clear_thread_cache, thread_cached


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
