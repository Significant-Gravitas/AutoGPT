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

from autogpt_libs.utils.cache import (
    async_cache,
    async_ttl_cache,
    clear_thread_cache,
    thread_cached,
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


class TestAsyncTTLCache:
    """Tests for the @async_ttl_cache decorator."""

    @pytest.mark.asyncio
    async def test_basic_caching(self):
        """Test basic caching functionality."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=60)
        async def cached_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x + y

        # First call
        result1 = await cached_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = await cached_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # No additional call

        # Different args - should call function again
        result3 = await cached_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=1)  # Short TTL
        async def short_lived_cache(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = await short_lived_cache(5)
        assert result1 == 10
        assert call_count == 1

        # Second call immediately - should use cache
        result2 = await short_lived_cache(5)
        assert result2 == 10
        assert call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # Third call after expiration - should call function again
        result3 = await short_lived_cache(5)
        assert result3 == 10
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_info(self):
        """Test cache info functionality."""

        @async_ttl_cache(maxsize=5, ttl_seconds=300)
        async def info_test_function(x: int) -> int:
            return x * 3

        # Check initial cache info
        info = info_test_function.cache_info()
        assert info["size"] == 0
        assert info["maxsize"] == 5
        assert info["ttl_seconds"] == 300

        # Add an entry
        await info_test_function(1)
        info = info_test_function.cache_info()
        assert info["size"] == 1

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test cache clearing functionality."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=60)
        async def clearable_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 4

        # First call
        result1 = await clearable_function(2)
        assert result1 == 8
        assert call_count == 1

        # Second call - should use cache
        result2 = await clearable_function(2)
        assert result2 == 8
        assert call_count == 1

        # Clear cache
        clearable_function.cache_clear()

        # Third call after clear - should call function again
        result3 = await clearable_function(2)
        assert result3 == 8
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_maxsize_cleanup(self):
        """Test that cache cleans up when maxsize is exceeded."""
        call_count = 0

        @async_ttl_cache(maxsize=3, ttl_seconds=60)
        async def size_limited_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x**2

        # Fill cache to maxsize
        await size_limited_function(1)  # call_count: 1
        await size_limited_function(2)  # call_count: 2
        await size_limited_function(3)  # call_count: 3

        info = size_limited_function.cache_info()
        assert info["size"] == 3

        # Add one more entry - should trigger cleanup
        await size_limited_function(4)  # call_count: 4

        # Cache size should be reduced (cleanup removes oldest entries)
        info = size_limited_function.cache_info()
        assert info["size"] is not None and info["size"] <= 3  # Should be cleaned up

    @pytest.mark.asyncio
    async def test_argument_variations(self):
        """Test caching with different argument patterns."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=60)
        async def arg_test_function(a: int, b: str = "default", *, c: int = 100) -> str:
            nonlocal call_count
            call_count += 1
            return f"{a}-{b}-{c}"

        # Different ways to call with same logical arguments
        result1 = await arg_test_function(1, "test", c=200)
        assert call_count == 1

        # Same arguments, same order - should use cache
        result2 = await arg_test_function(1, "test", c=200)
        assert call_count == 1
        assert result1 == result2

        # Different arguments - should call function
        result3 = await arg_test_function(2, "test", c=200)
        assert call_count == 2
        assert result1 != result3

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test that exceptions are not cached."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=60)
        async def exception_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if x < 0:
                raise ValueError("Negative value not allowed")
            return x * 2

        # Successful call - should be cached
        result1 = await exception_function(5)
        assert result1 == 10
        assert call_count == 1

        # Same successful call - should use cache
        result2 = await exception_function(5)
        assert result2 == 10
        assert call_count == 1

        # Exception call - should not be cached
        with pytest.raises(ValueError):
            await exception_function(-1)
        assert call_count == 2

        # Same exception call - should call again (not cached)
        with pytest.raises(ValueError):
            await exception_function(-1)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test caching behavior with concurrent calls."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=60)
        async def concurrent_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # Simulate work
            return x * x

        # Launch concurrent calls with same arguments
        tasks = [concurrent_function(3) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be the same
        assert all(result == 9 for result in results)

        # Note: Due to race conditions, call_count might be up to 5 for concurrent calls
        # This tests that the cache doesn't break under concurrent access
        assert 1 <= call_count <= 5


class TestAsyncCache:
    """Tests for the @async_cache decorator (no TTL)."""

    @pytest.mark.asyncio
    async def test_basic_caching_no_ttl(self):
        """Test basic caching functionality without TTL."""
        call_count = 0

        @async_cache(maxsize=10)
        async def cached_function(x: int, y: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x + y

        # First call
        result1 = await cached_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call with same args - should use cache
        result2 = await cached_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # No additional call

        # Third call after some time - should still use cache (no TTL)
        await asyncio.sleep(0.05)
        result3 = await cached_function(1, 2)
        assert result3 == 3
        assert call_count == 1  # Still no additional call

        # Different args - should call function again
        result4 = await cached_function(2, 3)
        assert result4 == 5
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_ttl_vs_ttl_behavior(self):
        """Test the difference between TTL and no-TTL caching."""
        ttl_call_count = 0
        no_ttl_call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=1)  # Short TTL
        async def ttl_function(x: int) -> int:
            nonlocal ttl_call_count
            ttl_call_count += 1
            return x * 2

        @async_cache(maxsize=10)  # No TTL
        async def no_ttl_function(x: int) -> int:
            nonlocal no_ttl_call_count
            no_ttl_call_count += 1
            return x * 2

        # First calls
        await ttl_function(5)
        await no_ttl_function(5)
        assert ttl_call_count == 1
        assert no_ttl_call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # Second calls after TTL expiry
        await ttl_function(5)  # Should call function again (TTL expired)
        await no_ttl_function(5)  # Should use cache (no TTL)
        assert ttl_call_count == 2  # TTL function called again
        assert no_ttl_call_count == 1  # No-TTL function still cached

    @pytest.mark.asyncio
    async def test_async_cache_info(self):
        """Test cache info for no-TTL cache."""

        @async_cache(maxsize=5)
        async def info_test_function(x: int) -> int:
            return x * 3

        # Check initial cache info
        info = info_test_function.cache_info()
        assert info["size"] == 0
        assert info["maxsize"] == 5
        assert info["ttl_seconds"] is None  # No TTL

        # Add an entry
        await info_test_function(1)
        info = info_test_function.cache_info()
        assert info["size"] == 1


class TestTTLOptional:
    """Tests for optional TTL functionality."""

    @pytest.mark.asyncio
    async def test_ttl_none_behavior(self):
        """Test that ttl_seconds=None works like no TTL."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=None)
        async def no_ttl_via_none(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x**2

        # First call
        result1 = await no_ttl_via_none(3)
        assert result1 == 9
        assert call_count == 1

        # Wait (would expire if there was TTL)
        await asyncio.sleep(0.1)

        # Second call - should still use cache
        result2 = await no_ttl_via_none(3)
        assert result2 == 9
        assert call_count == 1  # No additional call

        # Check cache info
        info = no_ttl_via_none.cache_info()
        assert info["ttl_seconds"] is None

    @pytest.mark.asyncio
    async def test_cache_options_comparison(self):
        """Test different cache options work as expected."""
        ttl_calls = 0
        no_ttl_calls = 0

        @async_ttl_cache(maxsize=10, ttl_seconds=1)  # With TTL
        async def ttl_function(x: int) -> int:
            nonlocal ttl_calls
            ttl_calls += 1
            return x * 10

        @async_cache(maxsize=10)  # Process-level cache (no TTL)
        async def process_function(x: int) -> int:
            nonlocal no_ttl_calls
            no_ttl_calls += 1
            return x * 10

        # Both should cache initially
        await ttl_function(3)
        await process_function(3)
        assert ttl_calls == 1
        assert no_ttl_calls == 1

        # Immediate second calls - both should use cache
        await ttl_function(3)
        await process_function(3)
        assert ttl_calls == 1
        assert no_ttl_calls == 1

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # After TTL expiry
        await ttl_function(3)  # Should call function again
        await process_function(3)  # Should still use cache
        assert ttl_calls == 2  # TTL cache expired, called again
        assert no_ttl_calls == 1  # Process cache never expires
