"""
Test Redis cache functionality.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from autogpt_libs.utils.cache import cached


# Test with Redis cache enabled
@pytest.mark.asyncio
async def test_redis_cache_async():
    """Test async function with Redis cache."""

    call_count = 0

    @cached(ttl_seconds=60, shared_cache=True)
    async def expensive_async_operation(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)
        return x + y

    # First call should execute function
    result1 = await expensive_async_operation(5, 3)
    assert result1 == 8
    assert call_count == 1

    # Second call with same args should use cache
    result2 = await expensive_async_operation(5, 3)
    assert result2 == 8
    assert call_count == 1  # Should not increment

    # Different args should execute function again
    result3 = await expensive_async_operation(10, 5)
    assert result3 == 15
    assert call_count == 2

    # Test cache_delete
    deleted = expensive_async_operation.cache_delete(5, 3)
    assert isinstance(deleted, bool)  # Depends on Redis availability

    # Test cache_clear
    expensive_async_operation.cache_clear()

    # Test cache_info
    info = expensive_async_operation.cache_info()
    assert "ttl_seconds" in info
    assert info["ttl_seconds"] == 60
    assert "shared_cache" in info


def test_redis_cache_sync():
    """Test sync function with Redis cache."""

    call_count = 0

    @cached(ttl_seconds=60, shared_cache=True)
    def expensive_sync_operation(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)
        return x * y

    # First call should execute function
    result1 = expensive_sync_operation(5, 3)
    assert result1 == 15
    assert call_count == 1

    # Second call with same args should use cache
    result2 = expensive_sync_operation(5, 3)
    assert result2 == 15
    assert call_count == 1  # Should not increment

    # Different args should execute function again
    result3 = expensive_sync_operation(10, 5)
    assert result3 == 50
    assert call_count == 2

    # Test cache management functions
    expensive_sync_operation.cache_clear()
    info = expensive_sync_operation.cache_info()
    assert info["shared_cache"] is True


def test_fallback_to_local_cache_when_redis_unavailable():
    """Test that cache falls back to local when Redis is unavailable."""

    with patch("autogpt_libs.utils.cache._get_redis_client", return_value=None):
        call_count = 0

        @cached(ttl_seconds=60, shared_cache=True)
        def operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Should still work with local cache
        result1 = operation(5)
        assert result1 == 10
        assert call_count == 1

        result2 = operation(5)
        assert result2 == 10
        assert call_count == 1  # Cached locally


@pytest.mark.asyncio
async def test_redis_cache_with_complex_types():
    """Test Redis cache with complex data types."""

    @cached(ttl_seconds=30, shared_cache=True)
    async def get_complex_data(user_id: str, filters: dict) -> dict:
        return {
            "user_id": user_id,
            "filters": filters,
            "data": [1, 2, 3, 4, 5],
            "nested": {"key1": "value1", "key2": ["a", "b", "c"]},
        }

    result1 = await get_complex_data("user123", {"status": "active", "limit": 10})
    result2 = await get_complex_data("user123", {"status": "active", "limit": 10})

    assert result1 == result2
    assert result1["user_id"] == "user123"
    assert result1["filters"]["status"] == "active"


def test_local_cache_without_shared():
    """Test that shared_cache=False uses local cache only."""

    call_count = 0

    @cached(maxsize=10, ttl_seconds=30, shared_cache=False)
    def local_only_operation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x**2

    result1 = local_only_operation(4)
    assert result1 == 16
    assert call_count == 1

    result2 = local_only_operation(4)
    assert result2 == 16
    assert call_count == 1

    # Check cache info shows it's not shared
    info = local_only_operation.cache_info()
    assert info["shared_cache"] is False
    assert info["maxsize"] == 10


if __name__ == "__main__":
    # Run basic tests
    print("Testing sync Redis cache...")
    test_redis_cache_sync()
    print("✓ Sync Redis cache test passed")

    print("Testing local cache...")
    test_local_cache_without_shared()
    print("✓ Local cache test passed")

    print("Testing fallback...")
    test_fallback_to_local_cache_when_redis_unavailable()
    print("✓ Fallback test passed")

    print("\nAll tests passed!")
