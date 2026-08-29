import time

import pytest

from backend.util.decorator import async_error_logged, error_logged, time_measured
from backend.util.retry import continuous_retry


@time_measured
def example_function(a: int, b: int, c: int) -> int:
    time.sleep(0.5)
    return a + b + c


@error_logged(swallow=True)
def example_function_with_error_swallowed(a: int, b: int, c: int) -> int:
    raise ValueError("This error should be swallowed")


@error_logged(swallow=False)
def example_function_with_error_not_swallowed(a: int, b: int, c: int) -> int:
    raise ValueError("This error should NOT be swallowed")


@async_error_logged(swallow=True)
async def async_function_with_error_swallowed() -> int:
    raise ValueError("This async error should be swallowed")


@async_error_logged(swallow=False)
async def async_function_with_error_not_swallowed() -> int:
    raise ValueError("This async error should NOT be swallowed")


def test_timer_decorator():
    """Test that the time_measured decorator correctly measures execution time."""
    info, res = example_function(1, 2, 3)
    assert info.cpu_time >= 0
    assert info.wall_time >= 0.4
    assert res == 6


def test_error_decorator_swallow_true():
    """Test that error_logged(swallow=True) logs and swallows errors."""
    res = example_function_with_error_swallowed(1, 2, 3)
    assert res is None


def test_error_decorator_swallow_false():
    """Test that error_logged(swallow=False) logs errors but re-raises them."""
    with pytest.raises(ValueError, match="This error should NOT be swallowed"):
        example_function_with_error_not_swallowed(1, 2, 3)


def test_async_error_decorator_swallow_true():
    """Test that async_error_logged(swallow=True) logs and swallows errors."""
    import asyncio

    async def run_test():
        res = await async_function_with_error_swallowed()
        return res

    res = asyncio.run(run_test())
    assert res is None


def test_async_error_decorator_swallow_false():
    """Test that async_error_logged(swallow=False) logs errors but re-raises them."""
    import asyncio

    async def run_test():
        await async_function_with_error_not_swallowed()

    with pytest.raises(ValueError, match="This async error should NOT be swallowed"):
        asyncio.run(run_test())


def test_continuous_retry_basic():
    """Test that continuous_retry decorator retries on exception."""

    class MockManager:
        def __init__(self):
            self.call_count = 0

        @continuous_retry(retry_delay=0.01)
        def failing_method(self):
            self.call_count += 1
            if self.call_count <= 2:
                # Fail on first two calls
                raise RuntimeError("Simulated failure")
            return "success"

    mock_manager = MockManager()

    # Should retry and eventually succeed
    result = mock_manager.failing_method()
    assert result == "success"
    assert mock_manager.call_count == 3  # Failed twice, succeeded on third
