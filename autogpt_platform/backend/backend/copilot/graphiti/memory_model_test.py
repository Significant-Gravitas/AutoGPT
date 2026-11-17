import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from .memory_model import MemoryForgetFailure


@pytest.mark.parametrize(
    "factory", [MemoryForgetFailure.query_error, MemoryForgetFailure.cleanup_error]
)
@pytest.mark.parametrize(
    "exception",
    [OSError, ConnectionError, TimeoutError, RedisConnectionError, RedisTimeoutError],
)
def test_forget_failures_hide_transport_details(factory, exception) -> None:
    detail = "Error connecting to private-memory.internal:6379 with secret-token"
    failure = factory("edge-1", exception(detail))

    assert detail not in failure.reason
    assert exception.__name__ in failure.reason


@pytest.mark.parametrize(
    "factory", [MemoryForgetFailure.query_error, MemoryForgetFailure.cleanup_error]
)
def test_forget_failures_keep_query_diagnostics(factory) -> None:
    detail = "Unknown function 'datetime'"
    failure = factory("edge-1", RuntimeError(detail))

    assert f"RuntimeError: {detail}" in failure.reason
