"""The executor's liveness lease on a running turn."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.copilot.turn_lease import (
    TURN_LEASE_TTL_SECONDS,
    TurnLease,
    turn_lease_held,
    turn_lease_key,
)


@pytest_asyncio.fixture(scope="session", loop_scope="session", name="server")
async def _server_noop() -> None:
    return None


@pytest_asyncio.fixture(
    scope="session", loop_scope="session", autouse=True, name="graph_cleanup"
)
async def _graph_cleanup_noop():
    yield


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_acquire_writes_the_lease_with_a_short_ttl():
    redis = MagicMock()
    lease = TurnLease("turn-1", "pod-a", redis=redis)

    assert lease.acquire() is True

    redis.set.assert_called_once_with(
        turn_lease_key("turn-1"), "pod-a", ex=TURN_LEASE_TTL_SECONDS
    )


def test_refresh_writes_again_only_once_the_interval_passed():
    redis = MagicMock()
    clock = _Clock()
    lease = TurnLease("turn-1", "pod-a", redis=redis, refresh_seconds=10, clock=clock)
    lease.acquire()

    clock.now = 9.0
    lease.refresh()
    assert redis.set.call_count == 1

    clock.now = 10.0
    lease.refresh()
    assert redis.set.call_count == 2


def test_a_failed_write_is_retried_on_the_next_refresh():
    redis = MagicMock()
    redis.set.side_effect = [ConnectionError("blip"), True]
    clock = _Clock()
    lease = TurnLease("turn-1", "pod-a", redis=redis, refresh_seconds=10, clock=clock)

    assert lease.acquire() is False
    assert lease.refresh() is True
    assert redis.set.call_count == 2


def test_release_deletes_only_its_own_lease():
    redis = MagicMock()
    lease = TurnLease("turn-1", "pod-a", redis=redis)

    lease.release()

    _script, numkeys, key, owner = redis.eval.call_args.args
    assert (numkeys, key, owner) == (1, turn_lease_key("turn-1"), "pod-a")


def test_the_lease_never_raises():
    redis = MagicMock()
    redis.set.side_effect = RuntimeError("redis down")
    redis.eval.side_effect = RuntimeError("redis down")
    lease = TurnLease("turn-1", "pod-a", redis=redis)

    assert lease.acquire() is False
    assert lease.refresh() is False
    lease.release()


@pytest.mark.asyncio
async def test_turn_lease_held_reads_the_key():
    redis = MagicMock(exists=AsyncMock(return_value=1))

    assert await turn_lease_held(redis, "turn-1") is True
    redis.exists.assert_awaited_once_with(turn_lease_key("turn-1"))
