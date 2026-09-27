"""Unit tests for ``scope_lock.graph_write_lock`` on the in-memory Redis
(``conftest.lock_redis``).

The live runs, through the production worker and a real Redis, are in
``recall_inflight_integration_test.py``.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from . import scope_lock
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key
from .scope_lock import LockState, graph_write_lock

_GROUP = "user_abc"
_KEY = write_lock_key(_GROUP)


@pytest.mark.asyncio
async def test_holds_the_key_with_its_token_and_releases_it(
    lock_redis: FakeRedis,
) -> None:
    async with graph_write_lock(_GROUP, wait_seconds=0) as lock:
        assert lock is LockState.HELD
        assert _KEY in lock_redis.values
        assert lock_redis.ttls[_KEY] == scope_lock.WRITE_LOCK_TTL_SECONDS

    assert _KEY not in lock_redis.values


@pytest.mark.asyncio
async def test_a_lock_held_for_the_whole_wait_is_busy_and_left_alone(
    lock_redis: FakeRedis,
) -> None:
    lock_redis.values[_KEY] = "another writer"

    async with graph_write_lock(_GROUP, wait_seconds=0) as lock:
        assert lock is LockState.BUSY

    assert lock_redis.values[_KEY] == "another writer"


@pytest.mark.asyncio
async def test_a_writer_waits_for_the_lock_to_come_free(
    lock_redis: FakeRedis,
) -> None:
    lock_redis.values[_KEY] = "another writer"

    async def finish() -> None:
        await asyncio.sleep(0.3)
        del lock_redis.values[_KEY]

    other = asyncio.create_task(finish())
    async with graph_write_lock(_GROUP, wait_seconds=5) as lock:
        assert lock is LockState.HELD
    await other


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raises", "hangs"])
async def test_without_redis_the_writer_goes_ahead_and_warns(
    failure: str, caplog: pytest.LogCaptureFixture
) -> None:
    async def hang() -> FakeRedis:
        await asyncio.sleep(60)
        return FakeRedis()

    down = hang if failure == "hangs" else AsyncMock(side_effect=ConnectionError())
    with (
        patch.object(scope_lock, "get_redis_async", down),
        patch.object(scope_lock, "_REDIS_TIMEOUT_SECONDS", 0.05),
    ):
        async with graph_write_lock(_GROUP, wait_seconds=5) as lock:
            assert lock is LockState.UNAVAILABLE

    assert "write lock unavailable" in caplog.text


@pytest.mark.asyncio
async def test_release_never_deletes_a_newer_holders_lock(
    lock_redis: FakeRedis,
) -> None:
    async with graph_write_lock(_GROUP, wait_seconds=0):
        lock_redis.values[_KEY] = "a writer after our lock expired"

    assert lock_redis.values[_KEY] == "a writer after our lock expired"


@pytest.mark.asyncio
async def test_a_long_hold_keeps_renewing_the_lock(lock_redis: FakeRedis) -> None:
    renewals: list[int] = []
    extend = lock_redis.eval

    async def counting(script: str, numkeys: int, key: str, *args: object) -> int:
        if script == scope_lock.EXTEND_SCRIPT:
            renewals.append(int(str(args[1])))
        return await extend(script, numkeys, key, *args)

    with (
        patch.object(scope_lock, "_RENEW_EVERY_SECONDS", 0.05),
        patch.object(lock_redis, "eval", counting),
    ):
        async with graph_write_lock(_GROUP, wait_seconds=0):
            await asyncio.sleep(0.3)

    assert len(renewals) >= 2
    assert set(renewals) == {scope_lock.WRITE_LOCK_TTL_SECONDS}


@pytest.mark.asyncio
async def test_a_failed_release_only_warns(
    lock_redis: FakeRedis, caplog: pytest.LogCaptureFixture
) -> None:
    with patch.object(lock_redis, "eval", AsyncMock(side_effect=RuntimeError("x"))):
        async with graph_write_lock(_GROUP, wait_seconds=0) as lock:
            assert lock is LockState.HELD

    assert "Releasing" in caplog.text
