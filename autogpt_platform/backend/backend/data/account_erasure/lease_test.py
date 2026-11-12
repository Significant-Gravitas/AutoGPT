"""The per-account erasure lease: taken once, refused while another run
holds it, an error when Redis can't tell, short-lived so a crashed run frees
it within minutes, kept alive by a heartbeat while a run lives, renewed and
released through the lock."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException

from backend.data.account_erasure import lease

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"


@pytest.fixture
def locks(monkeypatch):
    """`AsyncClusterLock` replaced by a mock whose `try_acquire` answers as
    the test says (its own owner ID, another one, or None)."""
    made: list[MagicMock] = []
    answers: dict[str, str | None] = {"owner": "self"}

    def make(*, redis, key, owner_id, timeout):
        lock = MagicMock(key=key, owner_id=owner_id, timeout=timeout)
        answer = answers["owner"]
        lock.try_acquire = AsyncMock(
            return_value=owner_id if answer == "self" else answer
        )
        lock.refresh = AsyncMock(return_value=True)
        lock.release = AsyncMock()
        made.append(lock)
        return lock

    monkeypatch.setattr(lease, "AsyncClusterLock", make)
    monkeypatch.setattr(lease, "get_redis_async", AsyncMock())
    return made, answers


@pytest.mark.asyncio
async def test_the_lease_is_per_account_and_outlives_an_erasure(locks):
    made, _ = locks

    held = await lease.acquire_erasure_lease(USER_ID)

    assert held is made[0]
    assert held.key == f"{lease.LEASE_KEY_PREFIX}{USER_ID}"
    assert held.timeout == lease.LEASE_TTL_SECONDS


@pytest.mark.asyncio
async def test_another_runs_lease_is_refused(locks):
    _, answers = locks
    answers["owner"] = "another-run"

    assert await lease.acquire_erasure_lease(USER_ID) is None


@pytest.mark.asyncio
async def test_redis_that_cannot_answer_is_an_error_not_a_free_lease(locks):
    _, answers = locks
    answers["owner"] = None

    with pytest.raises(lease.ErasureLeaseUnavailable):
        await lease.acquire_erasure_lease(USER_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        # What the cluster client raises when it can't connect at all.
        RedisClusterException("Redis Cluster cannot be connected"),
        RedisConnectionError("down"),
    ],
)
async def test_redis_that_cannot_connect_is_the_same_error(locks, monkeypatch, error):
    """A cold client connects on first use, and its own error would reach the
    CLI as a raw traceback."""
    monkeypatch.setattr(lease, "get_redis_async", AsyncMock(side_effect=error))

    with pytest.raises(lease.ErasureLeaseUnavailable):
        await lease.acquire_erasure_lease(USER_ID)


@pytest.mark.asyncio
async def test_renewal_and_release_go_through_the_lock(locks):
    made, _ = locks
    held = await lease.acquire_erasure_lease(USER_ID)
    assert held is not None
    made[0].refresh.return_value = False

    assert await lease.keep_erasure_lease(held) is False
    await lease.release_erasure_lease(held)

    made[0].release.assert_awaited_once_with()


def test_a_crashed_runs_lease_lapses_within_minutes():
    """Nobody renews a killed run's lease: a rerun gets through once it
    lapses, not an hour later. A live run's heartbeat renews it several times
    per lifetime, each time past the lock's rate limit (a tenth of it), under
    which a renewal only checks the owner and extends nothing."""
    assert lease.LEASE_TTL_SECONDS <= 5 * 60
    assert lease.HEARTBEAT_SECONDS * 3 <= lease.LEASE_TTL_SECONDS
    assert lease.HEARTBEAT_SECONDS > lease.LEASE_TTL_SECONDS // 10


@pytest.mark.asyncio
async def test_the_heartbeat_renews_the_lease_during_a_long_step(locks, monkeypatch):
    """One step can outlast the lifetime: the heartbeat keeps the lease while
    it runs, and stops with the erasure."""
    made, _ = locks
    monkeypatch.setattr(lease, "HEARTBEAT_SECONDS", 0.01)
    held = await lease.acquire_erasure_lease(USER_ID)
    assert held is not None

    async with lease.renew_in_background(held):
        await asyncio.sleep(0.1)

    renewals = made[0].refresh.await_count
    assert renewals >= 3
    await asyncio.sleep(0.05)
    assert made[0].refresh.await_count == renewals


@pytest.mark.asyncio
async def test_a_failed_beat_is_retried_and_the_block_still_ends(locks, monkeypatch):
    """A Redis blip loses a beat, not the lease: the next beat tries again.
    The check before each step decides whether the erasure goes on."""
    made, _ = locks
    monkeypatch.setattr(lease, "HEARTBEAT_SECONDS", 0.01)
    held = await lease.acquire_erasure_lease(USER_ID)
    assert held is not None
    made[0].refresh.side_effect = [False] + [True] * 100

    async with lease.renew_in_background(held):
        await asyncio.sleep(0.1)

    assert made[0].refresh.await_count >= 3


@pytest.mark.asyncio
async def test_the_heartbeat_stops_when_the_erasure_raises(locks, monkeypatch):
    made, _ = locks
    monkeypatch.setattr(lease, "HEARTBEAT_SECONDS", 0.01)
    held = await lease.acquire_erasure_lease(USER_ID)
    assert held is not None

    with pytest.raises(RuntimeError):
        async with lease.renew_in_background(held):
            raise RuntimeError("a step outside the report failed")

    renewals = made[0].refresh.await_count
    await asyncio.sleep(0.05)
    assert made[0].refresh.await_count == renewals
