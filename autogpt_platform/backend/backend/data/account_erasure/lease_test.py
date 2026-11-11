"""The per-account erasure lease: taken once, refused while another run
holds it, an error when Redis can't tell, renewed and released through the
lock."""

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
