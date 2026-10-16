"""The lock every writer of a chat's at-reset job holds, on any replica.

Redis is faked at the boundary; the route and scheduler tests cover what each
writer does when it can't have the lock.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import LockNotOwnedError

from backend.copilot import at_reset_lock
from backend.copilot.at_reset_lock import at_reset_job_lock
from backend.copilot.at_reset_lock_fixtures import FakeAtResetRedis

JOB = "at-reset-send-sess-1"
KEY = f"copilot:at_reset_lock:{JOB}"


@pytest.fixture
def redis(mocker: pytest_mock.MockerFixture) -> FakeAtResetRedis:
    fake = FakeAtResetRedis()
    mocker.patch.object(at_reset_lock, "get_redis_async", AsyncMock(return_value=fake))
    return fake


@pytest.mark.asyncio
async def test_it_holds_the_job_by_its_id_for_seconds(redis):
    async with at_reset_job_lock(JOB) as locked:
        assert locked
        assert redis.is_locked(KEY)
    assert not redis.is_locked(KEY)
    # A holder that dies frees the chat in seconds, and a writer waits for
    # the one ahead of it no longer than that.
    assert redis.lock_options[KEY] == {"timeout": 10, "blocking_timeout": 5}


@pytest.mark.asyncio
async def test_it_lets_go_when_the_write_fails(redis):
    with pytest.raises(RuntimeError):
        async with at_reset_job_lock(JOB):
            raise RuntimeError("scheduler down")
    assert not redis.is_locked(KEY)


@pytest.mark.asyncio
async def test_without_redis_it_is_not_held(mocker: pytest_mock.MockerFixture):
    mocker.patch.object(
        at_reset_lock,
        "get_redis_async",
        AsyncMock(side_effect=RedisConnectionError("down")),
    )
    async with at_reset_job_lock(JOB) as locked:
        assert locked is False


@pytest.mark.asyncio
async def test_it_is_not_held_when_another_writer_keeps_it_past_the_wait(
    redis, mocker: pytest_mock.MockerFixture
):
    lock = MagicMock(acquire=AsyncMock(return_value=False), release=AsyncMock())
    mocker.patch.object(redis, "lock", return_value=lock)
    async with at_reset_job_lock(JOB) as locked:
        assert locked is False
    lock.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_lock_held_past_its_expiry_is_let_go_quietly(
    redis, mocker: pytest_mock.MockerFixture
):
    # The next writer may hold it by then; the write itself went through.
    lock = MagicMock(
        acquire=AsyncMock(return_value=True),
        release=AsyncMock(side_effect=LockNotOwnedError("expired")),
    )
    mocker.patch.object(redis, "lock", return_value=lock)
    async with at_reset_job_lock(JOB) as locked:
        assert locked
    lock.release.assert_awaited_once()
