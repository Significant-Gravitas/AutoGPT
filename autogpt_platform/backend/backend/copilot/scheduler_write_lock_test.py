"""Scheduler write lock ownership after an expiry on real Redis."""

import asyncio
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from backend.copilot import scheduler_write_lock
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)


@pytest.mark.asyncio
async def test_an_expired_holder_cannot_release_its_successors_lock(
    mocker: MockerFixture,
):
    key = f"copilot:scheduler_write_lock:test:{uuid4().hex}"
    first_holds_lock = asyncio.Event()
    first_may_exit = asyncio.Event()
    # Redis expires the first holder on demand. Neither holder's renewal or
    # normal TTL should race the assertions under a slow test runner.
    mocker.patch.object(scheduler_write_lock, "_LOCK_TTL_SECONDS", 120)
    mocker.patch.object(scheduler_write_lock, "_LOCK_RENEW_EVERY_SECONDS", 120)

    async def first_holder() -> None:
        async with scheduler_write_lock.scheduler_write_lock(key) as held:
            assert held
            first_holds_lock.set()
            await first_may_exit.wait()

    async with asyncio.timeout(20):
        redis = await redis_client.get_redis_async()
        try:
            async with asyncio.TaskGroup() as tasks:
                first = tasks.create_task(first_holder())
                await first_holds_lock.wait()
                first_token = await redis.get(key)
                assert first_token is not None
                assert await redis.pexpire(key, 1)
                while await redis.exists(key):
                    await asyncio.sleep(0.01)

                async with scheduler_write_lock.scheduler_write_lock(key) as held:
                    assert held
                    successor_token = await redis.get(key)
                    assert successor_token is not None
                    assert successor_token != first_token
                    assert not first.done()

                    first_may_exit.set()
                    await first

                    assert await redis.get(key) == successor_token
                    assert not await redis.lock(key).acquire(blocking=False)

                assert not await redis.exists(key)
        finally:
            await redis.delete(key)
