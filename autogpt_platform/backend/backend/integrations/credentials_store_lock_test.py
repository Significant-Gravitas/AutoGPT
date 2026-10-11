"""The per-user credentials lock against a real Redis cluster.

Every copilot page load reads the user's credentials several times at once:
``/chat/transports``, ``/chat/connections``, ``/integrations/credentials`` and
``POST /chat/sessions``. When one holder of the per-user ``integrations`` lock
skipped its release, the key stayed until its 60 s TTL ran out, and each of
those requests hung for a minute. In prod they hit the frontend proxy's 30 s
timeout, and the copilot Playwright spec failed the same way in CI.
"""

import asyncio
import socket
import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from redis.asyncio.lock import Lock

from backend.integrations.credentials_store import IntegrationCredentialsStore


def _has_live_cluster() -> bool:
    """Probe the socket, not ``redis_client.connect()``: this runs at collection
    time and that helper retries a dead host for ~45 minutes before giving up."""
    from backend.data import redis_client

    try:
        socket.create_connection((redis_client.HOST, redis_client.PORT), 1).close()
    except OSError:
        return False
    return True


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not _has_live_cluster(), reason="local redis cluster not reachable"
    ),
]

# Far below the lock's 60 s TTL: a lock left behind shows up as a timeout here.
_STALL = 5.0


@pytest.fixture
def store() -> IntegrationCredentialsStore:
    from backend.data import redis_client

    redis_client._async_clients.clear()
    return IntegrationCredentialsStore()


@pytest.fixture
def user_id() -> str:
    return f"lock-test-{uuid.uuid4()}"


async def test_a_waiter_that_takes_the_lock_during_a_release_still_releases_it(
    store: IntegrationCredentialsStore, user_id: str, monkeypatch
):
    # Under load, the releasing coroutine resumes well after Redis has deleted
    # the key. A waiter polling the key takes it in that gap. Hold the releaser
    # there on purpose so the waiter always lands inside it.
    do_release = Lock.do_release

    async def release_then_resume_late(self: Lock, expected_token: bytes) -> None:
        await do_release(self, expected_token)
        await asyncio.sleep(0.3)

    monkeypatch.setattr(Lock, "do_release", release_then_resume_late)

    async def hold(seconds: float) -> None:
        async with await store.locked_user_integrations(user_id):
            await asyncio.sleep(seconds)

    first = asyncio.create_task(hold(0.2))
    await asyncio.sleep(0.05)
    # Takes the key while the first holder is still inside its release, and
    # is still holding it when that release finishes.
    second = asyncio.create_task(hold(0.5))
    await asyncio.sleep(0.05)
    third = asyncio.create_task(hold(0))

    await asyncio.wait_for(asyncio.gather(first, second, third), timeout=_STALL)


async def test_concurrent_holders_never_leave_the_lock_behind(
    store: IntegrationCredentialsStore, user_id: str
):
    async def page_load_read() -> None:
        for _ in range(10):
            async with await store.locked_user_integrations(user_id):
                await asyncio.sleep(0.001)

    await asyncio.wait_for(
        asyncio.gather(*(page_load_read() for _ in range(6))), timeout=_STALL * 4
    )
    async with asyncio.timeout(_STALL):
        async with await store.locked_user_integrations(user_id):
            pass


async def test_credential_reads_do_not_wait_for_a_writer(
    store: IntegrationCredentialsStore, user_id: str, mocker
):
    db_manager = MagicMock()
    db_manager.get_user_credentials = AsyncMock(return_value=[])
    mocker.patch.object(
        IntegrationCredentialsStore,
        "db_manager",
        new_callable=PropertyMock,
        return_value=db_manager,
    )

    writer_holds_lock = asyncio.Event()
    finish_write = asyncio.Event()

    async def slow_writer() -> None:
        async with await store.locked_user_integrations(user_id):
            writer_holds_lock.set()
            await finish_write.wait()

    writer = asyncio.create_task(slow_writer())
    await writer_holds_lock.wait()
    try:
        async with asyncio.timeout(1):
            await store.get_all_creds(user_id)
            await store.get_creds_by_provider(user_id, "github")
            await store.get_creds_by_id(user_id, "missing")
    finally:
        finish_write.set()
        await writer
