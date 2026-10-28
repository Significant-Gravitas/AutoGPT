"""The per-user credentials lock against a real Redis cluster.

Every copilot page load reads the user's credentials several times at once:
``/chat/transports``, ``/chat/connections``, ``/integrations/credentials`` and
``POST /chat/sessions``. When one holder of the per-user ``integrations`` lock
skipped its release, the key stayed until its 60 s TTL ran out, and each of
those requests hung for a minute. In prod they hit the frontend proxy's 30 s
timeout, and the copilot Playwright spec failed the same way in CI.
"""

import asyncio
import contextlib
import socket
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from redis.asyncio.lock import Lock

from backend.data import redis_client
from backend.integrations.credentials_store import IntegrationCredentialsStore


def _has_live_cluster() -> bool:
    """Probe the socket, not ``redis_client.connect()``: this runs at collection
    time and that helper retries a dead host for ~45 minutes before giving up."""
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


class _Holders:
    """Holds the user's lock and records how many holders were ever inside at once."""

    def __init__(self, store: IntegrationCredentialsStore, user_id: str):
        self.store = store
        self.user_id = user_id
        self.inside = 0
        self.most_inside = 0

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        async with await self.store.locked_user_integrations(self.user_id):
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
            try:
                yield
            finally:
                self.inside -= 1


async def _wait_for_event(event: asyncio.Event, what: str) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=_STALL)
    except TimeoutError:
        pytest.fail(f"timed out after {_STALL}s waiting for {what}")


async def _cancel(*tasks: asyncio.Task) -> None:
    """Let finished tasks settle and cancel stuck ones, without hanging the shard."""
    if pending := [task for task in tasks if not task.done()]:
        _, stuck = await asyncio.wait(pending, timeout=_STALL)
        for task in stuck:
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.fixture
def store() -> IntegrationCredentialsStore:
    redis_client._async_clients.clear()
    return IntegrationCredentialsStore()


@pytest.fixture
def user_id() -> str:
    return f"lock-test-{uuid.uuid4()}"


@pytest.fixture
def holders(store: IntegrationCredentialsStore, user_id: str) -> _Holders:
    return _Holders(store, user_id)


async def test_a_waiter_that_takes_the_lock_during_a_release_still_releases_it(
    holders: _Holders, monkeypatch
):
    # Under load, a releasing coroutine resumes well after Redis has deleted
    # the key, and a waiter polling the key takes it in that gap. Park the
    # first release there until the waiter holds the key.
    waiter_holds_lock = asyncio.Event()
    releaser_may_resume = asyncio.Event()
    parked: list[Lock] = []
    do_release = Lock.do_release

    async def release_then_park_once(self: Lock, expected_token: bytes) -> None:
        await do_release(self, expected_token)
        if not parked:
            parked.append(self)
            await releaser_may_resume.wait()

    monkeypatch.setattr(Lock, "do_release", release_then_park_once)

    first_holds_lock = asyncio.Event()
    first_may_leave = asyncio.Event()
    waiter_may_leave = asyncio.Event()

    async def first() -> None:
        async with holders.hold():
            first_holds_lock.set()
            await first_may_leave.wait()

    async def waiter() -> None:
        async with holders.hold():
            waiter_holds_lock.set()
            await waiter_may_leave.wait()

    first_task = asyncio.create_task(first())
    tasks = [first_task]
    try:
        await _wait_for_event(first_holds_lock, "the first holder to take the lock")
        waiter_task = asyncio.create_task(waiter())
        tasks.append(waiter_task)
        first_may_leave.set()

        await _wait_for_event(waiter_holds_lock, "the waiter to take the lock")
        releaser_may_resume.set()
        await asyncio.wait_for(first_task, timeout=_STALL)
        waiter_may_leave.set()
        await asyncio.wait_for(waiter_task, timeout=_STALL)
    finally:
        for event in (first_may_leave, releaser_may_resume, waiter_may_leave):
            event.set()
        await _cancel(*tasks)

    # The next request for this user must not wait out the lock's TTL.
    async with asyncio.timeout(_STALL):
        async with holders.hold():
            pass
    assert holders.most_inside == 1


async def test_concurrent_holders_never_leave_the_lock_behind(holders: _Holders):
    async def page_load_read() -> None:
        for _ in range(10):
            async with holders.hold():
                await asyncio.sleep(0)

    async with asyncio.timeout(_STALL * 4):
        await asyncio.gather(*(page_load_read() for _ in range(6)))
    async with asyncio.timeout(_STALL):
        async with holders.hold():
            pass
    assert holders.most_inside == 1


async def test_cancelling_acquire_after_redis_grants_the_key_does_not_orphan_it(
    store: IntegrationCredentialsStore, holders: _Holders, user_id: str, monkeypatch
):
    # Webhook setup wraps acquire() in wait_for and frees the key with
    # release(key). Cancel the acquire the moment Redis has granted the key,
    # while something else (e.g. release_all_locks at shutdown) holds the
    # mutex's bookkeeping lock.
    granted = asyncio.Event()
    do_acquire = Lock.do_acquire

    async def acquire_and_signal(self: Lock, token: bytes) -> bool:
        acquired = await do_acquire(self, token)
        if acquired:
            granted.set()
        return acquired

    monkeypatch.setattr(Lock, "do_acquire", acquire_and_signal)
    mutex = await store.locks()
    key = (f"user:{user_id}", "integrations")

    async with mutex.locks_lock:
        acquiring = asyncio.create_task(mutex.acquire(key))
        await asyncio.wait_for(granted.wait(), timeout=_STALL)
        acquiring.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await acquiring
        await mutex.release(key)

    async with asyncio.timeout(_STALL):
        async with holders.hold():
            pass


async def test_a_holder_whose_ttl_ran_out_cannot_release_its_successor(
    store: IntegrationCredentialsStore, user_id: str
):
    mutex = await store.locks()
    key = (f"user:{user_id}", "integrations")

    async with mutex.locked(key):
        # The first holder overran the TTL; Redis dropped its key.
        await mutex.redis.delete(str(key))
        successor = await asyncio.wait_for(mutex.acquire(key), timeout=_STALL)

    still_owned = await successor.owned()
    if still_owned:
        await successor.release()
    assert still_owned, "the expired holder released its successor's lock"


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
    try:
        await _wait_for_event(writer_holds_lock, "the writer to take the lock")
        async with asyncio.timeout(1):
            await store.get_all_creds(user_id)
            await store.get_creds_by_provider(user_id, "github")
            await store.get_creds_by_id(user_id, "missing")
    finally:
        finish_write.set()
        await _cancel(writer)
