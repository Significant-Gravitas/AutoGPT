"""Contention must finish before the already-held refresh lease expires."""

import asyncio
import copy
from contextlib import asynccontextmanager

import pytest
from redis.exceptions import LockNotOwnedError

from backend.integrations import credential_refresh as refresh
from backend.integrations import creds_manager as manager_module
from backend.integrations.creds_manager_refresh_test import Lock, Rig


class Clock:
    def __init__(self):
        self.now = 0.0
        self.timers = []

    @asynccontextmanager
    async def timeout(self, seconds):
        scope = asyncio.timeout(None)
        timer = self.now + seconds, scope
        self.timers.append(timer)
        try:
            async with scope:
                yield
        finally:
            self.timers.remove(timer)

    async def advance(self, seconds):
        self.now += seconds
        for deadline, scope in self.timers:
            if deadline <= self.now and not scope.expired():
                scope.reschedule(asyncio.get_running_loop().time() - 1)
        await asyncio.sleep(0)


class ExpiringLock(Lock):
    async def acquire(self):
        await super().acquire()
        self.expires = self.rig.clock.now + refresh.REFRESH_LEASE_SECONDS
        return self

    async def owned(self):
        if self.rig.clock.now >= self.expires:
            if self.rig.owners.get(self.key) == self.token:
                del self.rig.owners[self.key]
            return False
        return await super().owned()

    async def extend(self, seconds, *, replace_ttl):
        assert replace_ttl
        if not await self.owned():
            raise LockNotOwnedError("Synthetic lease expired")
        self.expires = self.rig.clock.now + seconds
        return True


class WaitingRig(Rig):
    def __init__(self, monkeypatch, boundaries):
        super().__init__(monkeypatch)
        self.clock = Clock()
        self.entered = {name: asyncio.Event() for name in boundaries}
        self.resume = {name: asyncio.Event() for name in boundaries}
        monkeypatch.setattr(
            manager_module, "timeout", self.clock.timeout, raising=False
        )

    async def pause(self, boundary):
        if boundary in self.entered:
            self.entered[boundary].set()
            await self.resume[boundary].wait()

    async def acquire(self, key, *, blocking_timeout=None):
        boundary = (
            "refresh"
            if key[-1] == "refresh"
            else "integrations" if key[-1] == "integrations" else "credential"
        )
        async with asyncio.timeout(blocking_timeout):
            await self.pause(boundary)
            return await ExpiringLock(self, str(key)).acquire()

    async def get_creds_by_id(self, _user, _credential):
        await self.pause("first_read")
        return copy.deepcopy(self.saved)

    async def _get_persisted_user_creds_unlocked(self, _user):
        await self.pause("persisted_read")
        return [copy.deepcopy(self.saved)]

    async def refresh(self):
        return await self.manager.refresh_if_needed("test-user", self.saved)

    async def release_all_waits(self, task):
        for event in self.resume.values():
            event.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture(autouse=True)
def reset_refresh_registry(monkeypatch):
    assert not refresh._operations
    monkeypatch.setattr(refresh, "_closing", False)
    yield
    assert not refresh._operations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["first_read", "credential", "integrations", "persisted_read"]
)
async def test_long_preparation_wait_is_retryable_without_exchanging(
    boundary, monkeypatch
):
    rig = WaitingRig(monkeypatch, [boundary])
    before = copy.deepcopy(rig.saved)
    task = asyncio.create_task(rig.refresh())
    try:
        await asyncio.wait_for(rig.entered[boundary].wait(), 5)
        await rig.clock.advance(61)
        rig.resume[boundary].set()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, 5)
        assert rig.saved == before
        assert rig.provider_calls == 0
        assert rig.pending_checkpoint is None
        assert not rig.owners
        assert (await rig.refresh()).generation == 1
        assert rig.provider_calls == 1
    finally:
        await rig.release_all_waits(task)


@pytest.mark.asyncio
async def test_later_waits_share_one_preparation_budget(monkeypatch):
    rig = WaitingRig(monkeypatch, ["credential", "integrations"])
    task = asyncio.create_task(rig.refresh())
    try:
        await asyncio.wait_for(rig.entered["credential"].wait(), 5)
        await rig.clock.advance(10)
        rig.resume["credential"].set()
        await asyncio.wait_for(rig.entered["integrations"].wait(), 5)
        await rig.clock.advance(10)
        rig.resume["integrations"].set()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, 5)
        assert rig.provider_calls == 0 and not rig.owners
        assert not rig.saved.metadata
    finally:
        await rig.release_all_waits(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["credential", "integrations"])
async def test_wait_timeout_releases_still_owned_locks_before_expiry(
    boundary, monkeypatch
):
    rig = WaitingRig(monkeypatch, [boundary])
    task = asyncio.create_task(rig.refresh())
    try:
        await asyncio.wait_for(rig.entered[boundary].wait(), 5)
        await rig.clock.advance(refresh.REFRESH_LEASE_SECONDS / 2 + 1)
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done, "Preparation did not enforce its own deadline"
        with pytest.raises(TimeoutError):
            await task
        assert not rig.resume[boundary].is_set()
        acquired = [event[9:] for event in rig.events if event.startswith("acquired:")]
        released = [event[9:] for event in rig.events if event.startswith("released:")]
        assert released == list(reversed(acquired))
        assert rig.clock.now < refresh.REFRESH_LEASE_SECONDS
        assert not rig.owners and rig.provider_calls == 0
        assert not rig.saved.metadata
    finally:
        await rig.release_all_waits(task)


@pytest.mark.asyncio
async def test_timed_out_default_wait_does_not_release_the_active_credential_lease(
    monkeypatch,
):
    rig = WaitingRig(monkeypatch, ["credential"])
    key = str(("user:test-user", "credentials:test-credential"))
    rig.owners[key] = "active-credential-lease"
    rig.resume["credential"].set()
    task = asyncio.create_task(rig.refresh())
    try:
        await asyncio.wait_for(rig.entered["credential"].wait(), 5)
        await rig.clock.advance(refresh.REFRESH_LEASE_SECONDS / 2 + 1)
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done, "Preparation did not enforce its own deadline"
        with pytest.raises(TimeoutError):
            await task
        assert rig.owners == {key: "active-credential-lease"}
        assert not rig.provider_calls and not rig.saved.metadata
        async with rig.condition:
            del rig.owners[key]
            rig.condition.notify_all()
        assert (await rig.refresh()).generation == 1
        assert not rig.owners
    finally:
        await rig.release_all_waits(task)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["first_read", "credential", "integrations", "persisted_read"]
)
async def test_short_preparation_wait_still_refreshes_once(boundary, monkeypatch):
    rig = WaitingRig(monkeypatch, [boundary])
    task = asyncio.create_task(rig.refresh())
    try:
        await asyncio.wait_for(rig.entered[boundary].wait(), 5)
        await rig.clock.advance(1)
        rig.resume[boundary].set()
        result = await asyncio.wait_for(task, 5)
        assert result.generation == 1 and result == rig.saved
        assert rig.provider_calls == 1 and not rig.owners
    finally:
        await rig.release_all_waits(task)


@pytest.mark.asyncio
async def test_wait_before_first_refresh_lock_does_not_spend_lease_budget(monkeypatch):
    rig = WaitingRig(monkeypatch, ["refresh"])
    task = asyncio.create_task(rig.refresh())
    try:
        await asyncio.wait_for(rig.entered["refresh"].wait(), 5)
        await rig.clock.advance(61)
        rig.resume["refresh"].set()
        assert (await asyncio.wait_for(task, 5)).generation == 1
        assert rig.provider_calls == 1 and not rig.owners
    finally:
        await rig.release_all_waits(task)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary", ["first_read", "credential", "integrations", "persisted_read"]
)
async def test_caller_cancellation_before_exchange_keeps_tokens(boundary, monkeypatch):
    rig = WaitingRig(monkeypatch, [boundary])
    before = copy.deepcopy(rig.saved)
    task = asyncio.create_task(rig.refresh())
    try:
        await asyncio.wait_for(rig.entered[boundary].wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert rig.saved == before and not rig.owners
        assert rig.provider_calls == 0 and rig.pending_checkpoint is None
    finally:
        await rig.release_all_waits(task)
