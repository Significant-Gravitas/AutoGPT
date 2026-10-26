"""Exercise the real manager with rotating providers and a bounded synthetic store."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field, replace
from uuid import uuid4

import pytest

from backend.integrations import credential_refresh as refresh
from backend.integrations import creds_manager as manager_module


@dataclass
class Credential:
    generation: int = 0
    id: str = "test-credential"
    provider: str = "codex"
    type: str = "oauth2"
    refresh_strategy: str = "oauth_handler"
    access_token_expires_at: int = 1
    access_token: str = "synthetic-access"
    refresh_token: str | None = "synthetic-refresh"
    provider_state: str | None = "synthetic-state"
    scopes: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    is_managed: bool = False

    def model_copy(self, *, update):
        return replace(self, **update)


class Lock:
    def __init__(self, rig, key):
        self.rig, self.key = rig, key
        self.token = uuid4().hex

    async def acquire(self):
        async with self.rig.condition:
            await self.rig.condition.wait_for(lambda: self.key not in self.rig.owners)
            self.rig.owners[self.key] = self.token
        self.rig.events.append("acquired:" + self.key)
        return self

    async def extend(self, *_args, **_kwargs):
        if self.rig.lose_lease == self.rig.phase:
            self.rig.owners.pop(self.key, None)
            return False
        return await self.owned()

    async def owned(self):
        return self.rig.owners.get(self.key) == self.token

    async def locked(self):
        return self.key in self.rig.owners

    async def release(self):
        async with self.rig.condition:
            assert await self.owned()
            del self.rig.owners[self.key]
            self.rig.condition.notify_all()
        self.rig.events.append("released:" + self.key)


class Rig:
    def __init__(self, monkeypatch):
        self.saved = Credential()
        self.events = []
        self.owners = {}
        self.condition = asyncio.Condition()
        self.provider_entered = asyncio.Event()
        self.provider_release = asyncio.Event()
        self.provider_release.set()
        self.store_entered = asyncio.Event()
        self.store_release = asyncio.Event()
        self.store_release.set()
        self.begin_entered = asyncio.Event()
        self.begin_release = asyncio.Event()
        self.begin_release.set()
        self.phase = "before"
        self.lose_lease = None
        self.provider_calls = 0
        self.provider_generation = 0
        self.ignore_provider_cancel = False
        self.fail_provider = False
        self.fail_store = False
        self.fail_notify = False
        self.commit_deadlines = []
        self.pending_checkpoint = None

        monkeypatch.setattr(manager_module, "_invoke_creds_changed_hook", self.notify)
        self.manager = object.__new__(manager_module.IntegrationCredentialsManager)
        self.manager.store = self
        self.manager._get_oauth_handler = self.handler
        self.manager._fresh_refresh_lock = self.acquire

    async def handler(self, _credential):
        return self

    ROTATES_REFRESH_TOKEN = True

    def needs_refresh(self, credential):
        return credential.access_token_expires_at < 4_000_000_000

    async def acquire(self, key, *, blocking_timeout=None):
        async with asyncio.timeout(blocking_timeout):
            return await Lock(self, str(key)).acquire()

    async def get_creds_by_id(self, user, _credential):
        lock = await self.acquire((f"user:{user}", "integrations"))
        try:
            return copy.deepcopy(self.saved)
        finally:
            await lock.release()

    async def _get_persisted_user_creds_unlocked(self, _user):
        return [copy.deepcopy(self.saved)]

    async def begin_refresh(self, _user, expected, nonce, _deadline):
        self.begin_entered.set()
        await self.begin_release.wait()
        assert self.saved == expected
        self.saved = replace(
            expected,
            access_token="",
            refresh_token=None,
            provider_state=None,
            metadata={refresh.REFRESH_PENDING: nonce},
        )
        self.pending_checkpoint = nonce
        self.phase = "prepared"
        self.events.append("prepared")
        return nonce

    async def refresh_tokens(self, credential):
        self.provider_calls += 1
        assert credential.generation == self.provider_generation
        self.provider_generation += 1
        self.provider_entered.set()
        self.phase = "accepted"
        self.events.append("accepted")
        try:
            await self.provider_release.wait()
        except asyncio.CancelledError:
            if not self.ignore_provider_cancel:
                raise
            await self.provider_release.wait()
        if self.fail_provider:
            raise ValueError("DO_NOT_LOG_SYNTHETIC_TOKEN")
        return replace(
            credential,
            generation=self.provider_generation,
            access_token_expires_at=4_000_000_000,
        )

    async def complete_refresh(self, _user, updated, checkpoint, deadline):
        self.commit_deadlines.append(deadline)
        self.store_entered.set()
        self.events.append("write_started")
        await self.store_release.wait()
        if self.fail_store:
            raise RuntimeError("DO_NOT_LOG_SYNTHETIC_TOKEN")
        if (
            time.time() >= deadline
            or self.saved.metadata.get(refresh.REFRESH_PENDING) != checkpoint
        ):
            return False
        self.saved = copy.deepcopy(updated)
        self.events.append("committed")
        return True

    async def notify(self, *_args):
        if self.fail_notify:
            raise ValueError("DO_NOT_LOG_SYNTHETIC_TOKEN")

    async def load(self):
        return await self.manager.get("test-user", "test-credential")


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    assert not refresh._operations
    monkeypatch.setattr(refresh, "_closing", False)
    monkeypatch.setattr(manager_module, "_REFRESH_LOCK_WAIT_SECONDS", 0.01)
    yield
    assert not refresh._operations


@pytest.mark.asyncio
async def test_success_checkpoints_once_and_reuses_saved_generation(monkeypatch):
    rig = Rig(monkeypatch)
    assert (await rig.load()).generation == 1
    assert (await rig.load()).generation == 1
    assert rig.provider_calls == 1 and not rig.owners
    assert rig.events.index("committed") < rig.events.index(
        "released:('user:test-user', 'credentials:test-credential')"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["provider", "store"])
async def test_caller_cancellation_keeps_ownership_until_checkpoint(
    boundary, monkeypatch
):
    rig = Rig(monkeypatch)
    getattr(rig, boundary + "_release").clear()
    task = asyncio.create_task(rig.load())
    await asyncio.wait_for(getattr(rig, boundary + "_entered").wait(), 1)
    for _ in range(5):
        task.cancel()
        await asyncio.sleep(0)
        assert len(rig.owners) == 3 and not task.done()
    getattr(rig, boundary + "_release").set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.saved.generation == 1 and rig.provider_calls == 1
    assert not rig.owners and (await rig.load()).generation == 1


@pytest.mark.asyncio
async def test_cancelled_caller_and_concurrent_loader_share_one_exchange(monkeypatch):
    rig = Rig(monkeypatch)
    rig.provider_release.clear()
    first = asyncio.create_task(rig.load())
    await asyncio.wait_for(rig.provider_entered.wait(), 1)
    second = asyncio.create_task(rig.load())
    first.cancel()
    await asyncio.sleep(0)
    assert not second.done()
    rig.provider_release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await second).generation == 1 and rig.provider_calls == 1


@pytest.mark.asyncio
async def test_cancellation_before_admission_never_calls_provider(monkeypatch):
    rig = Rig(monkeypatch)
    held = await rig.acquire(
        ("user:test-user", "credentials:test-credential", "refresh")
    )
    task = asyncio.create_task(rig.load())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await held.release()
    assert not rig.provider_calls and rig.saved.generation == 0
    assert not rig.owners


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "prepared", "accepted"])
async def test_lease_loss_never_writes_fresh_after_losing_ownership(phase, monkeypatch):
    rig = Rig(monkeypatch)
    rig.lose_lease = phase
    with pytest.raises(refresh.CredentialRefreshRequired):
        await rig.load()
    assert not rig.owners
    if phase == "accepted":
        assert rig.saved.metadata.get(refresh.REFRESH_PENDING)
        assert rig.saved.refresh_token is None and rig.saved.provider_state is None
        assert rig.provider_calls == 1 and not rig.commit_deadlines
        with pytest.raises(refresh.CredentialRefreshRequired):
            await rig.load()
        assert rig.provider_calls == 1
    else:
        assert rig.provider_calls == 0 and not rig.saved.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "store"])
async def test_uncertain_failure_is_durable_and_does_not_log_tokens(
    failure, caplog, monkeypatch
):
    rig = Rig(monkeypatch)
    setattr(rig, "fail_" + failure, True)
    with pytest.raises(refresh.CredentialRefreshRequired, match="Reconnect"):
        await rig.load()
    assert refresh.REFRESH_PENDING in rig.saved.metadata
    assert rig.saved.access_token == "" and rig.saved.refresh_token is None
    with pytest.raises(refresh.CredentialRefreshRequired):
        await rig.load()
    assert rig.provider_calls == 1
    assert "DO_NOT_LOG_SYNTHETIC_TOKEN" not in caplog.text


@pytest.mark.asyncio
async def test_postcommit_hook_failure_never_repeats_exchange(caplog, monkeypatch):
    rig = Rig(monkeypatch)
    rig.fail_notify = True
    assert (await rig.load()).generation == 1
    assert (await rig.load()).generation == 1
    assert rig.provider_calls == 1 and "DO_NOT_LOG_SYNTHETIC_TOKEN" not in caplog.text


@pytest.mark.asyncio
async def test_late_checkpoint_cannot_replace_reconnected_credential(monkeypatch):
    rig = Rig(monkeypatch)
    rig.store_release.clear()
    task = asyncio.create_task(rig.load())
    await asyncio.wait_for(rig.store_entered.wait(), 1)
    rig.saved = replace(
        Credential(), generation=99, access_token_expires_at=4_000_000_000
    )
    rig.store_release.set()
    with pytest.raises(refresh.CredentialRefreshRequired):
        await task
    assert (await rig.load()).generation == 99
    assert rig.provider_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["provider", "store"])
async def test_absolute_deadline_leaves_pending_and_no_writer(boundary, monkeypatch):
    monkeypatch.setattr(refresh, "REFRESH_SECONDS", 0.04)
    rig = Rig(monkeypatch)
    getattr(rig, boundary + "_release").clear()
    with pytest.raises(refresh.CredentialRefreshRequired):
        await asyncio.wait_for(rig.load(), 1)
    getattr(rig, boundary + "_release").set()
    await asyncio.sleep(0)
    assert not rig.owners and not refresh._operations
    assert rig.saved.metadata.get(refresh.REFRESH_PENDING)
    assert "committed" not in rig.events


@pytest.mark.asyncio
async def test_noncooperative_provider_is_retained_and_cannot_commit_after_deadline(
    monkeypatch,
):
    monkeypatch.setattr(refresh, "REFRESH_SECONDS", 0.02)
    monkeypatch.setattr(refresh, "REFRESH_CLEANUP_SECONDS", 0.02)
    rig = Rig(monkeypatch)
    rig.provider_release.clear()
    rig.ignore_provider_cancel = True
    with pytest.raises(refresh.CredentialRefreshRequired):
        await asyncio.wait_for(rig.load(), 1)
    rig.provider_release.set()
    await asyncio.sleep(0.01)
    assert not refresh._operations and not rig.owners
    assert "committed" not in rig.events


@pytest.mark.asyncio
async def test_shutdown_closes_admission_and_waits_before_loop_cleanup(monkeypatch):
    rig = Rig(monkeypatch)
    rig.provider_release.clear()
    caller = asyncio.create_task(rig.load())
    await asyncio.wait_for(rig.provider_entered.wait(), 1)
    drain = asyncio.create_task(
        asyncio.to_thread(refresh.drain_credential_refreshes, 1)
    )
    for _ in range(100):
        if refresh._closing:
            break
        await asyncio.sleep(0.001)
    assert refresh._closing and not drain.done()
    other = Rig(monkeypatch)
    with pytest.raises(refresh.CredentialRefreshStopping):
        await other.load()
    assert other.provider_calls == 0 and not other.owners
    caller.cancel()
    rig.provider_release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert await drain and rig.saved.generation == 1


@pytest.mark.asyncio
async def test_shutdown_deadline_is_bounded_and_repeated_stop_cannot_reopen_admission(
    monkeypatch,
):
    rig = Rig(monkeypatch)
    rig.provider_release.clear()
    caller = asyncio.create_task(rig.load())
    await asyncio.wait_for(rig.provider_entered.wait(), 1)
    assert not await asyncio.to_thread(refresh.drain_credential_refreshes, 0.01)
    assert not await asyncio.to_thread(refresh.drain_credential_refreshes, 0)
    assert refresh._closing and len(refresh._operations) == 1
    rig.provider_release.set()
    await caller
    assert await asyncio.to_thread(refresh.drain_credential_refreshes, 0.01)


@pytest.mark.asyncio
async def test_repeated_cancel_during_pre_admission_cleanup_releases_every_lock(
    monkeypatch,
):
    rig = Rig(monkeypatch)
    reading = asyncio.Event()
    releasing = asyncio.Event()
    allow_release = asyncio.Event()
    original_release = Lock.release

    async def read(_user):
        reading.set()
        await asyncio.Event().wait()

    async def release(lock):
        if reading.is_set() and lock.key.endswith("'integrations')"):
            releasing.set()
            await allow_release.wait()
        await original_release(lock)

    rig._get_persisted_user_creds_unlocked = read
    monkeypatch.setattr(Lock, "release", release)
    task = asyncio.create_task(rig.load())
    await reading.wait()
    task.cancel()
    await releasing.wait()
    for _ in range(4):
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and len(rig.owners) == 3
    allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.provider_calls == 0 and not rig.owners


@pytest.mark.asyncio
async def test_repeated_cancel_does_not_extend_absolute_exchange_deadline(monkeypatch):
    monkeypatch.setattr(refresh, "REFRESH_SECONDS", 0.04)
    rig = Rig(monkeypatch)
    rig.provider_release.clear()
    task = asyncio.create_task(rig.load())
    await rig.provider_entered.wait()
    started = time.monotonic()
    for _ in range(4):
        task.cancel()
        await asyncio.sleep(0.005)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 0.5
    assert not rig.owners and rig.saved.metadata.get(refresh.REFRESH_PENDING)


@pytest.mark.asyncio
async def test_mutating_provider_cannot_change_the_checkpoint_identity(monkeypatch):
    rig = Rig(monkeypatch)

    async def mutate(credential):
        credential.id = "different-id"
        return credential

    rig.refresh_tokens = mutate
    with pytest.raises(refresh.CredentialRefreshRequired):
        await rig.load()
    assert rig.saved.id == "test-credential" and not rig.commit_deadlines


@pytest.mark.asyncio
@pytest.mark.parametrize("key_suffix", ["refresh", "credential", "integrations"])
async def test_rotating_lock_timeout_releases_only_acquired_locks(
    key_suffix, monkeypatch
):
    rig = Rig(monkeypatch)
    key = (
        ("user:test-user", "integrations")
        if key_suffix == "integrations"
        else ("user:test-user", "credentials:test-credential")
        + (("refresh",) if key_suffix == "refresh" else ())
    )
    held = await rig.acquire(key)
    if key_suffix == "integrations":

        async def read_without_lock(*_args):
            return copy.deepcopy(rig.saved)

        rig.get_creds_by_id = read_without_lock
    lookup = asyncio.create_task(
        rig.manager.refresh_if_needed("test-user", rig.saved, lock=False)
    )
    try:
        done, _ = await asyncio.wait({lookup}, timeout=0.5)
        assert lookup in done, "Rotating lookup ignored its acquisition timeout"
        with pytest.raises(TimeoutError):
            await lookup
        assert await held.owned()
        assert len(rig.owners) == 1
        assert not rig.provider_calls
        assert not rig.saved.metadata
    finally:
        lookup.cancel()
        await asyncio.gather(lookup, return_exceptions=True)
        await held.release()
