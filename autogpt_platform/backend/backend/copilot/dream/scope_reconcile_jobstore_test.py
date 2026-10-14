"""The memory cron reconcile against the real scheduler and a real jobstore
(SQLite, or PostgreSQL when ``DIRECT_URL`` names one), with the registry's
fakes: an account cron's delete racing a restore, whole or cut short, and a
write that finds its row locked.

The probes are an independent review's. Round 3's: a restore that read the
row before a successful delete re-created the deleted cron for good, and a
write blocked on a locked row held its pass, and every pass after it, at
the same scope. Round 4's: that restore cut short after its stale add, by
a failed or locked write or a cancellation. A stale add is tagged as the
reconcile's, and the next pass takes it back.
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import create_engine, text

from backend.executor import scheduler as sm
from backend.executor.jobstore import ResilientSQLAlchemyJobStore

from . import registry, scope_reconcile
from .registry_fakes_test import (
    ACCOUNT,
    PARIS,
    PAUSED,
    TOKYO,
    USER,
    RegistryEnv,
    install_registry_env,
    job_ids,
)

_EXECUTION = sm.Jobstores.EXECUTION.value
TIMEOUT = 5
_NIGHTLY = f"dream_nightly_batch_{ACCOUNT.scope_key}"
_COMMUNITY = f"community_rebuild_{ACCOUNT.scope_key}"


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    return install_registry_env(monkeypatch)


@contextmanager
def _scheduler(store: ResilientSQLAlchemyJobStore) -> Iterator[sm.Scheduler]:
    """A ``Scheduler`` whose APScheduler, started paused, keeps its EXECUTION
    jobs in ``store``."""
    apscheduler = BackgroundScheduler(
        jobstores={_EXECUTION: store}, timezone=timezone.utc
    )
    apscheduler.start(paused=True)
    service = sm.Scheduler(register_system_tasks=False)
    service.scheduler, service._execution_jobstore = apscheduler, store
    try:
        yield service
    finally:
        apscheduler.remove_jobstore(_EXECUTION)
        apscheduler.shutdown(wait=True)
        store.engine.dispose()


def _sqlite(tmp_path: Path) -> ResilientSQLAlchemyJobStore:
    return ResilientSQLAlchemyJobStore(
        url=f"sqlite:///{(tmp_path / 'jobs.sqlite').as_posix()}"
    )


def _add_crons(service: sm.Scheduler, scope, timezone_name: str) -> None:
    for job in sm.DREAM_SYSTEM_JOBS:
        service._add_memory_job(
            *sm._MEMORY_CRONS[job.job_id_prefix], job, scope, timezone_name
        )


def _job(service: sm.Scheduler, job_id: str):
    return service.scheduler.get_job(job_id, jobstore=_EXECUTION)


async def _pass(service: sm.Scheduler) -> scope_reconcile.ReconcileReport:
    return await scope_reconcile.reconcile(sm._SchedulerCrons(service))


async def _set(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_delete_after_a_restores_read_is_not_undone(env, monkeypatch, tmp_path):
    """The review's probe: a restore re-timing the account's crons parks
    after its read, just before writing the nightly cron; the delete RPC
    forgets the cron and removes it; the restore then writes it back,
    tagged as the reconcile's. The next pass takes it back, and later
    passes leave it deleted."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    monkeypatch.setattr(sm, "_event_loop", asyncio.get_running_loop())
    with _scheduler(_sqlite(tmp_path)) as service:
        _add_crons(service, ACCOUNT, TOKYO)
        reached, release = threading.Event(), threading.Event()
        add = service._add_memory_job

        def parked_add(*args):
            if args[2].job_id_prefix == "dream_nightly_batch" and not reached.is_set():
                reached.set()
                assert release.wait(5)
            return add(*args)

        monkeypatch.setattr(service, "_add_memory_job", parked_add)
        restore = asyncio.create_task(_pass(service))
        await asyncio.wait_for(_set(reached), TIMEOUT)
        assert await asyncio.to_thread(service.delete_nightly_batch_schedule, USER)
        assert env.db.rows[ACCOUNT.scope_key].nightly_job_id is None
        release.set()
        await asyncio.wait_for(restore, 5)
        for _ in range(3):
            await _pass(service)

        assert _job(service, _NIGHTLY) is None
        assert str(_job(service, _COMMUNITY).trigger.timezone) == PARIS


@pytest.mark.asyncio
async def test_a_delete_forgets_the_cron_before_it_removes_it(
    env, monkeypatch, tmp_path
):
    """A pass that runs between the delete's two steps finds the row no
    longer recording the cron and leaves it, so the delete's removal is the
    last word. Removing first, the pass would restore the cron the row
    still records, and the forget after it would leave it running for
    good."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    monkeypatch.setattr(sm, "_event_loop", asyncio.get_running_loop())
    with _scheduler(_sqlite(tmp_path)) as service:
        _add_crons(service, ACCOUNT, PARIS)
        first, release = threading.Event(), threading.Event()
        forget, removed = sm._forget_dream_registration, sm._removed

        def parked(step):
            def run(*args):
                done = step(*args)
                if not first.is_set():
                    first.set()
                    assert release.wait(5)
                return done

            return run

        monkeypatch.setattr(sm, "_forget_dream_registration", parked(forget))
        monkeypatch.setattr(sm, "_removed", parked(removed))
        delete = asyncio.create_task(
            asyncio.to_thread(service.delete_nightly_batch_schedule, USER)
        )
        await asyncio.wait_for(_set(first), TIMEOUT)
        await asyncio.wait_for(_pass(service), TIMEOUT)
        release.set()

        assert await asyncio.wait_for(delete, 5)
        await _pass(service)
        assert _job(service, _NIGHTLY) is None
        assert env.db.rows[ACCOUNT.scope_key].nightly_job_id is None


@pytest.mark.asyncio
async def test_a_delete_whose_forget_fails_keeps_its_cron(env, monkeypatch, tmp_path):
    """The row still records the cron, so removing it would only have the
    reconcile restore it: the delete says it failed and leaves it."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    monkeypatch.setattr(sm, "_event_loop", asyncio.get_running_loop())
    monkeypatch.setattr(
        env.db, "forget_scope_job", AsyncMock(side_effect=ConnectionError("down"))
    )
    with _scheduler(_sqlite(tmp_path)) as service:
        _add_crons(service, ACCOUNT, PARIS)

        assert not await asyncio.to_thread(service.delete_nightly_batch_schedule, USER)

        assert _job(service, _NIGHTLY) is not None
        assert env.db.rows[ACCOUNT.scope_key].nightly_job_id == _NIGHTLY


# The flow a step belongs to: set where each flow starts, carried into its
# threads and tasks with the context.
_FLOW: ContextVar[str] = ContextVar("reconcile_test_flow")


class _Turns:
    """Runs flows one step at a time: each wrapped call waits its turn, and
    once every running flow is waiting, one of them, picked at random
    (seeded) by the flow's name, takes its step. A seed fixes the order of
    the flows' steps, whatever their timing."""

    def __init__(self, rng: random.Random, flows: list[str]) -> None:
        self._rng = rng
        self._lock = threading.Lock()
        self._waiting: dict[str, threading.Event] = {}
        self._running = set(flows)
        self._on = True

    def step(self, call):
        def run(*args, **kwargs):
            assert self._turn().wait(TIMEOUT), "no turn came"
            return call(*args, **kwargs)

        return run

    def astep(self, call):
        async def run(*args, **kwargs):
            await asyncio.wait_for(_set(self._turn()), TIMEOUT)
            return await call(*args, **kwargs)

        return run

    def finished(self, flow: str) -> None:
        with self._lock:
            self._running.discard(flow)
            self._next()

    def off(self) -> None:
        with self._lock:
            self._on = False
            for turn in self._waiting.values():
                turn.set()
            self._waiting.clear()

    def _turn(self) -> threading.Event:
        turn = threading.Event()
        with self._lock:
            if not self._on:
                turn.set()
                return turn
            self._waiting[_FLOW.get()] = turn
            self._next()
        return turn

    def _next(self) -> None:
        if self._waiting and self._waiting.keys() >= self._running:
            flow = self._rng.choice(sorted(self._waiting))
            self._waiting.pop(flow).set()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["healthy", "add_fails", "cancelled"])
@pytest.mark.parametrize("seed", range(20))
async def test_no_interleaving_of_a_restore_and_a_delete_leaves_a_cron_unrecorded(
    env, monkeypatch, tmp_path, seed: int, ending: str
):
    """A restore re-timing the account's crons races the nightly cron's
    delete, their registry reads and writes and jobstore writes taken one at
    a time in a seeded order. The restore runs to its end, its second add
    fails, or it is cancelled there. Once both are done, and two more
    passes, the scheduler holds exactly the crons the row records: after the
    delete, never the nightly one."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    monkeypatch.setattr(sm, "_event_loop", asyncio.get_running_loop())
    turns = _Turns(random.Random(seed), ["restore", "delete"])
    monkeypatch.setattr(
        env.db, "get_scope_schedule", turns.astep(env.db.get_scope_schedule)
    )
    monkeypatch.setattr(
        env.db, "forget_scope_job", turns.astep(env.db.forget_scope_job)
    )
    with _scheduler(_sqlite(tmp_path)) as service:
        _add_crons(service, ACCOUNT, TOKYO)
        add = turns.step(service._add_memory_job)
        adds = {"made": 0}

        def ending_add(*args):
            if adds["made"] == 1 and ending == "add_fails":
                raise ConnectionError("the second add failed")
            if adds["made"] == 1 and ending == "cancelled":
                raise asyncio.CancelledError("the pass's ceiling")
            adds["made"] += 1
            return add(*args)

        monkeypatch.setattr(service, "_add_memory_job", ending_add)
        monkeypatch.setattr(
            service.scheduler, "remove_job", turns.step(service.scheduler.remove_job)
        )

        async def restoring() -> None:
            _FLOW.set("restore")
            try:
                await _pass(service)
            except asyncio.CancelledError:
                pass
            finally:
                turns.finished("restore")

        def deleting() -> bool:
            _FLOW.set("delete")
            try:
                return service.delete_nightly_batch_schedule(USER)
            finally:
                turns.finished("delete")

        await asyncio.wait_for(
            asyncio.gather(restoring(), asyncio.to_thread(deleting)), 4 * TIMEOUT
        )
        turns.off()
        monkeypatch.setattr(service, "_add_memory_job", add)
        for _ in range(2):
            await _pass(service)

        row = env.db.rows[ACCOUNT.scope_key]
        assert row.nightly_job_id is None, "the delete forgot it"
        assert _job(service, _NIGHTLY) is None
        assert _job(service, _COMMUNITY) is not None
        assert str(_job(service, _COMMUNITY).trigger.timezone) == PARIS


def _postgres_url() -> str | None:
    direct = os.getenv("DIRECT_URL") or ""
    return direct.split("?")[0] if direct.startswith("postgresql") else None


@pytest.mark.skipif(_postgres_url() is None, reason="needs the PostgreSQL test DB")
@pytest.mark.asyncio
async def test_a_write_held_by_a_locked_row_fails_its_scope_and_the_pass_goes_on(
    env, monkeypatch
):
    """The review's blocked write on PostgreSQL: another transaction holds
    the first scope's community cron row while every scope needs repair.
    That scope's write gives up after the write timeout and fails it; the
    same pass re-times the next scope and removes a paused one's crons, and
    leaves no write waiting on the row. The next cycle repairs the first."""
    monkeypatch.setattr(sm, "MEMORY_CRON_WRITE_TIMEOUT_SECONDS", 0.5)
    locked, later, paused = (
        registry.MemoryScope.for_user(f"pg-cut-{n}") for n in (0, 1, 9)
    )
    for scope in (locked, later, paused):
        await registry.ensure_scope_scheduled(scope)
    await env.db.set_scope_state(paused.owner_user_id, paused.scope_key, PAUSED)
    name = f"memcore_reconcile_{uuid.uuid4().hex[:8]}"
    url = _postgres_url()
    assert url is not None
    engine = create_engine(
        url, pool_size=4, max_overflow=0, connect_args={"application_name": name}
    )
    locker = create_engine(url).connect()
    try:
        with _scheduler(
            ResilientSQLAlchemyJobStore(engine=engine, tablename=name)
        ) as service:
            for scope in (locked, later, paused):
                _add_crons(service, scope, TOKYO)
            hold = locker.begin()
            try:
                locker.execute(
                    text(f"SELECT id FROM {name} WHERE id = :id FOR UPDATE"),
                    {"id": f"community_rebuild_{locked.scope_key}"},
                ).all()
                started = time.monotonic()

                report = await asyncio.wait_for(_pass(service), 20)

                assert time.monotonic() - started < 10
                assert report.failed == [locked.scope_key]
                assert report.removed == [paused.scope_key]
                zones = {
                    job.id: str(job.trigger.timezone) for job in service._memory_jobs()
                }
                assert {zones[job_id] for job_id in job_ids(later)} == {PARIS}
                assert {zones[job_id] for job_id in job_ids(locked)} == {TOKYO}
                assert not job_ids(paused) & zones.keys()
                waiting = locker.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE application_name = :name "
                        "AND wait_event_type = 'Lock'"
                    ),
                    {"name": name},
                ).scalar()
                assert waiting == 0, "no write is left waiting on the row"
            finally:
                # Before the scheduler shuts down: a write still waiting on
                # the row holds the jobstores lock its shutdown needs.
                hold.rollback()

            await _pass(service)

            zones = {
                job.id: str(job.trigger.timezone) for job in service._memory_jobs()
            }
            assert {zones[job_id] for job_id in job_ids(locked)} == {PARIS}
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {name}"))
    finally:
        locker.close()
        locker.engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["error", "cancel", "pg_timeout"])
async def test_a_restore_cut_short_after_a_stale_add_leaves_no_cron_behind(
    env, monkeypatch, tmp_path, ending: str
):
    """The round-4 review's partial restore, with the production delete and
    jobstore: the restore re-timing the account's crons parks before its
    community add, the delete RPC forgets that cron and removes it, and the
    add then puts it back. The nightly add after it fails, is cancelled, or
    waits on a row another transaction holds (on PostgreSQL) until the write
    timeout. The community cron came back tagged as the reconcile's, so the
    next pass takes it back, and three more leave it gone."""
    if ending == "pg_timeout" and _postgres_url() is None:
        pytest.skip("needs the PostgreSQL test DB")
    await registry.ensure_scope_scheduled(ACCOUNT)
    monkeypatch.setattr(sm, "_event_loop", asyncio.get_running_loop())
    name = f"memcore_partial_{uuid.uuid4().hex[:8]}"
    if ending == "pg_timeout":
        monkeypatch.setattr(sm, "MEMORY_CRON_WRITE_TIMEOUT_SECONDS", 0.2)
        url = _postgres_url()
        assert url is not None
        store = ResilientSQLAlchemyJobStore(url=url, tablename=name)
    else:
        store = _sqlite(tmp_path)
    locker = store.engine.connect() if ending == "pg_timeout" else None
    try:
        with _scheduler(store) as service:
            _add_crons(service, ACCOUNT, TOKYO)
            reached, release = threading.Event(), threading.Event()
            add = service._add_memory_job

            def parked_community(*args):
                if args[2].job_id_prefix == "community_rebuild":
                    reached.set()
                    assert release.wait(TIMEOUT)
                elif ending == "error":
                    raise ConnectionError("the nightly add failed")
                elif ending == "cancel":
                    raise asyncio.CancelledError("the pass's ceiling")
                return add(*args)

            monkeypatch.setattr(service, "_add_memory_job", parked_community)
            hold = locker.begin() if locker is not None else None
            try:
                if locker is not None:
                    locker.execute(
                        text(f"SELECT id FROM {name} WHERE id = :id FOR UPDATE"),
                        {"id": _NIGHTLY},
                    ).all()
                restore = asyncio.create_task(_pass(service))
                await asyncio.wait_for(_set(reached), TIMEOUT)
                assert await asyncio.to_thread(
                    service.delete_community_rebuild_schedule, USER
                )
                release.set()
                if ending == "cancel":
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(restore, 10)
                else:
                    report = await asyncio.wait_for(restore, 10)
                    assert report.failed == [ACCOUNT.scope_key]
            finally:
                release.set()
                if hold is not None:
                    hold.rollback()
            assert _job(service, _COMMUNITY) is not None, "the stale add landed"
            monkeypatch.setattr(service, "_add_memory_job", add)

            report = await _pass(service)
            assert report.dropped == [_COMMUNITY]
            for _ in range(3):
                await _pass(service)

            assert env.db.rows[ACCOUNT.scope_key].community_job_id is None
            assert _job(service, _COMMUNITY) is None
            assert str(_job(service, _NIGHTLY).trigger.timezone) == PARIS
            if ending == "pg_timeout":
                with store.engine.begin() as connection:
                    connection.execute(text(f"DROP TABLE IF EXISTS {name}"))
    finally:
        if locker is not None:
            locker.close()
