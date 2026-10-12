"""Every due run takes effect exactly once across a shutdown and a restart.

Real APScheduler, the ledger-recording thread pool, a persistent SQLAlchemy
jobstore (SQLite) and dispatches that run as coroutines on a separate event
loop, like the scheduler's own. A dispatch's effect is recorded inside
``committing``. The second scheduler stands in for the pod that replaces the
stopped one.
"""

import asyncio
import random
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from apscheduler.job import Job
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_STOPPED

from backend.executor import scheduler as scheduler_module
from backend.executor import scheduler_runs
from backend.executor.scheduler_runs import (
    LedgerThreadPoolExecutor,
    RunAbandoned,
    RunLedger,
    committing,
    current_run,
    dispatching,
)
from backend.executor.scheduler_shutdown import (
    DrainingBackgroundScheduler,
    stop_scheduler,
)

RunKey = tuple[str, datetime | None]


class _World:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        # Per job: seconds stalled before and after the effect, first run only.
        self.stalls: dict[str, tuple[float, float]] = {}
        self.stalling = True
        self.effects: list[RunKey] = []
        self.lock = threading.Lock()


_world: _World


def dispatch(name: str) -> None:
    run = current_run()
    asyncio.run_coroutine_threadsafe(_dispatch(run, name), _world.loop).result(60)


async def _dispatch(run, name: str) -> None:
    before, after = _world.stalls.get(name, (0.0, 0.0)) if _world.stalling else (0, 0)
    await asyncio.sleep(before)
    dispatching(run)
    with committing():
        with _world.lock:
            _world.effects.append((name, run.run_time if run else None))
    await asyncio.sleep(after)


@pytest.fixture
def world():
    global _world
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    _world = _World(loop=loop)
    yield _world
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


# Where a job pass stalls: before submitting a run, or after submitting it but
# before advancing or removing the job.
Stall = tuple[float, bool]


class _StallingExecutor(LedgerThreadPoolExecutor):
    """Holds a job pass (and so the jobstore lock) around one submission, like
    a pass stuck on a slow jobstore mid-way."""

    def __init__(self, max_workers: int, ledger: RunLedger, stall: dict[str, Stall]):
        super().__init__(max_workers=max_workers, ledger=ledger)
        self.stall = stall
        self.in_stall = threading.Event()

    def _do_submit_job(self, job: Job, run_times: list[datetime]) -> None:
        seconds, after_submit = self.stall.pop(job.id, (0.0, False))
        if after_submit:
            super()._do_submit_job(job, run_times)
        if seconds:
            self.in_stall.set()
            time.sleep(seconds)
        if not after_submit:
            super()._do_submit_job(job, run_times)


def _scheduler(
    tmp_path: Path, workers: int = 4, stall: dict[str, Stall] | None = None
) -> tuple[DrainingBackgroundScheduler, _StallingExecutor]:
    ledger = RunLedger(commit_aware=lambda job: job.func is dispatch)
    executor = _StallingExecutor(workers, ledger, stall or {})
    scheduler = DrainingBackgroundScheduler(
        ledger=ledger,
        executors={"default": executor},
        jobstores={
            "default": SQLAlchemyJobStore(url=f"sqlite:///{tmp_path / 'jobs.sqlite'}")
        },
        job_defaults={"coalesce": True, "misfire_grace_time": None},
        timezone=timezone.utc,
    )
    scheduler.start()
    return scheduler, executor


def _restart_and_settle(tmp_path: Path, world: _World, expected: set[RunKey]):
    """Start the next scheduler on the same jobstore and let it catch up."""
    world.stalling = False
    scheduler, _ = _scheduler(tmp_path)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not expected <= set(world.effects):
        time.sleep(0.02)
    time.sleep(0.3)  # long enough for a second, duplicate run to show up
    assert stop_scheduler(scheduler, world.loop, 5.0)


def _wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def _at(seconds: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def test_a_pass_outliving_the_drain_still_runs_its_one_shot_once(
    world: _World, tmp_path: Path
):
    # The pass submits ``once``, whose effect lands at once, then stalls before
    # removing it for longer than the drain waits.
    scheduler, executor = _scheduler(tmp_path, stall={"once": (0.8, True)})
    due = _at(0.05)
    scheduler.add_job(dispatch, "date", id="once", run_date=due, args=["once"])
    assert executor.in_stall.wait(timeout=5)

    stopped = stop_scheduler(scheduler, world.loop, grace_seconds=0.2)
    _restart_and_settle(tmp_path, world, {("once", due)})

    assert stopped
    assert Counter(world.effects) == {("once", due): 1}


def test_a_run_cut_short_before_its_effect_is_replayed_once(
    world: _World, tmp_path: Path
):
    # The dispatch stalls before its effect; shutdown cancels it.
    world.stalls["hourly"] = (30.0, 0.0)
    scheduler, _ = _scheduler(tmp_path)
    due = _at(0.05)
    scheduler.add_job(
        dispatch, "interval", id="hourly", hours=1, next_run_time=due, args=["hourly"]
    )
    assert _wait_for(lambda: bool(scheduler.ledger and scheduler.ledger._runs))

    stopped = stop_scheduler(scheduler, world.loop, grace_seconds=0.2)
    assert world.effects == []
    _restart_and_settle(tmp_path, world, {("hourly", due)})

    assert stopped
    assert Counter(world.effects) == {("hourly", due): 1}


def test_a_run_cut_short_after_its_effect_is_not_replayed(
    world: _World, tmp_path: Path
):
    # The effect happened; the dispatch is still waiting on its outcome.
    world.stalls["hourly"] = (0.0, 30.0)
    scheduler, _ = _scheduler(tmp_path)
    due = _at(0.05)
    scheduler.add_job(
        dispatch, "interval", id="hourly", hours=1, next_run_time=due, args=["hourly"]
    )
    assert _wait_for(lambda: bool(world.effects))

    stopped = stop_scheduler(scheduler, world.loop, grace_seconds=0.2)
    _restart_and_settle(tmp_path, world, {("hourly", due)})

    assert stopped
    assert Counter(world.effects) == {("hourly", due): 1}


# Chosen so that removing any one guard fails at least one seed: the in-lock
# stop (5, 20), the pass guard (0, 7, 11, 12), handing runs back (most seeds)
# and refusing abandoned commits (0, 6, 17, 38).
_SEEDS = [0, 1, 2, 3, 5, 6, 7, 11, 12, 17, 20, 38]


@pytest.mark.parametrize("seed", _SEEDS)
def test_shutdown_at_any_moment_runs_every_due_job_exactly_once(
    world: _World, tmp_path: Path, seed: int
):
    """Seeded interleavings of pass timing, a pass stalled on the jobstore,
    dispatch stalls before and after their effect, the shutdown moment, the
    grace period (and with it the drain timeout) and where cancellation lands.
    """
    rng = random.Random(seed)
    jobs = {f"job{i}": rng.choice(["once", "hourly"]) for i in range(rng.randint(3, 6))}
    world.stalls = {
        name: (
            rng.choice([0, rng.uniform(0, 0.8)]),
            rng.choice([0, rng.uniform(0, 0.5)]),
        )
        for name in jobs
    }
    pass_stall = (
        {rng.choice(list(jobs)): (rng.uniform(0, 1.0), rng.random() < 0.5)}
        if rng.random() < 0.8
        else {}
    )
    scheduler, _ = _scheduler(tmp_path, workers=rng.randint(1, 4), stall=pass_stall)
    expected: set[RunKey] = set()
    for name, kind in jobs.items():
        due = _at(rng.uniform(0, 0.6))
        if kind == "once":
            scheduler.add_job(dispatch, "date", id=name, run_date=due, args=[name])
        else:
            scheduler.add_job(
                dispatch, "interval", id=name, hours=1, next_run_time=due, args=[name]
            )
        expected.add((name, due))

    time.sleep(rng.uniform(0.05, 0.7))
    stopped = stop_scheduler(
        scheduler, world.loop, grace_seconds=rng.uniform(0.05, 0.6)
    )
    assert scheduler.state == STATE_STOPPED
    _restart_and_settle(tmp_path, world, expected)

    assert stopped
    assert Counter(world.effects) == {key: 1 for key in expected}


@pytest.fixture
def make_run():
    """Runs of a real job, as the ledger records them at submission."""
    jobs = BackgroundScheduler(timezone=timezone.utc)
    jobs.start(paused=True)

    def make(ledger: RunLedger):
        job = jobs.add_job(dispatch, "date", run_date=_at(60), args=["acting"])
        return ledger.open(job, [_at(0)])

    yield make
    jobs.shutdown(wait=False)


@pytest.mark.asyncio
async def test_an_early_exit_that_acts_counts_as_the_run_taking_effect(
    monkeypatch: pytest.MonkeyPatch, make_run
):
    """A turn that skips by unscheduling itself has acted: shutdown must not
    hand it back, or the next scheduler re-adds the schedule it deleted."""
    delete = AsyncMock()
    monkeypatch.setattr(
        scheduler_module, "get_scheduler_client", lambda: Mock(delete_schedule=delete)
    )
    ledger = RunLedger(commit_aware=lambda job: True)
    acted, refused = make_run(ledger), make_run(ledger)

    dispatching(acted)
    await scheduler_module._best_effort_unschedule("s1", None, "u", reason="gone")
    assert acted.state == "done"

    ledger.close(commit_wait=0)
    assert ledger.close(commit_wait=0) == [refused]
    dispatching(refused)
    with pytest.raises(RunAbandoned):
        await scheduler_module._best_effort_unschedule("s2", None, "u", reason="gone")
    assert [c.kwargs["schedule_id"] for c in delete.await_args_list] == ["s1"]


def test_a_run_that_cannot_be_handed_back_fails_the_stop_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_run
):
    ledger = RunLedger(commit_aware=lambda job: True)
    broken, fine = make_run(ledger), make_run(ledger)
    ledger.close(commit_wait=0)
    handed_back = []

    def give_back_run(scheduler, run):
        if run is broken:
            raise RuntimeError("unpicklable job")
        handed_back.append(run)

    monkeypatch.setattr(scheduler_runs, "_give_back_run", give_back_run)
    scheduler, _ = _scheduler(tmp_path)
    try:
        assert ledger.give_back(scheduler, lock_wait=1) is False
    finally:
        scheduler.shutdown()
    assert handed_back == [fine]
