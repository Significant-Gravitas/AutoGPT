"""Scheduler shutdown: no job lost, none run twice, and the process exits.

Real APScheduler, real thread pools and a persistent SQLAlchemy jobstore (on
SQLite here), with the race windows opened deterministically. Each test fails
against stock ``BackgroundScheduler.shutdown(wait=True)``.
"""

import asyncio
import logging
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import uvicorn
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.base import STATE_STOPPED
from fastapi import FastAPI

from backend.executor import scheduler_shutdown
from backend.executor.scheduler import Scheduler
from backend.executor.scheduler_shutdown import (
    DrainingBackgroundScheduler,
    stop_scheduler,
)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


class _Jobs:
    def __init__(self) -> None:
        self.once_runs = 0
        self.tick_runs = 0
        self.long_started = threading.Event()
        self.release_long = threading.Event()


# Persistent jobs are pickled by reference, so their functions and the state
# they touch live at module level.
_jobs = _Jobs()


def once() -> None:
    _jobs.once_runs += 1


def tick() -> None:
    _jobs.tick_runs += 1


def long_job() -> None:
    _jobs.long_started.set()
    _jobs.release_long.wait(timeout=30)


class _GatedExecutor(ThreadPoolExecutor):
    """Holds the submission of job ``once`` until the test lets it through, so
    shutdown can be started while a job pass is mid-way."""

    def __init__(self) -> None:
        super().__init__(max_workers=4)
        self.in_submit = threading.Event()
        self.release = threading.Event()

    def _do_submit_job(self, job, run_times):
        if job.id == "once":
            self.in_submit.set()
            self.release.wait(timeout=10)
        return super()._do_submit_job(job, run_times)


@pytest.fixture
def jobs():
    global _jobs
    _jobs = _Jobs()
    yield _jobs
    _jobs.release_long.set()


def _store(tmp_path: Path) -> SQLAlchemyJobStore:
    return SQLAlchemyJobStore(url=f"sqlite:///{tmp_path / 'jobs.sqlite'}")


def _start(tmp_path: Path, executor: ThreadPoolExecutor) -> DrainingBackgroundScheduler:
    scheduler = DrainingBackgroundScheduler(
        executors={"default": executor},
        jobstores={"default": _store(tmp_path)},
        # The production defaults.
        job_defaults={"coalesce": True, "misfire_grace_time": None},
        timezone=timezone.utc,
    )
    scheduler.start()
    return scheduler


def _stored_job(tmp_path: Path, job_id: str):
    """What the next scheduler to start would find in the jobstore."""
    store = _store(tmp_path)
    store.start(DrainingBackgroundScheduler(), "default")
    try:
        return store.lookup_job(job_id)
    finally:
        store.shutdown()


def _wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_a_one_shot_job_submitted_as_shutdown_starts_runs_exactly_once(
    jobs: _Jobs, tmp_path: Path
):
    gate = _GatedExecutor()
    scheduler = _start(tmp_path, gate)
    scheduler.add_job(once, "date", id="once", run_date=datetime.now(timezone.utc))
    assert gate.in_submit.wait(timeout=5)

    # SIGTERM arrives while the pass that submitted ``once`` is still going.
    shutdown = threading.Thread(
        target=stop_scheduler, args=(scheduler, None, 5.0), daemon=True
    )
    shutdown.start()
    # Stock shutdown flips the state at once, under the pass's feet; the
    # draining scheduler waits for the pass to finish first.
    flipped_mid_pass = _wait_for(lambda: scheduler.state == STATE_STOPPED, 1.0)
    gate.release.set()
    shutdown.join(timeout=10)

    assert not flipped_mid_pass
    assert jobs.once_runs == 1
    # Removed as it ran, so the next scheduler doesn't run it again.
    assert _stored_job(tmp_path, "once") is None


def test_a_pass_queued_behind_shutdown_submits_nothing_and_loses_nothing(
    jobs: _Jobs, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    scheduler = _start(tmp_path, ThreadPoolExecutor(max_workers=4))
    scheduler.add_job(long_job, "date", id="long", run_date=datetime.now(timezone.utc))
    assert jobs.long_started.wait(timeout=5)
    due_at = datetime.now(timezone.utc) + timedelta(seconds=0.5)
    scheduler.add_job(tick, "interval", id="tick", hours=1, next_run_time=due_at)

    caplog.set_level(logging.ERROR, logger="apscheduler")
    shutdown = threading.Thread(
        target=stop_scheduler, args=(scheduler, None, 10.0), daemon=True
    )
    shutdown.start()
    # Shutdown has begun and is waiting for ``long_job``.
    assert _wait_for(lambda: scheduler.state == STATE_STOPPED)
    # ``tick`` falls due; a pass wakes for it and queues on the lock.
    time.sleep(max(0.0, (due_at - datetime.now(timezone.utc)).total_seconds()))
    queued_pass = threading.Thread(target=scheduler._process_jobs, daemon=True)
    queued_pass.start()

    jobs.release_long.set()
    shutdown.join(timeout=10)
    queued_pass.join(timeout=5)

    assert not [r for r in caplog.records if "Error submitting job" in r.getMessage()]
    assert jobs.tick_runs == 0
    # Still due, so the next scheduler runs it rather than skipping past it.
    stored = _stored_job(tmp_path, "tick")
    assert stored is not None and stored.next_run_time == due_at.replace(
        microsecond=stored.next_run_time.microsecond
    )
    assert stored.next_run_time <= due_at + timedelta(seconds=1)


def test_a_pass_stuck_on_the_jobstore_cannot_hold_shutdown(tmp_path: Path):
    scheduler = _start(tmp_path, ThreadPoolExecutor(max_workers=2))
    stuck = threading.Event()
    release = threading.Event()

    def stuck_pass() -> None:
        # A pass blocked on a slow database, holding the jobstore lock.
        with scheduler._jobstores_lock:
            stuck.set()
            release.wait(timeout=30)

    threading.Thread(target=stuck_pass, daemon=True).start()
    assert stuck.wait(timeout=5)
    result: list[bool] = []
    stopping = threading.Thread(
        target=lambda: result.append(stop_scheduler(scheduler, None, 0.5)),
        daemon=True,
    )
    started = time.monotonic()
    stopping.start()
    stopping.join(timeout=0.5 + scheduler_shutdown._CANCEL_GRACE_S + 5)
    release.set()

    # It gives up on the pass and reports the stop incomplete, so the caller
    # exits instead of waiting until Kubernetes kills the pod.
    assert result == [False]
    assert time.monotonic() - started < 0.5 + scheduler_shutdown._CANCEL_GRACE_S + 3


_EXIT_PROBE = textwrap.dedent(
    """
    import asyncio, sys, threading, time
    from datetime import datetime, timezone
    from apscheduler.executors.pool import ThreadPoolExecutor
    from apscheduler.jobstores.memory import MemoryJobStore
    from backend.executor.scheduler_shutdown import (
        DrainingBackgroundScheduler, exit_if_jobs_hung, stop_scheduler,
    )

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    started = threading.Event()

    def waits_on_the_event_loop():
        # How our jobs run: a worker blocks on a coroutine on the scheduler's loop.
        started.set()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(300), loop).result(300)

    def stuck_in_sync_code():
        started.set()
        time.sleep(300)

    job = {"loop": waits_on_the_event_loop, "stuck": stuck_in_sync_code}[sys.argv[1]]
    scheduler = DrainingBackgroundScheduler(
        executors={"default": ThreadPoolExecutor(2)},
        jobstores={"default": MemoryJobStore()},
        timezone=timezone.utc,
    )
    scheduler.start()
    scheduler.add_job(job, next_run_time=datetime.now(timezone.utc))
    started.wait(10)
    stopped = stop_scheduler(scheduler, loop, grace_seconds=0.5)
    print("stopped", stopped, flush=True)
    exit_if_jobs_hung(stopped)
    """
)


@pytest.mark.parametrize(
    ("job", "stopped", "exit_code"), [("loop", True, 0), ("stuck", False, 1)]
)
def test_the_process_exits_soon_after_the_grace_period(
    tmp_path: Path, job: str, stopped: bool, exit_code: int
):
    probe = tmp_path / "probe.py"
    probe.write_text(_EXIT_PROBE)
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(probe), job],
        cwd=_BACKEND_ROOT,
        env={**os.environ, "PYTHONPATH": str(_BACKEND_ROOT)},
        capture_output=True,
        text=True,
        # Before the fix the process outlived its job's 300s wait.
        timeout=60,
    )
    elapsed = time.monotonic() - started

    assert f"stopped {stopped}" in result.stdout, result.stderr
    assert result.returncode == exit_code
    # Grace, then at most the cancel grace, plus interpreter start-up.
    assert elapsed < 0.5 + scheduler_shutdown._CANCEL_GRACE_S + 15


@pytest.mark.asyncio
async def test_the_rpc_server_stops_waiting_for_a_hung_call(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "backend.executor.scheduler.config.scheduler_rpc_drain_seconds", 1
    )
    app = FastAPI()
    entered = asyncio.Event()

    @app.get("/slow")
    async def slow() -> None:
        entered.set()
        await asyncio.sleep(60)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            timeout_graceful_shutdown=Scheduler.rpc_graceful_shutdown_seconds(),
        )
    )
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    async with httpx.AsyncClient() as client:
        call = asyncio.create_task(client.get(f"http://127.0.0.1:{port}/slow"))
        await entered.wait()

        started = time.monotonic()
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=10)
        call.cancel()

    assert time.monotonic() - started < 3
