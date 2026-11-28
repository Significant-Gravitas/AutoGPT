"""The table of system jobs a scheduler start registers, as a snapshot: under
default settings, and with each setting that switches a job on or off, or moves
its first run, flipped. The jobstore persists each job under its id and its
function's import path, so a renamed id or a moved function is a change to
stored rows, and shows here as one."""

import json
import pickle
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic_core import PydanticUndefined
from pytest_snapshot.plugin import Snapshot

from backend.executor import scheduler as scheduler_module
from backend.executor.scheduler import Jobstores, Scheduler
from backend.scheduler_service import SYSTEM_JOBS
from backend.util import system_jobs
from backend.util.settings import Config
from backend.util.system_jobs import run_on_scheduler_loop

FLIPPED_SETTINGS = (
    "auth_identity_orphan_sweep_enabled",
    "account_erasure_sweep_enabled",
    "scheduler_startup_embedding_backfill",
)
# A trigger built without a timezone takes the host's; pinned so the table reads
# the same on every machine, and named "<local>" in it.
_LOCAL_ZONE = ZoneInfo("Pacific/Kiritimati")


def test_the_system_job_table(monkeypatch, tmp_path, snapshot: Snapshot):
    _pin_settings_to_their_defaults(monkeypatch)
    default = _job_table(tmp_path / "default")
    lines = ["default settings:", *default.values()]
    for name in FLIPPED_SETTINGS:
        value = not getattr(scheduler_module.config, name)
        with monkeypatch.context() as flip:
            flip.setattr(scheduler_module.config, name, value)
            flipped = _job_table(tmp_path / name)
        lines.append(f"{name}={value}:")
        for job_id in sorted(default.keys() | flipped.keys()):
            if default.get(job_id) != flipped.get(job_id):
                lines += [f"- {default[job_id]}"] if job_id in default else []
                lines += [f"+ {flipped[job_id]}"] if job_id in flipped else []

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match("\n".join(lines) + "\n", "scheduler_system_jobs")


def test_a_scheduler_without_system_tasks_registers_none(tmp_path):
    assert _job_table(tmp_path, register_system_tasks=False) == {}


def test_a_moved_job_body_runs_on_the_loop_the_scheduler_lends_it(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(system_jobs, "_loop_runner", None)

    async def running_thread() -> str:
        return threading.current_thread().name

    with _started(tmp_path):
        assert run_on_scheduler_loop(running_thread(), timeout=5) == (
            "SchedulerEventLoop"
        )


# One kept interval job registered before start, one kept cron registered after.
@pytest.mark.parametrize(
    "job_id", ["dream_pass_reaper", "refresh_near_expiry_credentials"]
)
def test_a_start_replaces_a_stored_job_whose_function_has_moved(
    monkeypatch, tmp_path, job_id
):
    _pin_settings_to_their_defaults(monkeypatch)
    with _started(tmp_path) as service:
        declared = service.scheduler.get_job(job_id).func_ref
    _point_stored_job_at(tmp_path / "jobs.sqlite", job_id, "backend.executor:moved")

    with _started(tmp_path) as service:
        assert service.scheduler.get_job(job_id).func_ref == declared


def _point_stored_job_at(database, job_id: str, func_ref: str) -> None:
    """What a deploy that moved the job's function leaves in the jobstore."""
    with sqlite3.connect(database) as db:
        (state,) = db.execute(
            "SELECT job_state FROM apscheduler_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        job_state = pickle.loads(state)
        job_state["func"] = func_ref
        db.execute(
            "UPDATE apscheduler_jobs SET job_state = ? WHERE id = ?",
            (pickle.dumps(job_state), job_id),
        )


def _pin_settings_to_their_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, field in Config.model_fields.items():
        default = field.get_default(call_default_factory=True)
        if default is not PydanticUndefined:
            monkeypatch.setattr(scheduler_module.config, name, default)


def _job_table(directory, register_system_tasks: bool = True) -> dict[str, str]:
    """One JSON line per job a real start registered, by id."""
    started_at = datetime.now(timezone.utc)
    with _started(directory, register_system_tasks) as service:
        registered_by = datetime.now(timezone.utc)
        return {
            job.id: json.dumps(
                _row(job, alias.value, started_at, registered_by), sort_keys=True
            )
            for alias in Jobstores
            for job in sorted(
                service.scheduler.get_jobs(jobstore=alias.value), key=lambda j: j.id
            )
        }


@contextmanager
def _started(directory, register_system_tasks: bool = True) -> Iterator[Scheduler]:
    """A real start. APScheduler starts paused on a SQLite jobstore, so no job
    runs and no platform row is read."""
    directory.mkdir(parents=True, exist_ok=True)
    real_start = BackgroundScheduler.start

    def start_paused(self, *args, **kwargs):
        real_start(self, paused=True)

    service = Scheduler(
        register_system_tasks=register_system_tasks, system_jobs=SYSTEM_JOBS
    )
    with (
        patch.object(
            Scheduler,
            "_jobstore_database",
            return_value=(None, f"sqlite:///{directory}/jobs.sqlite"),
        ),
        patch.object(Scheduler, "_start_schedule_index_backfill"),
        patch.object(BackgroundScheduler, "start", start_paused),
        patch.object(scheduler_module, "load_dotenv"),
        patch("apscheduler.triggers.cron.get_localzone", return_value=_LOCAL_ZONE),
        patch("apscheduler.triggers.interval.get_localzone", return_value=_LOCAL_ZONE),
        # Blocks forever serving RPC; the start-up work before it is under test.
        patch("backend.util.service.AppService.run_service"),
    ):
        service.run_service()
    try:
        yield service
    finally:
        service.cleanup()


def _row(job: Job, jobstore: str, started_at: datetime, registered_by: datetime):
    zone = str(job.trigger.timezone)
    return {
        "id": job.id,
        "name": job.name,
        "func": job.func_ref,
        "args": list(job.args),
        "kwargs": job.kwargs,
        "trigger": f"{type(job.trigger).__name__} {job.trigger}",
        "timezone": "<local>" if zone == str(_LOCAL_ZONE) else zone,
        "jobstore": jobstore,
        "max_instances": job.max_instances,
        "coalesce": job.coalesce,
        "misfire_grace_time": job.misfire_grace_time,
        "first_run": _first_run(job, started_at, registered_by),
    }


def _first_run(job: Job, started_at: datetime, registered_by: datetime) -> str:
    """When the job first runs, relative to the start; a cron's is its schedule."""
    if job.next_run_time is None:
        return "paused"
    if not isinstance(job.trigger, IntervalTrigger):
        return "by schedule"
    if job.next_run_time >= started_at + job.trigger.interval - timedelta(seconds=1):
        return "one interval after start"
    if job.next_run_time <= registered_by:
        return "at start"
    return "other"
