"""Fixtures for scheduler index integration tests."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from sqlalchemy import create_engine

from backend.executor.schedule_index import ScheduleIndex
from backend.executor.scheduler import Scheduler


def _scheduler_with_index() -> Scheduler:
    scheduler = Scheduler(register_system_tasks=False)
    scheduler._schedule_index = ScheduleIndex(create_engine("sqlite://"))
    scheduler._schedule_index.ensure_table()
    scheduler._schedule_index_ready = True
    scheduler.scheduler = MagicMock()
    return scheduler


def _graph_job_kwargs(schedule_id: str, user_id: str, graph_id: str) -> dict:
    return {
        "kind": "graph",
        "schedule_id": schedule_id,
        "user_id": user_id,
        "graph_id": graph_id,
        "graph_version": 1,
        "cron": "0 0 * * *",
        "input_data": {},
        "input_credentials": {},
    }


def _mock_job(kwargs: dict) -> MagicMock:
    job = MagicMock()
    job.kwargs = kwargs
    job.id = kwargs.get("schedule_id", "fake-id")
    job.name = "fake-name"
    job.next_run_time = datetime(2026, 5, 22, 10, 0, tzinfo=timezone.utc)
    job.trigger = MagicMock()
    job.trigger.timezone = "UTC"
    return job
