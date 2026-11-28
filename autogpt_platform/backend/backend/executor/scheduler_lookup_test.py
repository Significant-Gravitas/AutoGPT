"""Reads that answer without loading every schedule: one by id, and each
graph's next run from the schedule index."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Column, Float, LargeBinary, MetaData, String, Table, insert

from backend.executor.schedule_index_test_helpers import (
    _graph_job_kwargs,
    _mock_job,
    _scheduler_with_index,
)
from backend.executor.scheduler import (
    GraphExecutionJobArgs,
    Jobstores,
    Scheduler,
    _index_entry,
)

# job id -> (owner, organization, team, expert)
_TENANCY = {
    "own": ("u1", "org-1", None, None),
    "org-home": ("u2", "org-1", None, None),
    "my-team": ("u2", "org-1", "t1", None),
    "other-team": ("u2", "org-1", "t2", None),
    "other-org": ("u2", "org-2", None, None),
    "their-expert": ("u2", "org-1", None, "e1"),
}


@pytest.fixture
def scheduler() -> Scheduler:
    scheduler = _scheduler_with_index()
    assert scheduler._schedule_index is not None
    jobs = {}
    for job_id, (user_id, organization_id, team_id, expert_id) in _TENANCY.items():
        args = GraphExecutionJobArgs(
            **_graph_job_kwargs(job_id, user_id, "g1")
            | {
                "organization_id": organization_id,
                "team_id": team_id,
                "expert_id": expert_id,
            }
        )
        jobs[job_id] = _mock_job(args.model_dump(mode="json"))
        scheduler._schedule_index.upsert(_index_entry(job_id, args))
    scheduler.scheduler.get_job.side_effect = lambda job_id, jobstore=None: jobs.get(
        job_id
    )
    scheduler.scheduler.get_jobs.return_value = list(jobs.values())
    return scheduler


def test_one_schedule_is_found_exactly_when_the_listing_shows_it(
    scheduler: Scheduler,
):
    scope = {"user_id": "u1", "organization_id": "org-1", "team_ids": ["t1"]}
    listed = {
        s.id
        for s in scheduler.get_graph_execution_schedules(**scope, include_paused=True)
    }
    found = {
        job_id
        for job_id in _TENANCY
        if scheduler.get_graph_execution_schedule(job_id, **scope)
    }

    assert listed == {"own", "org-home", "my-team"}
    assert found == listed


def test_without_an_organization_only_the_owners_schedule_is_found(
    scheduler: Scheduler,
):
    assert scheduler.get_graph_execution_schedule("own", user_id="u1")
    assert scheduler.get_graph_execution_schedule("org-home", user_id="u1") is None
    assert scheduler.get_graph_execution_schedule("missing", user_id="u1") is None


_SOON = datetime(2026, 12, 1, 9, tzinfo=UTC)
_LATER = datetime(2026, 12, 2, 9, tzinfo=UTC)


@pytest.fixture
def next_run_scheduler() -> Scheduler:
    """u1's graph schedules on g1 (two) and g2 (paused), u2's on g3, with
    the jobstore's next-run column beside the index and no job loadable."""
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    jobs = Table(
        "apscheduler_jobs",
        MetaData(),
        Column("id", String(191), primary_key=True),
        Column("next_run_time", Float(25), index=True),
        Column("job_state", LargeBinary, nullable=False),
    )
    jobs.metadata.create_all(index._engine)
    rows = [
        ("later", "u1", "g1", _LATER),
        ("soon", "u1", "g1", _SOON),
        ("paused", "u1", "g2", None),
        ("theirs", "u2", "g3", _SOON),
    ]
    with index._engine.begin() as conn:
        for job_id, user_id, graph_id, next_run in rows:
            index.upsert(
                _index_entry(
                    job_id,
                    GraphExecutionJobArgs(
                        **_graph_job_kwargs(job_id, user_id, graph_id)
                    ),
                )
            )
            conn.execute(
                insert(jobs).values(
                    id=job_id,
                    next_run_time=next_run.timestamp() if next_run else None,
                    job_state=b"",
                )
            )
    scheduler._persistent_jobstores = {
        Jobstores.EXECUTION.value: MagicMock(jobs_t=jobs)
    }
    scheduler.scheduler.get_job.side_effect = AssertionError("a job was loaded")
    scheduler.scheduler.get_jobs.side_effect = AssertionError("jobs were loaded")
    return scheduler


@pytest.mark.parametrize(
    "graph_id, expected",
    [(None, {"g1": _SOON.isoformat()}), ("g1", {"g1": _SOON.isoformat()}), ("g2", {})],
)
def test_each_graphs_next_run_is_read_from_the_index(
    next_run_scheduler: Scheduler, graph_id: str | None, expected: dict
):
    runs = next_run_scheduler.get_graph_schedule_next_runs("u1", graph_id=graph_id)

    assert runs == expected


def test_no_next_runs_are_answered_while_the_index_is_not_ready(
    next_run_scheduler: Scheduler,
):
    next_run_scheduler._schedule_index_ready = False

    assert next_run_scheduler.get_graph_schedule_next_runs("u1") is None
