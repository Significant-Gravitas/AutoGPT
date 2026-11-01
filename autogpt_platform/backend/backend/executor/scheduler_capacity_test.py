"""The per-user limit on active schedules, read through the schedule index."""

from unittest.mock import MagicMock

import pytest

from backend.executor import scheduler as scheduler_module
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


@pytest.fixture
def full_scheduler(monkeypatch: pytest.MonkeyPatch) -> Scheduler:
    """u1 holds the limit of two active schedules; ``held-*`` are its jobs."""
    monkeypatch.setattr(scheduler_module.config, "max_schedules_per_user", 2)
    scheduler = _scheduler_with_index()
    # As run_service leaves it, so the count goes through the index.
    scheduler._persistent_jobstores = {Jobstores.EXECUTION.value: MagicMock()}
    held = {
        job_id: _mock_job(_args(job_id, "u1").model_dump(mode="json"))
        for job_id in ("held-1", "held-2")
    }
    assert scheduler._schedule_index is not None
    scheduler._schedule_index.upsert_many(
        [_index_entry(job_id, _args(job_id, "u1")) for job_id in held]
    )
    scheduler.scheduler.get_job.side_effect = lambda job_id, jobstore=None: held.get(
        job_id
    )
    scheduler.scheduler.add_job.side_effect = lambda *_, id, **__: _mock_job(
        _graph_job_kwargs(id, "u1", "g1")
    )
    return scheduler


def test_a_user_at_the_limit_cannot_add_another_schedule(full_scheduler: Scheduler):
    with pytest.raises(ValueError, match="active schedules"):
        _persist(full_scheduler, _args("new", "u1"))
    full_scheduler.scheduler.add_job.assert_not_called()


def test_replacing_a_schedule_under_its_own_id_is_allowed_at_the_limit(
    full_scheduler: Scheduler,
):
    _persist(full_scheduler, _args("held-1", "u1"))
    full_scheduler.scheduler.add_job.assert_called_once()


def test_another_users_schedules_do_not_count_against_the_limit(
    full_scheduler: Scheduler,
):
    _persist(full_scheduler, _args("theirs", "u2"))
    full_scheduler.scheduler.add_job.assert_called_once()


def _args(schedule_id: str, user_id: str) -> GraphExecutionJobArgs:
    return GraphExecutionJobArgs(**_graph_job_kwargs(schedule_id, user_id, "g1"))


def _persist(scheduler: Scheduler, args: GraphExecutionJobArgs) -> None:
    scheduler._persist_schedule(
        dispatch_func=MagicMock(), job_args=args, trigger=MagicMock(), name="n"
    )
