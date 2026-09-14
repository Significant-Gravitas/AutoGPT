"""Recovery from index write and authoritative job lookup failures."""

from unittest.mock import MagicMock, patch

import pytest

from backend.executor.schedule_index_test_helpers import (
    _graph_job_kwargs,
    _mock_job,
    _scheduler_with_index,
)
from backend.executor.scheduler import GraphExecutionJobArgs, _index_entry


def test_failed_upsert_keeps_persisted_schedule_visible_until_reconcile():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("s1", "u1", "g1"))
    job = _mock_job(args.model_dump(mode="json"))
    scheduler.scheduler.get_jobs.return_value = []
    assert scheduler._get_jobs_cached() == []
    scheduler.scheduler.add_job.return_value = job
    scheduler.scheduler.get_jobs.return_value = [job]
    scheduler.scheduler.get_job.return_value = job

    with patch.object(index, "upsert", side_effect=RuntimeError("transient write")):
        assert (
            scheduler._persist_schedule(
                dispatch_func=MagicMock(), job_args=args, trigger=MagicMock(), name="n"
            )
            == job
        )

    assert index.all_job_ids() == set()
    assert scheduler._schedule_index_ready is False
    assert [s.id for s in scheduler.get_execution_schedules(user_id="u1")] == ["s1"]
    assert scheduler.get_execution_schedules(user_id="other") == []

    scheduler._reconcile_schedule_index()
    assert scheduler._schedule_index_ready is True
    assert index.all_job_ids() == {"s1"}
    with patch.object(scheduler, "_get_jobs_cached") as full_scan:
        assert [s.id for s in scheduler.get_execution_schedules(user_id="u1")] == ["s1"]
    full_scan.assert_not_called()


def test_transient_point_lookup_preserves_live_row_and_listing():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("s1", "u1", "g1"))
    job = _mock_job(args.model_dump(mode="json"))
    index.upsert(_index_entry("s1", args))
    scheduler.scheduler.get_job.side_effect = [RuntimeError("database timeout"), job]
    scheduler.scheduler.get_jobs.return_value = [job]

    assert [s.id for s in scheduler.get_execution_schedules(user_id="u1")] == ["s1"]
    assert index.all_job_ids() == {"s1"}
    with patch.object(scheduler, "_get_jobs_cached") as full_scan:
        assert [s.id for s in scheduler.get_execution_schedules(user_id="u1")] == ["s1"]
    full_scan.assert_not_called()


def test_reconcile_does_not_trust_snapshot_before_failed_upsert():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    old_args = GraphExecutionJobArgs(**_graph_job_kwargs("old", "u1", "g1"))
    new_args = GraphExecutionJobArgs(**_graph_job_kwargs("new", "u1", "g1"))
    old_job = _mock_job(old_args.model_dump(mode="json"))
    new_job = _mock_job(new_args.model_dump(mode="json"))
    scheduler.scheduler.add_job.return_value = new_job
    scheduler.scheduler.get_job.return_value = old_job

    def scan_before_concurrent_write(*args, **kwargs):
        with patch.object(index, "upsert", side_effect=RuntimeError("write failed")):
            scheduler._persist_schedule(
                dispatch_func=MagicMock(),
                job_args=new_args,
                trigger=MagicMock(),
                name="new",
            )
        return [old_job]

    scheduler.scheduler.get_jobs.side_effect = scan_before_concurrent_write
    scheduler._reconcile_schedule_index()
    assert index.all_job_ids() == {"old"}
    assert scheduler._schedule_index_ready is False

    scheduler.scheduler.get_jobs.side_effect = None
    scheduler.scheduler.get_jobs.return_value = [old_job, new_job]
    assert {s.id for s in scheduler.get_execution_schedules(user_id="u1")} == {
        "old",
        "new",
    }
    scheduler._reconcile_schedule_index()
    assert scheduler._schedule_index_ready is True
    assert index.all_job_ids() == {"old", "new"}


def test_empty_scan_reconcile_preserves_existing_index_rows():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("s1", "u1", "g1"))
    index.upsert(_index_entry("s1", args))
    scheduler._schedule_index_ready = False
    scheduler.scheduler.get_jobs.return_value = []
    scheduler.scheduler.get_job.return_value = None

    scheduler._reconcile_schedule_index()

    assert index.all_job_ids() == {"s1"}
    assert scheduler._schedule_index_ready is True
    scheduler.scheduler.get_job.assert_not_called()


def test_system_only_scan_drops_definitively_missing_index_rows():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("gone", "u1", "g1"))
    index.upsert(_index_entry("gone", args))
    scheduler.scheduler.get_jobs.return_value = [_mock_job({"user_id": "u1"})]
    scheduler.scheduler.get_job.return_value = None

    scheduler._reconcile_schedule_index()

    assert index.all_job_ids() == set()
    assert scheduler._schedule_index_ready is True


def test_failed_write_during_candidate_lookup_falls_back():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("s1", "u1", "g1"))
    job = _mock_job(args.model_dump(mode="json"))
    scheduler.scheduler.add_job.return_value = job
    scheduler.scheduler.get_jobs.return_value = [job]

    def query_before_failed_write(**kwargs):
        with patch.object(index, "upsert", side_effect=RuntimeError("write failed")):
            scheduler._persist_schedule(
                dispatch_func=MagicMock(), job_args=args, trigger=MagicMock(), name="n"
            )
        return []

    with patch.object(
        index, "candidate_job_ids", side_effect=query_before_failed_write
    ):
        assert [s.id for s in scheduler.get_execution_schedules(user_id="u1")] == ["s1"]


@pytest.mark.parametrize("has_existing_row", [True, False])
def test_empty_scan_does_not_reenable_index_after_known_failed_write(has_existing_row):
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    old_args = GraphExecutionJobArgs(**_graph_job_kwargs("old", "u1", "g1"))
    new_args = GraphExecutionJobArgs(**_graph_job_kwargs("new", "u1", "g1"))
    if has_existing_row:
        index.upsert(_index_entry("old", old_args))
    with patch.object(index, "upsert", side_effect=RuntimeError("write failed")):
        scheduler._upsert_schedule_index_row("new", new_args)
    scheduler.scheduler.get_jobs.return_value = []

    scheduler._reconcile_schedule_index()

    assert scheduler._schedule_index_ready is False
    assert index.all_job_ids() == ({"old"} if has_existing_row else set())


def test_corrupt_job_falls_back_then_cleans_up_after_definitive_absence():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("bad", "u1", "g1"))
    index.upsert(_index_entry("bad", args))
    scheduler.scheduler.get_job.side_effect = [LookupError("renamed callable"), None]
    scheduler.scheduler.get_jobs.return_value = []

    assert scheduler.get_execution_schedules(user_id="u1") == []
    assert index.all_job_ids() == {"bad"}
    assert scheduler.get_execution_schedules(user_id="u1") == []
    assert index.all_job_ids() == set()


def test_shutdown_during_candidate_lookup_keeps_shared_index_rows():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("live", "u1", "g1"))
    index.upsert(_index_entry("live", args))

    def stopping_scheduler(*args, **kwargs):
        scheduler.scheduler.running = False
        return None

    scheduler.scheduler.get_job.side_effect = stopping_scheduler
    scheduler.get_execution_schedules(user_id="u1")
    assert index.all_job_ids() == {"live"}


def test_shutdown_during_reconcile_verification_keeps_shared_index_rows():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None
    args = GraphExecutionJobArgs(**_graph_job_kwargs("live", "u1", "g1"))
    index.upsert(_index_entry("live", args))
    scheduler.scheduler.get_jobs.return_value = [_mock_job({"user_id": "u1"})]

    def stopping_scheduler(*args, **kwargs):
        scheduler.scheduler.running = False
        return None

    scheduler.scheduler.get_job.side_effect = stopping_scheduler
    scheduler._reconcile_schedule_index()
    assert index.all_job_ids() == {"live"}
