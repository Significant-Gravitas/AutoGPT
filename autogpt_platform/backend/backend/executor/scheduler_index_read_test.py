"""Indexed scheduler reads and visibility predicates with a real SQLite index."""

from unittest.mock import MagicMock, patch

from backend.executor.schedule_index_test_helpers import (
    _graph_job_kwargs,
    _mock_job,
    _scheduler_with_index,
)
from backend.executor.scheduler import (
    CopilotTurnJobArgs,
    GraphExecutionJobArgs,
    Scheduler,
    _index_entry,
)


def test_index_entry_from_graph_args_normalizes_empty_org():
    args = GraphExecutionJobArgs(**_graph_job_kwargs("s1", "u1", "g1"))
    entry = _index_entry("job-id-not-schedule-id", args)
    # job_id comes from the jobstore key, not kwargs — legacy rows can have
    # schedule_id=None while still being addressable by job.id.
    assert entry.job_id == "job-id-not-schedule-id"
    assert entry.user_id == "u1"
    assert entry.kind == "graph"
    assert entry.graph_id == "g1"
    assert entry.session_id is None
    assert entry.organization_id is None  # "" default → NULL


def test_index_entry_from_copilot_args_carries_session():
    args = CopilotTurnJobArgs(
        schedule_id="s1",
        user_id="u1",
        session_id="sess-1",
        message="m",
        cron="0 0 * * *",
        organization_id="org-1",
    )
    entry = _index_entry("s1", args)
    assert entry.kind == "copilot_turn"
    assert entry.graph_id is None
    assert entry.session_id == "sess-1"
    assert entry.organization_id == "org-1"


def test_get_schedule_jobs_falls_back_without_index_or_before_backfill():
    scheduler = Scheduler(register_system_tasks=False)
    scheduler.scheduler = MagicMock()
    full_scan = [MagicMock()]
    with patch.object(scheduler, "_get_jobs_cached", return_value=full_scan) as scan:
        assert (
            scheduler._get_schedule_jobs(
                user_id="u1",
                graph_id=None,
                session_id=None,
                kind=None,
                organization_id=None,
            )
            == full_scan
        )
        scan.assert_called_once()

    scheduler = _scheduler_with_index()
    scheduler._schedule_index_ready = False
    with patch.object(scheduler, "_get_jobs_cached", return_value=full_scan) as scan:
        assert (
            scheduler._get_schedule_jobs(
                user_id="u1",
                graph_id=None,
                session_id=None,
                kind=None,
                organization_id=None,
            )
            == full_scan
        )
        scan.assert_called_once()


def test_get_schedule_jobs_unfiltered_read_uses_full_scan():
    scheduler = _scheduler_with_index()
    full_scan = [MagicMock()]
    with patch.object(scheduler, "_get_jobs_cached", return_value=full_scan) as scan:
        assert (
            scheduler._get_schedule_jobs(
                user_id=None,
                graph_id=None,
                session_id=None,
                kind="graph",  # kind alone is not an identity filter
                organization_id=None,
            )
            == full_scan
        )
        scan.assert_called_once()


def test_get_schedule_jobs_loads_candidates_and_drops_dangling_rows():
    scheduler = _scheduler_with_index()
    live_args = GraphExecutionJobArgs(**_graph_job_kwargs("live", "u1", "g1"))
    gone_args = GraphExecutionJobArgs(**_graph_job_kwargs("gone", "u1", "g1"))
    assert scheduler._schedule_index is not None
    scheduler._schedule_index.upsert_many(
        [_index_entry("live", live_args), _index_entry("gone", gone_args)]
    )

    live_job = _mock_job(live_args.model_dump(mode="json"))
    scheduler.scheduler.get_job.side_effect = lambda job_id, jobstore=None: (
        live_job if job_id == "live" else None
    )

    with patch.object(scheduler, "_get_jobs_cached") as scan:
        jobs = scheduler._get_schedule_jobs(
            user_id="u1",
            graph_id=None,
            session_id=None,
            kind=None,
            organization_id=None,
        )
    scan.assert_not_called()
    assert jobs == [live_job]
    # The dangling row (fired one-shot removed by APScheduler) is cleaned up.
    assert scheduler._schedule_index.all_job_ids() == {"live"}


def test_get_schedule_jobs_index_error_falls_back_to_full_scan():
    scheduler = _scheduler_with_index()
    full_scan = [MagicMock()]
    assert scheduler._schedule_index is not None
    with (
        patch.object(
            scheduler._schedule_index,
            "candidate_job_ids",
            side_effect=RuntimeError("db down"),
        ),
        patch.object(scheduler, "_get_jobs_cached", return_value=full_scan) as scan,
    ):
        assert (
            scheduler._get_schedule_jobs(
                user_id="u1",
                graph_id=None,
                session_id=None,
                kind=None,
                organization_id=None,
            )
            == full_scan
        )
        scan.assert_called_once()


def test_get_execution_schedules_via_index_applies_exact_predicate():
    """Index candidates still flow through the ownership predicate."""
    scheduler = _scheduler_with_index()
    mine = GraphExecutionJobArgs(**_graph_job_kwargs("mine", "u1", "g1"))
    assert scheduler._schedule_index is not None
    scheduler._schedule_index.upsert(_index_entry("mine", mine))

    job = _mock_job(mine.model_dump(mode="json"))
    scheduler.scheduler.get_job.return_value = job

    with patch.object(scheduler, "_get_jobs_cached") as scan:
        results = scheduler.get_execution_schedules(user_id="u1")
    scan.assert_not_called()
    assert [r.id for r in results] == ["mine"]

    # A stale/hostile index row for another user's schedule is trimmed by
    # the predicate even though the index nominated it.
    other = GraphExecutionJobArgs(**_graph_job_kwargs("other", "u2", "g2"))
    scheduler._schedule_index.upsert(
        _index_entry("other", other).model_copy(update={"user_id": "u1"})
    )
    scheduler.scheduler.get_job.side_effect = lambda job_id, jobstore=None: (
        job if job_id == "mine" else _mock_job(other.model_dump(mode="json"))
    )
    results = scheduler.get_execution_schedules(user_id="u1")
    assert [r.id for r in results] == ["mine"]


def test_persist_schedule_writes_index_row_and_delete_removes_it():
    scheduler = _scheduler_with_index()
    args = GraphExecutionJobArgs(**_graph_job_kwargs("s1", "u1", "g1"))
    job = _mock_job(args.model_dump(mode="json"))
    scheduler.scheduler.add_job.return_value = job

    scheduler._persist_schedule(
        dispatch_func=MagicMock(),
        job_args=args,
        trigger=MagicMock(),
        name="n",
    )
    assert scheduler._schedule_index is not None
    assert scheduler._schedule_index.all_job_ids() == {"s1"}

    with patch.object(
        scheduler, "_authorized_job", return_value=(job, MagicMock(kind="graph"))
    ):
        scheduler.delete_graph_execution_schedule("s1", "u1")
    assert scheduler._schedule_index.all_job_ids() == set()


def test_persist_schedule_survives_index_write_failure():
    scheduler = _scheduler_with_index()
    args = GraphExecutionJobArgs(**_graph_job_kwargs("s1", "u1", "g1"))
    scheduler.scheduler.add_job.return_value = _mock_job(args.model_dump(mode="json"))
    assert scheduler._schedule_index is not None
    with patch.object(
        scheduler._schedule_index, "upsert", side_effect=RuntimeError("db down")
    ):
        job = scheduler._persist_schedule(
            dispatch_func=MagicMock(),
            job_args=args,
            trigger=MagicMock(),
            name="n",
        )
    assert job is not None  # schedule creation must not fail on index errors


def test_reconcile_rebuilds_index_and_keeps_racing_rows():
    scheduler = _scheduler_with_index()
    index = scheduler._schedule_index
    assert index is not None

    in_store = GraphExecutionJobArgs(**_graph_job_kwargs("in-store", "u1", "g1"))
    store_jobs = [
        _mock_job(in_store.model_dump(mode="json")),
        _mock_job({"user_id": "u1"}),  # maintenance job: unparseable → skipped
    ]
    scheduler.scheduler.get_jobs.return_value = store_jobs

    # Row for a job that fired and was auto-removed (confirmed gone) …
    fired = GraphExecutionJobArgs(**_graph_job_kwargs("fired", "u1", "g1"))
    index.upsert(_index_entry("fired", fired))
    # … and a row added concurrently with the scan: not in the snapshot but
    # still present in the live store — it must survive the reconcile.
    racing = GraphExecutionJobArgs(**_graph_job_kwargs("racing", "u1", "g1"))
    index.upsert(_index_entry("racing", racing))
    racing_job = _mock_job(racing.model_dump(mode="json"))
    scheduler.scheduler.get_job.side_effect = lambda job_id, jobstore=None: (
        racing_job if job_id == "racing" else None
    )

    scheduler._schedule_index_ready = False
    scheduler._reconcile_schedule_index()

    assert index.all_job_ids() == {"in-store", "racing"}
    assert scheduler._schedule_index_ready is True
