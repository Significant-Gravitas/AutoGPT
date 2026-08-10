from datetime import date, datetime, timedelta, timezone

from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.execution_cost_summary import UserDailyCost, UserExecutionCostSummary

from .activity import compose_briefing, compose_week_summary

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)


def _execution(
    *,
    exec_id: str,
    status: ExecutionStatus,
    ended_at: datetime,
    activity_status: str | None = None,
) -> GraphExecutionMeta:
    return GraphExecutionMeta(
        id=exec_id,
        user_id="user",
        graph_id="graph",
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=status,
        started_at=ended_at - timedelta(minutes=1),
        ended_at=ended_at,
        stats=GraphExecutionMeta.Stats(activity_status=activity_status),
    )


def test_briefing_ignores_runs_outside_the_24h_window() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="fresh",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=2),
            ),
            _execution(
                exec_id="stale",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=30),
            ),
        ],
        expert_by_id={},
        agent_by_graph={"graph": ("Inbox triage", None)},
    )

    assert briefing.window_started_at == NOW - timedelta(hours=24)
    assert briefing.completed_count == 1
    assert [outcome.id for outcome in briefing.outcomes] == ["fresh"]


def test_briefing_lists_failures_before_successes() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="ok",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=1),
                activity_status="Sorted 12 emails. Nothing needed a reply.",
            ),
            _execution(
                exec_id="broken",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=3),
            ),
        ],
        expert_by_id={},
        agent_by_graph={"graph": ("Inbox triage", None)},
    )

    assert [outcome.status for outcome in briefing.outcomes] == ["failed", "completed"]
    assert briefing.failed_count == 1
    assert briefing.outcomes[0].title == "Inbox triage needs a retry"
    assert briefing.outcomes[1].title == "Sorted 12 emails."
    assert briefing.outcomes[1].summary == "Nothing needed a reply."


def test_briefing_counts_unlisted_successes_as_routine() -> None:
    executions = [
        _execution(
            exec_id=f"run-{index}",
            status=ExecutionStatus.COMPLETED,
            ended_at=NOW - timedelta(hours=1),
        )
        for index in range(6)
    ]

    briefing = compose_briefing(
        now=NOW,
        executions=executions,
        expert_by_id={},
        agent_by_graph={"graph": ("Inbox triage", None)},
    )

    assert briefing.completed_count == 6
    assert len(briefing.outcomes) == 4
    assert briefing.routine_count == 2


def test_week_summary_maps_status_counts_and_credits() -> None:
    summary = compose_week_summary(
        UserExecutionCostSummary(
            total_cents=420,
            run_count=10,
            billable_run_count=8,
            failed_cost_cents=15,
            success_run_count=7,
            failed_run_count=2,
            review_run_count=1,
            total_duration_seconds=93.5,
            duration_run_count=9,
            by_agent=[],
            top_runs=[],
            daily=[
                UserDailyCost(
                    date=date(2026, 8, 9),
                    cost_cents=120,
                    run_count=3,
                    success_count=2,
                    failed_count=1,
                    review_count=0,
                )
            ],
        ),
        250,
    )

    assert summary.run_count == 10
    assert summary.completed_count == 7
    assert summary.failed_count == 2
    assert summary.review_count == 1
    assert summary.total_runtime_seconds == 93.5
    assert summary.timed_run_count == 9
    assert summary.total_cost_cents == 420
    assert summary.credits_balance == 250
    assert summary.daily[0].completed_count == 2
    assert summary.daily[0].failed_count == 1
