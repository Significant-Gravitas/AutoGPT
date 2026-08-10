from datetime import date, datetime, timedelta, timezone

from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.execution_cost_summary import UserDailyCost, UserExecutionCostSummary
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .activity import (
    compose_active_tasks,
    compose_briefing,
    compose_upcoming_tasks,
    compose_week_summary,
)
from .helpers import AgentRef

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)
TRIAGE = {"graph": AgentRef("Inbox triage", "library-agent")}


def _execution(
    *,
    exec_id: str,
    status: ExecutionStatus,
    ended_at: datetime,
    activity_status: str | None = None,
    error: str | None = None,
    expert_id: str | None = None,
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
        expert_id=expert_id,
        stats=GraphExecutionMeta.Stats(activity_status=activity_status, error=error),
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
        agent_by_graph={"graph": AgentRef("Inbox triage", None)},
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
        agent_by_graph={"graph": AgentRef("Inbox triage", None)},
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
        agent_by_graph={"graph": AgentRef("Inbox triage", None)},
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


def test_briefing_keeps_a_raw_error_out_of_the_headline() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="broken",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=1),
                error="KeyError: 'recipient'",
            )
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
    )

    assert briefing.outcomes[0].title == "Inbox triage needs a retry"
    assert briefing.outcomes[0].summary == "KeyError: 'recipient'"


def test_briefing_orders_each_status_group_most_recent_first() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="old-failure",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=8),
            ),
            _execution(
                exec_id="new-failure",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=1),
            ),
            _execution(
                exec_id="old-success",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=9),
            ),
            _execution(
                exec_id="new-success",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=2),
            ),
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
    )

    assert [outcome.id for outcome in briefing.outcomes] == [
        "new-failure",
        "old-failure",
        "new-success",
        "old-success",
    ]


def test_active_tasks_map_status_and_cap_the_list() -> None:
    executions = [
        _execution(
            exec_id=f"queued-{index}",
            status=ExecutionStatus.QUEUED,
            ended_at=NOW,
        )
        for index in range(3)
    ]
    executions.insert(
        0,
        _execution(exec_id="running", status=ExecutionStatus.RUNNING, ended_at=NOW),
    )
    executions.append(
        _execution(exec_id="done", status=ExecutionStatus.COMPLETED, ended_at=NOW)
    )

    tasks = compose_active_tasks(executions, {}, TRIAGE)

    assert [task.id for task in tasks] == ["running", "queued-0", "queued-1"]
    assert [task.status for task in tasks] == ["running", "queued", "queued"]
    assert tasks[0].title == "Inbox triage"
    assert tasks[0].expert is None
    assert tasks[0].link == (
        "/library/agents/library-agent?activeTab=runs&activeItem=running"
    )


def test_active_task_falls_back_when_the_graph_is_unknown() -> None:
    tasks = compose_active_tasks(
        [_execution(exec_id="run", status=ExecutionStatus.RUNNING, ended_at=NOW)],
        {},
        {},
    )

    assert tasks[0].title == "Agent task"
    assert tasks[0].link is None


def _graph_job(job_id: str, next_run: str) -> GraphExecutionJobInfo:
    return GraphExecutionJobInfo(
        id=job_id,
        schedule_id=job_id,
        name=job_id,
        agent_name="Inbox triage",
        next_run_time=next_run,
        user_id="user",
        graph_id="graph",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
    )


def _copilot_job(job_id: str, next_run: str) -> CopilotTurnJobInfo:
    return CopilotTurnJobInfo(
        id=job_id,
        schedule_id=job_id,
        name="Follow up on the invoice",
        next_run_time=next_run,
        user_id="user",
        cron="0 9 * * *",
        message="ping",
    )


def test_upcoming_tasks_sort_by_next_run_and_cap_the_list() -> None:
    jobs = [
        _graph_job("late", "2026-08-10T18:00:00Z"),
        _copilot_job("followup", "2026-08-10T12:00:00Z"),
        _graph_job("early", "2026-08-10T09:00:00Z"),
        _graph_job("mid", "2026-08-10T11:00:00Z"),
        _graph_job("latest", "2026-08-10T23:00:00Z"),
    ]

    tasks = compose_upcoming_tasks(jobs, {})

    assert [task.id for task in tasks] == ["early", "mid", "followup", "late"]
    assert [task.kind for task in tasks] == ["agent", "agent", "followup", "agent"]
    assert tasks[0].title == "Inbox triage"
    assert tasks[2].title == "Follow up on the invoice"


def test_upcoming_tasks_skip_unparseable_next_run_times() -> None:
    tasks = compose_upcoming_tasks(
        [
            _graph_job("broken", "not-a-timestamp"),
            _graph_job("good", "2026-08-10T09:00:00Z"),
        ],
        {},
    )

    assert [task.id for task in tasks] == ["good"]
