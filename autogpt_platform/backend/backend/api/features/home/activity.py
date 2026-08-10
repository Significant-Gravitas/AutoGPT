from datetime import datetime, timedelta

from backend.api.features.experts.models import Expert
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.execution_cost_summary import UserExecutionCostSummary
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .helpers import as_utc, parse_datetime, run_link, split_summary, to_home_expert
from .models import (
    HomeActiveTask,
    HomeBriefing,
    HomeBriefingOutcome,
    HomeDailyActivity,
    HomeUpcomingTask,
    HomeWeekSummary,
)

_MAX_OUTCOMES = 4
_MAX_ACTIVE = 3
_MAX_UPCOMING = 4


def compose_briefing(
    *,
    now: datetime,
    executions: list[GraphExecutionMeta],
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, tuple[str, str | None]],
) -> HomeBriefing:
    window_start = now - timedelta(hours=24)
    terminal = [
        execution
        for execution in executions
        if execution.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
        and as_utc(execution.ended_at or execution.started_at or now) >= window_start
    ]
    terminal.sort(key=lambda execution: execution.status == ExecutionStatus.COMPLETED)
    outcomes = [
        _briefing_outcome(execution, expert_by_id, agent_by_graph)
        for execution in terminal[:_MAX_OUTCOMES]
    ]
    completed = sum(
        execution.status == ExecutionStatus.COMPLETED for execution in terminal
    )
    failed = sum(execution.status == ExecutionStatus.FAILED for execution in terminal)
    shown_completed = sum(outcome.status == "completed" for outcome in outcomes)
    return HomeBriefing(
        generated_at=now,
        window_started_at=window_start,
        completed_count=completed,
        failed_count=failed,
        routine_count=max(0, completed - shown_completed),
        outcomes=outcomes,
    )


def compose_active_tasks(
    executions: list[GraphExecutionMeta],
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, tuple[str, str | None]],
) -> list[HomeActiveTask]:
    active = [
        execution
        for execution in executions
        if execution.status in {ExecutionStatus.RUNNING, ExecutionStatus.QUEUED}
    ]
    return [
        HomeActiveTask(
            id=execution.id,
            title=agent_by_graph.get(execution.graph_id, ("Agent task", None))[0],
            status=(
                "running" if execution.status == ExecutionStatus.RUNNING else "queued"
            ),
            expert=(
                to_home_expert(expert_by_id[execution.expert_id])
                if execution.expert_id in expert_by_id
                else None
            ),
            started_at=execution.started_at,
            link=run_link(
                agent_by_graph.get(execution.graph_id, ("", None))[1], execution.id
            ),
        )
        for execution in active[:_MAX_ACTIVE]
    ]


def compose_upcoming_tasks(
    schedules: list[GraphExecutionJobInfo | CopilotTurnJobInfo],
    expert_by_graph: dict[str, Expert],
) -> list[HomeUpcomingTask]:
    upcoming = []
    for schedule in schedules:
        next_run = parse_datetime(schedule.next_run_time)
        if next_run is None:
            continue
        if isinstance(schedule, GraphExecutionJobInfo):
            expert = expert_by_graph.get(schedule.graph_id)
            upcoming.append(
                HomeUpcomingTask(
                    id=schedule.id,
                    title=schedule.agent_name or schedule.name,
                    kind="agent",
                    expert=to_home_expert(expert) if expert else None,
                    next_run_time=next_run,
                )
            )
        else:
            upcoming.append(
                HomeUpcomingTask(
                    id=schedule.id,
                    title=schedule.name,
                    kind="followup",
                    next_run_time=next_run,
                )
            )
    upcoming.sort(key=lambda task: task.next_run_time)
    return upcoming[:_MAX_UPCOMING]


def compose_week_summary(
    summary: UserExecutionCostSummary, credits_balance: int | None
) -> HomeWeekSummary:
    return HomeWeekSummary(
        run_count=summary.run_count,
        completed_count=summary.success_run_count,
        review_count=summary.review_run_count,
        failed_count=summary.failed_run_count,
        total_runtime_seconds=summary.total_duration_seconds,
        timed_run_count=summary.duration_run_count,
        total_cost_cents=summary.total_cents,
        credits_balance=credits_balance,
        daily=[
            HomeDailyActivity(
                date=day.date,
                completed_count=day.success_count,
                review_count=day.review_count,
                failed_count=day.failed_count,
            )
            for day in summary.daily
        ],
    )


def _briefing_outcome(
    execution: GraphExecutionMeta,
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, tuple[str, str | None]],
) -> HomeBriefingOutcome:
    name, library_id = agent_by_graph.get(execution.graph_id, ("Agent task", None))
    raw_summary = execution.stats.activity_status if execution.stats else None
    error = execution.stats.error if execution.stats else None
    failed = execution.status == ExecutionStatus.FAILED
    title, detail = split_summary(
        raw_summary or error,
        fallback_title=f"{name} {'needs a retry' if failed else 'finished'}",
        fallback_detail=(
            "Open the run to inspect the failure and choose the next step."
            if failed
            else "Completed successfully."
        ),
    )
    return HomeBriefingOutcome(
        id=execution.id,
        status="failed" if failed else "completed",
        title=title,
        summary=detail,
        expert=(
            to_home_expert(expert_by_id[execution.expert_id])
            if execution.expert_id in expert_by_id
            else None
        ),
        agent_name=name,
        occurred_at=execution.ended_at or execution.started_at,
        duration_seconds=execution.stats.duration if execution.stats else 0,
        cost_cents=execution.stats.cost if execution.stats else 0,
        link=run_link(library_id, execution.id),
    )
