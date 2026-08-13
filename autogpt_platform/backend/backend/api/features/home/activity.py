from backend.api.features.experts.models import Expert
from backend.copilot.briefing.outcome import as_utc_or_none, run_link
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.execution_cost_summary import UserExecutionCostSummary
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .helpers import UNKNOWN_AGENT, AgentRef, parse_datetime, to_home_expert
from .models import HomeActiveTask, HomeDailyActivity, HomeUpcomingTask, HomeWeekSummary

_MAX_ACTIVE = 3
_MAX_UPCOMING = 4


def compose_active_tasks(
    executions: list[GraphExecutionMeta],
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, AgentRef],
) -> list[HomeActiveTask]:
    active = [
        execution
        for execution in executions
        if execution.status in {ExecutionStatus.RUNNING, ExecutionStatus.QUEUED}
    ]
    return [
        _active_task(execution, expert_by_id, agent_by_graph)
        for execution in active[:_MAX_ACTIVE]
    ]


def _active_task(
    execution: GraphExecutionMeta,
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, AgentRef],
) -> HomeActiveTask:
    agent = agent_by_graph.get(execution.graph_id, UNKNOWN_AGENT)
    return HomeActiveTask(
        id=execution.id,
        title=agent.name,
        status="running" if execution.status == ExecutionStatus.RUNNING else "queued",
        expert=(
            to_home_expert(expert_by_id[execution.expert_id])
            if execution.expert_id in expert_by_id
            else None
        ),
        started_at=as_utc_or_none(execution.started_at),
        link=run_link(agent.library_agent_id, execution.id),
    )


def compose_upcoming_tasks(
    schedules: list[GraphExecutionJobInfo | CopilotTurnJobInfo],
    expert_by_schedule: dict[str, Expert],
) -> list[HomeUpcomingTask]:
    upcoming = []
    for schedule in schedules:
        next_run = parse_datetime(schedule.next_run_time)
        if next_run is None:
            continue
        if isinstance(schedule, GraphExecutionJobInfo):
            expert = expert_by_schedule.get(schedule.id)
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
