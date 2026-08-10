from backend.api.features.experts.models import Expert
from backend.executor.scheduler import GraphExecutionJobInfo

from .helpers import parse_datetime, setup_count, to_home_expert
from .models import HomeAgentStatus, HomeTeamSummary


def compose_agent_statuses(
    *,
    experts: list[Expert],
    running_expert_ids: set[str],
    schedule_by_graph: dict[str, GraphExecutionJobInfo],
) -> list[HomeAgentStatus]:
    statuses = [
        _agent_status(expert, running_expert_ids, schedule_by_graph)
        for expert in experts
    ]
    rank = {"working": 0, "paused": 1, "needs_setup": 2, "failed": 3, "ready": 4}
    return sorted(statuses, key=lambda item: rank[item.status])


def compose_team_summary(agents: list[HomeAgentStatus]) -> HomeTeamSummary:
    return HomeTeamSummary(
        total=len(agents),
        ready=sum(agent.status == "ready" for agent in agents),
        working=sum(agent.status == "working" for agent in agents),
        needs_attention=sum(
            agent.status in {"paused", "needs_setup", "failed"} for agent in agents
        ),
    )


def _agent_status(
    expert: Expert,
    running_expert_ids: set[str],
    schedule_by_graph: dict[str, GraphExecutionJobInfo],
) -> HomeAgentStatus:
    if expert.id in running_expert_ids:
        status, detail = "working", "Working on a task now"
    elif expert.schedules_paused_at:
        status, detail = "paused", "Scheduled work is paused"
    elif setup_count(expert) > 0:
        status, detail = "needs_setup", "A connection needs setup"
    elif expert.last_run_status == "FAILED":
        status, detail = "failed", "Last run failed"
    else:
        status, detail = "ready", "Ready for the next task"
    times = [
        parse_datetime(schedule_by_graph[workflow.graph_id].next_run_time)
        for workflow in expert.workflows
        if workflow.graph_id in schedule_by_graph
    ]
    next_run = min((value for value in times if value), default=None)
    return HomeAgentStatus(
        expert=to_home_expert(expert),
        status=status,
        detail=detail,
        next_run_time=next_run,
    )
