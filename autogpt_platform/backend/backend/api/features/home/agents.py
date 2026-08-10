from datetime import datetime

from backend.api.features.experts.models import Expert

from .helpers import setup_count, to_home_expert
from .models import HomeAgentStatus, HomeTeamSummary


def compose_agent_statuses(
    *,
    experts: list[Expert],
    running_expert_ids: set[str],
    next_run_by_expert: dict[str, datetime],
) -> list[HomeAgentStatus]:
    statuses = [
        _agent_status(expert, running_expert_ids, next_run_by_expert)
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
    next_run_by_expert: dict[str, datetime],
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
    return HomeAgentStatus(
        expert=to_home_expert(expert),
        status=status,
        detail=detail,
        next_run_time=next_run_by_expert.get(expert.id),
    )
