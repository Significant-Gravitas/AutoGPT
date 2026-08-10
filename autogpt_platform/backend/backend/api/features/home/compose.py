from datetime import datetime

from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts.models import Expert
from backend.api.features.library.model import LibraryAgentRef
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.data.execution_cost_summary import UserExecutionCostSummary
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .activity import (
    compose_active_tasks,
    compose_briefing,
    compose_upcoming_tasks,
    compose_week_summary,
)
from .agents import compose_agent_statuses, compose_team_summary
from .attention import compose_attention_items
from .helpers import agent_names_by_graph, experts_by_graph
from .models import HomeDashboardResponse


def compose_home_dashboard(
    *,
    now: datetime,
    experts: list[Expert],
    executions: list[GraphExecutionMeta],
    reviews: list[PendingHumanReviewModel],
    schedules: list[GraphExecutionJobInfo | CopilotTurnJobInfo],
    library_refs: list[LibraryAgentRef],
    cost_summary: UserExecutionCostSummary,
    credits_balance: int | None,
    timezone_name: str,
) -> HomeDashboardResponse:
    hired = [
        expert
        for expert in experts
        if not expert.is_template and not expert.is_archived
    ]
    expert_by_id = {expert.id: expert for expert in hired}
    expert_by_graph = experts_by_graph(hired)
    agent_by_graph = agent_names_by_graph(hired, library_refs)
    schedule_by_graph = {
        schedule.graph_id: schedule
        for schedule in schedules
        if isinstance(schedule, GraphExecutionJobInfo)
    }
    running_expert_ids = {
        execution.expert_id
        for execution in executions
        if execution.expert_id
        and execution.status in {ExecutionStatus.RUNNING, ExecutionStatus.QUEUED}
    }
    agents = compose_agent_statuses(
        experts=hired,
        running_expert_ids=running_expert_ids,
        schedule_by_graph=schedule_by_graph,
    )

    return HomeDashboardResponse(
        generated_at=now,
        timezone=timezone_name,
        is_demo=False,
        attention=compose_attention_items(
            now=now,
            experts=hired,
            reviews=reviews,
            schedules=schedules,
            credits_balance=credits_balance,
        ),
        briefing=compose_briefing(
            now=now,
            executions=executions,
            expert_by_id=expert_by_id,
            agent_by_graph=agent_by_graph,
        ),
        active_tasks=compose_active_tasks(executions, expert_by_id, agent_by_graph),
        upcoming_tasks=compose_upcoming_tasks(schedules, expert_by_graph),
        team=compose_team_summary(agents),
        agents=agents,
        week=compose_week_summary(cost_summary, credits_balance),
    )
