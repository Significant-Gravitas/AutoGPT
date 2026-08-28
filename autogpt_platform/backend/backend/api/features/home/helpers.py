from datetime import datetime

from pydantic import BaseModel, ConfigDict

from backend.api.features.experts.models import Expert
from backend.api.features.library.model import LibraryAgentRef
from backend.copilot.briefing.outcome import DEFAULT_AGENT_NAME, as_utc
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.data.execution import GraphExecutionMeta

from .models import HomeExpert


def to_home_expert(expert: Expert) -> HomeExpert:
    return HomeExpert(
        id=expert.id,
        name=expert.name,
        role=expert.role,
        avatar_url=expert.avatar_url,
    )


def experts_by_schedule(
    experts: list[Expert], schedules: list[GraphExecutionJobInfo]
) -> dict[str, Expert]:
    """Map job id to owning expert.

    `graph_id` is not a schedule key — one graph can back several jobs, and an
    org context mixes in teammates' jobs — so attribute a job by its own expert
    stamp, falling back to the workflow whose schedule created it.
    """
    expert_by_id = {expert.id: expert for expert in experts}
    expert_by_schedule_id = {
        workflow.schedule_id: expert
        for expert in experts
        for workflow in expert.workflows
        if workflow.schedule_id
    }
    owners = {
        schedule.id: expert_by_id.get(schedule.expert_id or "")
        or expert_by_schedule_id.get(schedule.id)
        for schedule in schedules
    }
    return {schedule_id: expert for schedule_id, expert in owners.items() if expert}


def next_runs_by_expert(
    schedules: list[GraphExecutionJobInfo], expert_by_schedule: dict[str, Expert]
) -> dict[str, datetime]:
    """Earliest upcoming run per expert, across every job that expert owns."""
    earliest: dict[str, datetime] = {}
    for schedule in schedules:
        expert = expert_by_schedule.get(schedule.id)
        next_run = parse_datetime(schedule.next_run_time)
        if not expert or next_run is None:
            continue
        current = earliest.get(expert.id)
        if current is None or next_run < current:
            earliest[expert.id] = next_run
    return earliest


class AgentRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    library_agent_id: str | None


UNKNOWN_AGENT = AgentRef(name=DEFAULT_AGENT_NAME, library_agent_id=None)
AgentScopeKey = tuple[str, str | None, str | None]
AgentScopeMap = dict[AgentScopeKey, AgentRef] | dict[str, AgentRef]


def agent_refs_by_graph(
    experts: list[Expert], refs: list[LibraryAgentRef]
) -> dict[AgentScopeKey, AgentRef]:
    agents = {
        (ref.graph_id, ref.organization_id, ref.team_id): AgentRef(
            name=ref.name or UNKNOWN_AGENT.name, library_agent_id=ref.id
        )
        for ref in refs
    }
    for expert in experts:
        for workflow in expert.workflows:
            if workflow.graph_id:
                key = (workflow.graph_id, expert.organization_id, expert.team_id)
                current = agents.get(key, UNKNOWN_AGENT)
                agents[key] = AgentRef(
                    name=workflow.name or current.name,
                    library_agent_id=workflow.library_agent_id
                    or current.library_agent_id,
                )
    return agents


def agent_ref_for_execution(
    agents: AgentScopeMap, execution: GraphExecutionMeta
) -> AgentRef:
    exact = (execution.graph_id, execution.organization_id, execution.team_id)
    return agents.get(exact, agents.get(execution.graph_id, UNKNOWN_AGENT))


def setup_count(expert: Expert) -> int:
    return sum(
        bool(workflow.schedule_cron and not workflow.schedule_id)
        for workflow in expert.workflows
    )


def parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return as_utc(parsed)
