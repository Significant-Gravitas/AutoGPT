from datetime import datetime, timezone
from urllib.parse import quote

from backend.api.features.experts.models import Expert
from backend.api.features.library.model import LibraryAgentRef
from backend.executor.scheduler import GraphExecutionJobInfo

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


def agent_names_by_graph(
    experts: list[Expert], refs: list[LibraryAgentRef]
) -> dict[str, tuple[str, str | None]]:
    names: dict[str, tuple[str, str | None]] = {
        ref.graph_id: (ref.name or "Agent task", ref.id) for ref in refs
    }
    for expert in experts:
        for workflow in expert.workflows:
            if workflow.graph_id:
                current = names.get(workflow.graph_id, ("Agent task", None))
                names[workflow.graph_id] = (
                    workflow.name or current[0],
                    workflow.library_agent_id or current[1],
                )
    return names


def setup_count(expert: Expert) -> int:
    return sum(
        bool(workflow.schedule_cron and not workflow.schedule_id)
        for workflow in expert.workflows
    )


def as_utc(value: datetime) -> datetime:
    """Stored timestamps can come back naive; comparing those to an aware `now`
    raises, so pin anything naive to UTC before it reaches arithmetic or sorting.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def as_utc_or_none(value: datetime | None) -> datetime | None:
    return as_utc(value) if value else None


def parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return as_utc(parsed)


def run_link(library_id: str | None, execution_id: str) -> str | None:
    if not library_id:
        return None
    return (
        f"/library/agents/{quote(library_id)}"
        f"?activeTab=runs&activeItem={quote(execution_id)}"
    )


def split_summary(
    value: str | None, *, fallback_title: str, fallback_detail: str
) -> tuple[str, str]:
    compact = " ".join(value.split()) if value else ""
    if not compact:
        return fallback_title, fallback_detail
    if ". " not in compact:
        return compact[:120], fallback_detail
    title, detail = compact.split(". ", 1)
    return f"{title}.", detail
