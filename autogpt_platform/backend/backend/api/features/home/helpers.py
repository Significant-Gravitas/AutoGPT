from datetime import datetime, timezone
from urllib.parse import quote

from backend.api.features.experts.models import Expert
from backend.api.features.library.model import LibraryAgentRef

from .models import HomeExpert


def to_home_expert(expert: Expert) -> HomeExpert:
    return HomeExpert(
        id=expert.id,
        name=expert.name,
        role=expert.role,
        avatar_url=expert.avatar_url,
    )


def experts_by_graph(experts: list[Expert]) -> dict[str, Expert]:
    return {
        workflow.graph_id: expert
        for expert in experts
        for workflow in expert.workflows
        if workflow.graph_id
    }


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


def parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


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
