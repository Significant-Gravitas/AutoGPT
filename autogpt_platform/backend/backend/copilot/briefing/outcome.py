"""One execution, one briefing item — shared by every consumer.

The 9am job (``backend/copilot/briefing/generate.py``) and the home dashboard
(``backend/api/features/home/briefing.py``) both describe "what happened
overnight". Composing that description in two places let the copilot thread and
the home card disagree about the same run, so both now come through here.
"""

from datetime import datetime, timezone
from urllib.parse import quote

from backend.api.features.experts.models import Expert
from backend.data.execution import ExecutionStatus, GraphExecutionMeta

from .models import BriefingRunItem

# Shared fallback for a run whose agent couldn't be resolved in the library.
DEFAULT_AGENT_NAME = "Agent task"
_FAILED_DETAIL = "Open the run to inspect the failure and choose the next step."
_COMPLETED_DETAIL = "Completed successfully."
# Longest a summary's first sentence may run before it is clipped into a title.
_TITLE_MAX = 120


def compose_run_outcome(
    execution: GraphExecutionMeta,
    *,
    agent_name: str,
    library_agent_id: str | None,
    expert: Expert | None,
) -> BriefingRunItem:
    """Describe one terminal execution as a briefing item."""
    failed = execution.status == ExecutionStatus.FAILED
    stats = execution.stats
    raw_summary = stats.activity_status if stats else None
    fallback_title, fallback_detail = outcome_fallbacks(
        agent_name, failed=failed, error=stats.error if stats else None
    )
    # Only an AI summary may become the headline. A raw exception string is the
    # detail, never the card title.
    title, detail = split_summary(
        raw_summary, fallback_title=fallback_title, fallback_detail=fallback_detail
    )
    return BriefingRunItem(
        expert_id=expert.id if expert else None,
        expert_name=expert.name if expert else None,
        expert_role=expert.role if expert else None,
        expert_avatar_url=expert.avatar_url if expert else None,
        agent_name=agent_name,
        graph_id=execution.graph_id,
        execution_id=execution.id,
        library_agent_id=library_agent_id,
        status="FAILED" if failed else "COMPLETED",
        summary=raw_summary,
        title=title,
        detail=detail,
        occurred_at=as_utc_or_none(execution.ended_at or execution.started_at),
        duration_seconds=stats.duration if stats else 0,
        cost_cents=stats.cost if stats else 0,
        link=run_link(library_agent_id, execution.id),
    )


def outcome_fallbacks(
    agent_name: str, *, failed: bool, error: str | None
) -> tuple[str, str]:
    """Headline and detail for a run that carries no AI summary."""
    if failed:
        return f"{agent_name} needs a retry", error or _FAILED_DETAIL
    return f"{agent_name} finished", _COMPLETED_DETAIL


def split_summary(
    value: str | None, *, fallback_title: str, fallback_detail: str
) -> tuple[str, str]:
    compact = " ".join(value.split()) if value else ""
    if not compact:
        return fallback_title, fallback_detail
    if ". " not in compact:
        return compact[:_TITLE_MAX], fallback_detail
    title, detail = compact.split(". ", 1)
    # A first sentence can run past the limit too — clip it like the
    # single-sentence branch, or one rambling summary becomes a card headline
    # hundreds of characters wide.
    return f"{title[: _TITLE_MAX - 1].rstrip()}.", detail


def run_link(library_agent_id: str | None, execution_id: str) -> str | None:
    """Deep link that opens a specific run on the library agent page.

    ``activeTab``/``activeItem`` are the params that page actually parses
    (see ``NewAgentLibraryView``). Both ids are encoded with ``safe=""`` so
    they stay single components: ``quote`` keeps ``/`` by default, and an id
    carrying one would otherwise redraw the path boundary. That also stops a
    link target from carrying markdown or URL metacharacters.

    Same route contract as the frontend's ``getReviewLink``
    (``frontend/src/lib/review-links.ts``) — change both together.
    """
    if not library_agent_id:
        return None
    return (
        f"/library/agents/{quote(library_agent_id, safe='')}"
        f"?activeTab=runs&activeItem={quote(execution_id, safe='')}"
    )


def as_utc(value: datetime) -> datetime:
    """Stored timestamps can come back naive; comparing those to an aware `now`
    raises, so pin anything naive to UTC before it reaches arithmetic or sorting.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def as_utc_or_none(value: datetime | None) -> datetime | None:
    return as_utc(value) if value else None
