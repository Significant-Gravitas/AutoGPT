from datetime import datetime
from typing import NamedTuple

from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts.models import Expert
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.data.execution import GraphExecutionMeta

from .models import BriefingContent, BriefingDecisionItem, BriefingRunItem

_TERMINAL_STATUSES = {"COMPLETED", "FAILED"}
_MAX_RUN_ITEMS = 10


class AgentInfo(NamedTuple):
    name: str
    library_agent_id: str | None


def _run_link(info: AgentInfo | None, execution_id: str) -> str | None:
    if info and info.library_agent_id:
        return f"/library/agents/{info.library_agent_id}?executionId={execution_id}"
    return None


def _activity_summary(stats: GraphExecutionMeta.Stats | dict | None) -> str | None:
    """Pull the AI-generated activity summary off an execution's stats.

    ``GraphExecutionMeta.stats`` is a ``Stats`` pydantic model in
    production; tests stub it with a plain dict. Handle both.
    """
    if isinstance(stats, dict):
        return stats.get("activity_status")
    return getattr(stats, "activity_status", None)


def compose_briefing(
    *,
    experts: list[Expert],
    executions: list[GraphExecutionMeta],
    reviews: list[PendingHumanReviewModel],
    agent_info_by_graph_id: dict[str, AgentInfo],
    generated_at: datetime,
    tz_name: str,
) -> BriefingContent | None:
    experts_by_id = {e.id: e for e in experts}
    zero_expert_fallback = not experts

    # ExecutionStatus (backend.data.execution.ExecutionStatus) is a
    # prisma StrEnum, so str(e.status) already yields the plain value
    # (e.g. "COMPLETED") regardless of whether e.status is a real enum
    # member or a bare string (as in tests).
    terminal = [e for e in executions if str(e.status) in _TERMINAL_STATUSES]
    if not zero_expert_fallback:
        terminal = [e for e in terminal if e.expert_id]
    terminal.sort(key=lambda e: str(e.status) != "FAILED")

    run_items = []
    for e in terminal[:_MAX_RUN_ITEMS]:
        info = agent_info_by_graph_id.get(e.graph_id)
        expert = experts_by_id.get(e.expert_id) if e.expert_id else None
        run_items.append(
            BriefingRunItem(
                expert_id=expert.id if expert else None,
                expert_name=expert.name if expert else None,
                expert_avatar_url=expert.avatar_url if expert else None,
                agent_name=info.name if info else "Agent",
                graph_id=e.graph_id,
                execution_id=e.id,
                library_agent_id=info.library_agent_id if info else None,
                status=str(e.status),
                summary=_activity_summary(e.stats),
                link=_run_link(info, e.id),
            )
        )

    expert_id_by_exec = {e.id: e.expert_id for e in executions}
    decision_items = []
    for r in reviews:
        if r.graph_exec_id.startswith(COPILOT_SESSION_PREFIX):
            link = f"/copilot?sessionId={r.graph_exec_id.removeprefix(COPILOT_SESSION_PREFIX)}"
            expert = None
        else:
            info = agent_info_by_graph_id.get(r.graph_id)
            link = _run_link(info, r.graph_exec_id) or "/library"
            expert = experts_by_id.get(expert_id_by_exec.get(r.graph_exec_id) or "")
        decision_items.append(
            BriefingDecisionItem(
                node_exec_id=r.node_exec_id,
                graph_exec_id=r.graph_exec_id,
                title=r.instructions or "Review needed",
                expert_id=expert.id if expert else None,
                expert_name=expert.name if expert else None,
                expert_avatar_url=expert.avatar_url if expert else None,
                link=link,
            )
        )

    if not run_items and not decision_items:
        return None
    return BriefingContent(
        generated_at=generated_at,
        timezone=tz_name,
        zero_expert_fallback=zero_expert_fallback,
        run_items=run_items,
        decision_items=decision_items,
    )


def render_briefing_markdown(content: BriefingContent) -> str:
    lines = ["## ☀️ Your morning briefing", ""]
    if content.run_items:
        lines.append("**What ran**")
        for i in content.run_items:
            who = f"{i.expert_name}: " if i.expert_name else ""
            outcome = "completed" if i.status == "COMPLETED" else "failed"
            name = f"[{i.agent_name}]({i.link})" if i.link else i.agent_name
            lines.append(f"- {who}{name} — {outcome}")
        lines.append("")
    found = [i for i in content.run_items if i.summary]
    if found:
        lines.append("**What was found**")
        lines.extend(f"- **{i.agent_name}**: {i.summary}" for i in found)
        lines.append("")
    if content.decision_items:
        lines.append(f"**Needs your decision ({len(content.decision_items)})**")
        lines.extend(f"- [{d.title}]({d.link})" for d in content.decision_items)
    return "\n".join(lines).strip()
