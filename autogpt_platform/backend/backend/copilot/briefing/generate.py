import uuid
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import NamedTuple
from zoneinfo import ZoneInfo

from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts.models import Expert
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.data.db_accessors import (
    execution_db,
    experts_db,
    library_db,
    review_db,
    user_db,
)
from backend.data.execution import GraphExecutionMeta
from backend.util.clients import get_database_manager_async_client
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.timezone_utils import get_user_timezone_or_utc

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
        else:
            info = agent_info_by_graph_id.get(r.graph_id)
            link = _run_link(info, r.graph_exec_id) or "/library"
        # _enrich_pending_reviews already resolved expert attribution on the
        # review model (including copilot-session reviews and executions older
        # than the 24h window); the local lookup only backfills gaps.
        fallback = experts_by_id.get(
            r.expert_id or expert_id_by_exec.get(r.graph_exec_id) or ""
        )
        decision_items.append(
            BriefingDecisionItem(
                node_exec_id=r.node_exec_id,
                graph_exec_id=r.graph_exec_id,
                title=r.instructions or "Review needed",
                expert_id=r.expert_id or (fallback.id if fallback else None),
                expert_name=r.expert_name or (fallback.name if fallback else None),
                expert_avatar_url=r.expert_avatar_url
                or (fallback.avatar_url if fallback else None),
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


# Fixed namespace so the same (user, local calendar date) always derives the
# same message id — retries and double-fires of the scheduler job dedupe via
# append_plain_session_message's message_id uniqueness check.
_BRIEFING_NAMESPACE = uuid.UUID("7f1c2d3e-9a4b-4c5d-8e6f-0a1b2c3d4e5f")


def _merge_agent_info(agent_info: dict[str, AgentInfo], experts: list[Expert]) -> None:
    for expert in experts:
        for wf in expert.workflows:
            if not wf.graph_id:
                continue
            existing = agent_info.get(wf.graph_id)
            agent_info[wf.graph_id] = AgentInfo(
                wf.name or (existing.name if existing else "Agent"),
                wf.library_agent_id
                or (existing.library_agent_id if existing else None),
            )


async def generate_and_deliver_briefing(user_id: str) -> dict:
    if not await is_feature_enabled(Flag.MORNING_BRIEFING, user_id, default=False):
        return {"status": "skipped", "reason": "flag_disabled"}

    user = await user_db().get_user_by_id(user_id)
    tz_name = get_user_timezone_or_utc(user.timezone)
    now_local = datetime.now(ZoneInfo(tz_name))
    briefing_date = now_local.date()

    client = get_database_manager_async_client()
    if await client.get_briefing_for_date(user_id, briefing_date):
        return {"status": "skipped", "reason": "already_delivered"}

    window_start = (now_local - timedelta(hours=24)).astimezone(dt_timezone.utc)
    experts = await experts_db().list_experts(user_id)
    executions = await execution_db().get_graph_executions(
        user_id=user_id, created_time_gte=window_start
    )
    reviews = await review_db().get_pending_reviews_for_user(user_id, 1, 100)

    library = await library_db().list_library_agents(user_id, page_size=100)
    agent_info: dict[str, AgentInfo] = {
        agent.graph_id: AgentInfo(agent.name, agent.id) for agent in library.agents
    }
    _merge_agent_info(agent_info, experts)

    content = compose_briefing(
        experts=experts,
        executions=executions,
        reviews=reviews,
        agent_info_by_graph_id=agent_info,
        generated_at=now_local,
        tz_name=tz_name,
    )
    if content is None:
        return {"status": "skipped", "reason": "nothing_to_say"}

    record = await client.create_briefing(
        user_id, briefing_date, content.model_dump(mode="json")
    )
    message_id = str(
        uuid.uuid5(
            _BRIEFING_NAMESPACE,
            f"morning-briefing:{user_id}:{briefing_date.isoformat()}",
        )
    )
    session_id = await client.append_plain_session_message(
        user_id=user_id,
        content=render_briefing_markdown(content),
        message_id=message_id,
        metadata={"kind": "morning_briefing", "briefing_id": record.id},
    )
    return {
        "status": "delivered",
        "briefing_id": record.id,
        "session_id": session_id,
    }
