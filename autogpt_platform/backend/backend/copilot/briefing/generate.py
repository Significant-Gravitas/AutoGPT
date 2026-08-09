import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import NamedTuple, TypedDict
from urllib.parse import quote
from zoneinfo import ZoneInfo

from pydantic import ValidationError

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
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.util.clients import get_database_manager_async_client
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.timezone_utils import get_user_timezone_or_utc

from .models import BriefingContent, BriefingDecisionItem, BriefingRunItem

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"COMPLETED", "FAILED"}
_BRIEFED_STATUSES = [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]
_MAX_RUN_ITEMS = 10
_MAX_DECISION_ITEMS = 10
# Headroom over _MAX_RUN_ITEMS: the fetched window is filtered again in
# Python (expert-owned runs only), so fetching exactly 10 could yield zero
# briefable runs for a user who also runs plain agents.
_EXECUTION_FETCH_LIMIT = 50
# Shared fallback for a run whose agent couldn't be resolved in the library.
_DEFAULT_AGENT_NAME = "Agent"
_LIBRARY_LINK = "/library"


class BriefingResult(TypedDict, total=False):
    """Outcome of one :func:`generate_and_deliver_briefing` call.

    ``status`` is always present; the rest depend on it — ``reason`` on a
    skip, ``briefing_id``/``session_id`` on a delivery.
    """

    status: str
    reason: str
    briefing_id: str
    session_id: str | None


class AgentInfo(NamedTuple):
    name: str
    library_agent_id: str | None


def _run_link(info: AgentInfo | None, execution_id: str) -> str | None:
    """Deep link that opens a specific run on the library agent page.

    ``activeTab``/``activeItem`` are the params that page actually parses
    (see ``NewAgentLibraryView``); ids are percent-encoded so a link target
    can never carry markdown or URL metacharacters.
    """
    if info and info.library_agent_id:
        return (
            f"/library/agents/{quote(info.library_agent_id)}"
            f"?activeTab=runs&activeItem={quote(execution_id)}"
        )
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
    for execution in terminal[:_MAX_RUN_ITEMS]:
        info = agent_info_by_graph_id.get(execution.graph_id)
        expert = experts_by_id.get(execution.expert_id) if execution.expert_id else None
        run_items.append(
            BriefingRunItem(
                expert_id=expert.id if expert else None,
                expert_name=expert.name if expert else None,
                expert_avatar_url=expert.avatar_url if expert else None,
                agent_name=info.name if info else _DEFAULT_AGENT_NAME,
                graph_id=execution.graph_id,
                execution_id=execution.id,
                library_agent_id=info.library_agent_id if info else None,
                status=str(execution.status),
                summary=_activity_summary(execution.stats),
                link=_run_link(info, execution.id),
            )
        )

    expert_id_by_exec = {e.id: e.expert_id for e in executions}
    decision_items = []
    # Capped like run_items: an uncapped list turns a user with 100 waiting
    # reviews into a 100-line assistant message that also rides along in
    # every later LLM turn of that session. The renderer points the overflow
    # at the needs-attention list.
    for review in reviews[:_MAX_DECISION_ITEMS]:
        if review.graph_exec_id.startswith(COPILOT_SESSION_PREFIX):
            session_id = review.graph_exec_id.removeprefix(COPILOT_SESSION_PREFIX)
            link = f"/copilot?sessionId={quote(session_id)}"
        else:
            info = agent_info_by_graph_id.get(review.graph_id)
            link = _run_link(info, review.graph_exec_id) or _LIBRARY_LINK
        # _enrich_pending_reviews already resolved expert attribution on the
        # review model (including copilot-session reviews and executions older
        # than the 24h window); the local lookup only backfills gaps.
        fallback = experts_by_id.get(
            review.expert_id or expert_id_by_exec.get(review.graph_exec_id) or ""
        )
        decision_items.append(
            BriefingDecisionItem(
                node_exec_id=review.node_exec_id,
                graph_exec_id=review.graph_exec_id,
                title=review.instructions or "Review needed",
                expert_id=review.expert_id or (fallback.id if fallback else None),
                expert_name=review.expert_name or (fallback.name if fallback else None),
                expert_avatar_url=review.expert_avatar_url
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
        decision_total=len(reviews),
    )


# Agent names, AI-generated summaries and review instructions can all
# originate from a third-party/marketplace agent. Escaping the markdown
# metacharacters that carry structure stops that text from breaking out of
# the link syntax it is interpolated into and spoofing the label or target
# rendered in the user's thread.
_MARKDOWN_META_RE = re.compile(r"([\\`*_\[\]()<>])")


def _md(text: str) -> str:
    """Escape untrusted text for inline interpolation into markdown."""
    collapsed = " ".join(text.split())
    return _MARKDOWN_META_RE.sub(r"\\\1", collapsed)


def _md_link(label: str, target: str | None) -> str:
    """Render ``label`` as a markdown link, or as plain text if it can't be.

    Composed targets are relative, percent-encoded paths. Enforcing that in
    code — rather than by convention — keeps an absolute or ``javascript:``
    target from ever reaching the user's thread as a clickable link.
    """
    if not target or not target.startswith("/") or target.startswith("//"):
        return label
    return f"[{label}]({target})"


def render_briefing_markdown(content: BriefingContent) -> str:
    lines = ["## ☀️ Your morning briefing", ""]
    if content.run_items:
        lines.append("**What ran**")
        for item in content.run_items:
            who = f"{_md(item.expert_name)}: " if item.expert_name else ""
            outcome = "completed" if item.status == "COMPLETED" else "failed"
            name = _md_link(_md(item.agent_name), item.link)
            lines.append(f"- {who}{name} — {outcome}")
        lines.append("")
    found = [item for item in content.run_items if item.summary]
    if found:
        lines.append("**What was found**")
        lines.extend(
            f"- **{_md(item.agent_name)}**: {_md(item.summary or '')}" for item in found
        )
        lines.append("")
    if content.decision_items:
        total = max(content.decision_total, len(content.decision_items))
        lines.append(f"**Needs your decision ({total})**")
        lines.extend(
            f"- {_md_link(_md(d.title), d.link)}" for d in content.decision_items
        )
        remaining = total - len(content.decision_items)
        if remaining > 0:
            lines.append(f"- …and {remaining} more on your home page")
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
                wf.name or (existing.name if existing else _DEFAULT_AGENT_NAME),
                wf.library_agent_id
                or (existing.library_agent_id if existing else None),
            )


def _stored_briefing_content(content: dict) -> BriefingContent | None:
    """Re-validate a stored briefing for redelivery.

    ``None`` means the stored shape is unreadable (written by a different
    composer version) — the caller recomposes instead of failing the run.
    """
    try:
        return BriefingContent.model_validate(content)
    except ValidationError:
        logger.warning(
            "Stored briefing content failed validation; recomposing instead",
            exc_info=True,
        )
        return None


async def _compose_fresh_briefing(
    user_id: str, now_local: datetime, tz_name: str
) -> BriefingContent | None:
    window_start = (now_local - timedelta(hours=24)).astimezone(dt_timezone.utc)
    # The three reads are independent; run them concurrently so the briefing
    # costs one round-trip's latency rather than three. The library lookup
    # can't join them — it needs the graph ids the executions resolve to.
    experts, executions, reviews = await asyncio.gather(
        experts_db().list_experts(user_id),
        # Filter and bound in the query: a user running minute-crons can
        # accumulate thousands of executions a day, and every one of them
        # would otherwise be serialized over the DatabaseManager RPC just to
        # render at most _MAX_RUN_ITEMS bullets.
        execution_db().get_graph_executions(
            user_id=user_id,
            created_time_gte=window_start,
            statuses=_BRIEFED_STATUSES,
            limit=_EXECUTION_FETCH_LIMIT,
        ),
        review_db().get_pending_reviews_for_user(user_id, 1, 100),
    )
    # Resolve only the graphs actually referenced, rather than paging the
    # library: a user with >100 agents used to miss the very one being
    # briefed and fall back to an unlinkable "Agent" row.
    graph_ids = list({e.graph_id for e in executions} | {r.graph_id for r in reviews})
    refs = await library_db().get_library_agent_refs_by_graph_ids(user_id, graph_ids)
    agent_info: dict[str, AgentInfo] = {
        ref.graph_id: AgentInfo(ref.name or _DEFAULT_AGENT_NAME, ref.id) for ref in refs
    }
    _merge_agent_info(agent_info, experts)

    return compose_briefing(
        experts=experts,
        executions=executions,
        reviews=reviews,
        agent_info_by_graph_id=agent_info,
        generated_at=now_local,
        tz_name=tz_name,
    )


async def generate_and_deliver_briefing(user_id: str) -> BriefingResult:
    if not await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False):
        return {"status": "skipped", "reason": "flag_disabled"}

    user = await user_db().get_user_by_id(user_id)
    tz_name = get_user_timezone_or_utc(user.timezone)
    now_local = datetime.now(ZoneInfo(tz_name))
    briefing_date = now_local.date()

    client = get_database_manager_async_client()
    record = await client.get_briefing_for_date(user_id, briefing_date)
    if record and record.delivered_at:
        return {"status": "skipped", "reason": "already_delivered"}

    # An undelivered record means a prior run stored the briefing but the
    # session post failed — redeliver the stored content so the record and
    # the posted message can't diverge.
    stored = _stored_briefing_content(record.content) if record else None
    if record is not None and stored is not None:
        content = stored
    else:
        content = await _compose_fresh_briefing(user_id, now_local, tz_name)
        if content is None:
            if record is not None:
                # An unreadable row whose recompose now yields nothing would
                # otherwise be re-gathered and re-composed on every future
                # run. Stamp it so this user's cron stops reprocessing it.
                await client.mark_briefing_delivered(user_id, record.id)
            return {"status": "skipped", "reason": "nothing_to_say"}
        if record is None:
            record = await client.create_briefing(
                user_id, briefing_date, content.model_dump(mode="json")
            )
        else:
            # Recompose path: write the fresh content back, or the home card
            # (which re-validates) would keep skipping this row and show an
            # older briefing than the one posted into the thread.
            await client.update_briefing_content(
                user_id, record.id, content.model_dump(mode="json")
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
    await client.mark_briefing_delivered(user_id, record.id)
    return {
        "status": "delivered",
        "briefing_id": record.id,
        "session_id": session_id,
    }
