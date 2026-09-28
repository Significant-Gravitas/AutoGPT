"""The user's hand-offs as one list: Otto's Delegations tab, an expert's Work
tab, and the Home rows that point at them.

A delegation is a ``delegate_to_expert`` thread (see ``delegation_db``), or
a hand-off still waiting on its approval card ("proposed"). Its status is
read from the thread itself: the turn state, a parked question, and whether
the last message is a reply, a failure, or a stop.
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ValidationError

from backend.api.features.experts import experts_db
from backend.copilot import delegation_db, delegation_list_db
from backend.copilot.delegation_list_db import HeldHandoff, ThreadMessages
from backend.copilot.delegation_settings import start_of_utc_day
from backend.copilot.delegation_threads_db import (
    DelegatedThread,
    DelegationFilter,
    ThreadStatus,
    count_delegated_threads,
    list_delegated_threads,
)
from backend.copilot.gate.handoff import HandoffCard
from backend.copilot.model import ChatSessionInfo
from backend.copilot.tools.models import DelegatedExpertInfo

DelegationStatus = Literal[
    "proposed",
    "queued",
    "running",
    "needs_input",
    "completed",
    "failed",
    "cancelled",
]
_MICRODOLLARS = 1_000_000
_TITLE_CHARS = 80
_TERMINAL: frozenset[str] = frozenset(
    {"needs_input", "completed", "failed", "cancelled"}
)


class DelegationSummary(BaseModel):
    sub_session_id: str | None
    review_id: str | None = None
    parent_session_id: str | None
    expert: DelegatedExpertInfo | None
    title: str
    brief: str
    status: DelegationStatus
    created_at: datetime
    finished_at: datetime | None
    elapsed_seconds: float | None
    cost_usd: float | None
    files_count: int
    question: str | None
    question_options: list[str]
    asked_at: datetime | None = None
    # Who handed the work off; None is Otto.
    delegated_by_expert_id: str | None = None


class DelegationCounts(BaseModel):
    working: int
    needs_you: int
    completed: int
    failed: int
    spent_today_usd: float


class DelegationListResponse(BaseModel):
    delegations: list[DelegationSummary]
    # Every hand-off matching the filters, across all pages.
    total: int
    summary: DelegationCounts


async def list_delegations(
    user_id: str,
    *,
    expert_id: str | None = None,
    parent_session_id: str | None = None,
    status: DelegationStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> DelegationListResponse:
    """A page of hand-offs, newest first, over held cards and threads alike.

    Every filter runs in SQL, so a page is cut from matching rows only;
    ``total`` counts every match, and ``summary`` every hand-off in scope
    whatever the ``status`` filter and page.
    """
    scope = DelegationFilter(expert_id=expert_id, parent_session_id=parent_session_id)
    window = offset + limit
    held = (
        await delegation_list_db.list_held_handoffs(user_id, scope, limit=window)
        if status in (None, "proposed")
        else []
    )
    threads = (
        await list_delegated_threads(user_id, scope, status=status, limit=window)
        if status != "proposed"
        else []
    )
    rows = [
        *(p for h in held if (p := _proposal(h))),
        *await summarize_threads(user_id, threads),
    ]
    rows.sort(key=lambda row: row.created_at, reverse=True)
    counts = await count_delegated_threads(user_id, scope)
    held_count = await delegation_list_db.count_held_handoffs(user_id, scope)
    return DelegationListResponse(
        delegations=rows[offset:window],
        total=_total(counts, held_count, status),
        summary=await _summary_counts(user_id, counts, held_count),
    )


async def delegated_questions(user_id: str, limit: int = 10) -> list[DelegationSummary]:
    """Delegated threads paused on a question to the user, newest first."""
    threads = await list_delegated_threads(
        user_id, DelegationFilter(), status="needs_input", limit=limit
    )
    return await summarize_threads(user_id, threads)


async def recent_delegations(user_id: str, limit: int = 50) -> list[DelegationSummary]:
    """The newest delegated threads, whatever their state."""
    threads = await list_delegated_threads(user_id, DelegationFilter(), limit=limit)
    return await summarize_threads(user_id, threads)


async def summarize_threads(
    user_id: str, found: list[DelegatedThread]
) -> list[DelegationSummary]:
    """One summary per delegated thread, read in a fixed number of queries."""
    threads = [t.session for t in found]
    status_by_id: dict[str, DelegationStatus] = {
        t.session.session_id: t.status for t in found
    }
    ids = [t.session_id for t in threads]
    messages = await delegation_list_db.get_thread_messages(user_id, ids)
    costs = await delegation_db.get_session_costs(user_id, ids)
    files = await delegation_list_db.count_session_files(user_id, ids)
    experts = await expert_infos(user_id, {t.expert_id for t in threads if t.expert_id})
    return [
        _summary(
            t,
            status_by_id[t.session_id],
            messages.get(t.session_id, ThreadMessages()),
            costs.get(t.session_id),
            files.get(t.session_id, 0),
            experts.get(t.expert_id or ""),
        )
        for t in threads
    ]


async def expert_infos(
    user_id: str, expert_ids: set[str]
) -> dict[str, DelegatedExpertInfo]:
    infos: dict[str, DelegatedExpertInfo] = {}
    for expert_id in expert_ids:
        expert = await experts_db.get_expert(
            user_id, expert_id, include_workflows=False, include_archived=True
        )
        if expert is not None:
            infos[expert_id] = DelegatedExpertInfo(
                id=expert.id,
                name=expert.name,
                role=expert.role,
                avatar_url=expert.avatar_url,
                color=expert.color,
            )
    return infos


def brief_of(first_message: str | None) -> str:
    """The task as the delegator wrote it, without the hand-off preamble.

    ``delegate_to_expert`` frames the prompt with bracketed paragraphs (who
    asked, optional context) before the task itself.
    """
    text = (first_message or "").strip()
    while text.startswith("[") and "]\n\n" in text:
        head, rest = text.split("]\n\n", 1)
        if "\n\n" in head:
            break
        text = rest.strip()
    return text


def title_of(brief: str) -> str:
    first_line = next((line.strip() for line in brief.splitlines() if line.strip()), "")
    return first_line[:_TITLE_CHARS] or "Delegated task"


def _summary(
    thread: ChatSessionInfo,
    status: DelegationStatus,
    messages: ThreadMessages,
    cost_microdollars: int | None,
    files_count: int,
    expert: DelegatedExpertInfo | None,
) -> DelegationSummary:
    brief = brief_of(messages.first_user_content)
    finished_at = messages.last_at if status in _TERMINAL else None
    pending = thread.metadata.pending_question
    return DelegationSummary(
        sub_session_id=thread.session_id,
        parent_session_id=thread.metadata.delegated_by_session_id,
        expert=expert,
        title=title_of(brief),
        brief=brief,
        status=status,
        created_at=_utc(thread.started_at),
        finished_at=finished_at,
        elapsed_seconds=_elapsed(thread.started_at, finished_at),
        cost_usd=(cost_microdollars or 0) / _MICRODOLLARS,
        files_count=files_count,
        question=pending.text if pending else None,
        question_options=pending.options if pending else [],
        asked_at=pending.asked_at if pending else None,
        delegated_by_expert_id=thread.metadata.delegated_by_expert_id,
    )


def _proposal(held: HeldHandoff) -> DelegationSummary | None:
    try:
        card = HandoffCard.model_validate(held.payload.get("handoff"))
    except ValidationError:
        return None
    return DelegationSummary(
        sub_session_id=None,
        review_id=held.review_id,
        parent_session_id=held.parent_session_id,
        expert=DelegatedExpertInfo(
            id=card.expert_id,
            name=card.expert_name,
            role=card.expert_role,
            avatar_url=card.expert_avatar_url,
            color=card.expert_color,
        ),
        title=title_of(card.brief),
        brief=card.brief,
        status="proposed",
        created_at=_utc(held.created_at),
        finished_at=None,
        elapsed_seconds=None,
        cost_usd=None,
        files_count=0,
        question=None,
        question_options=[],
    )


def _total(
    counts: dict[ThreadStatus, int], held: int, status: DelegationStatus | None
) -> int:
    if status is None:
        return held + sum(counts.values())
    if status == "proposed":
        return held
    return counts.get(status, 0)


async def _summary_counts(
    user_id: str, counts: dict[ThreadStatus, int], held: int
) -> DelegationCounts:
    spent = await delegation_db.get_delegation_spend_since(user_id, start_of_utc_day())
    return DelegationCounts(
        working=counts.get("running", 0) + counts.get("queued", 0),
        needs_you=counts.get("needs_input", 0) + held,
        completed=counts.get("completed", 0),
        failed=counts.get("failed", 0),
        spent_today_usd=round(spent / _MICRODOLLARS, 6),
    )


def _elapsed(started: datetime, finished: datetime | None) -> float:
    """Raw reads return the zone-less UTC columns naive; compare them as UTC."""
    end = _utc(finished) if finished else datetime.now(UTC)
    return round(max(0.0, (end - _utc(started)).total_seconds()), 2)


def _utc(at: datetime) -> datetime:
    return at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
