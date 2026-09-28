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
from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
)
from backend.copilot.delegation_list_db import HeldHandoff, ThreadMessages
from backend.copilot.delegation_settings import start_of_utc_day
from backend.copilot.gate.handoff import HandoffCard
from backend.copilot.model import (
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
    ChatSessionInfo,
)
from backend.copilot.stream_registry import CANCELLED_MESSAGE
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
    summary: DelegationCounts


async def list_delegations(
    user_id: str,
    *,
    expert_id: str | None = None,
    parent_session_id: str | None = None,
    status: DelegationStatus | None = None,
    limit: int = 50,
) -> DelegationListResponse:
    """Newest first; ``summary`` counts the listed rows (before ``status``)."""
    threads = await delegation_list_db.list_delegated_sessions(
        user_id,
        expert_id=expert_id,
        parent_session_id=parent_session_id,
        limit=limit,
    )
    rows = [
        *await _proposed(user_id, expert_id, parent_session_id),
        *await summarize_threads(user_id, threads),
    ]
    rows.sort(key=lambda row: row.created_at, reverse=True)
    spent = await delegation_db.get_delegation_spend_since(user_id, start_of_utc_day())
    return DelegationListResponse(
        delegations=[r for r in rows if status is None or r.status == status][:limit],
        summary=_counts(rows, spent),
    )


async def delegated_questions(user_id: str, limit: int = 10) -> list[DelegationSummary]:
    """Delegated threads paused on a question to the user, newest first."""
    threads = await delegation_list_db.list_delegated_sessions(
        user_id, limit=limit, question_only=True
    )
    return await summarize_threads(user_id, threads)


async def recent_delegations(user_id: str, limit: int = 50) -> list[DelegationSummary]:
    """The newest delegated threads, whatever their state."""
    threads = await delegation_list_db.list_delegated_sessions(user_id, limit=limit)
    return await summarize_threads(user_id, threads)


async def summarize_threads(
    user_id: str, threads: list[ChatSessionInfo]
) -> list[DelegationSummary]:
    """One summary per delegated thread, read in a fixed number of queries."""
    ids = [t.session_id for t in threads]
    messages = await delegation_list_db.get_thread_messages(user_id, ids)
    costs = await delegation_db.get_session_costs(user_id, ids)
    files = await delegation_list_db.count_session_files(user_id, ids)
    experts = await expert_infos(user_id, {t.expert_id for t in threads if t.expert_id})
    return [
        _summary(
            t,
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


def status_of(thread: ChatSessionInfo, messages: ThreadMessages) -> DelegationStatus:
    if thread.chat_status == CHAT_STATUS_RUNNING:
        return "running"
    if thread.chat_status == CHAT_STATUS_QUEUED:
        return "queued"
    if thread.metadata.pending_question is not None:
        return "needs_input"
    content = messages.last_content or ""
    if messages.last_role != "assistant":
        # Idle, yet the turn never answered: it died before a reply or marker.
        return "failed"
    if content.startswith((COPILOT_ERROR_PREFIX, COPILOT_RETRYABLE_ERROR_PREFIX)):
        return "cancelled" if CANCELLED_MESSAGE in content else "failed"
    return "completed"


def _summary(
    thread: ChatSessionInfo,
    messages: ThreadMessages,
    cost_microdollars: int | None,
    files_count: int,
    expert: DelegatedExpertInfo | None,
) -> DelegationSummary:
    brief = brief_of(messages.first_user_content)
    status = status_of(thread, messages)
    finished_at = messages.last_at if status in _TERMINAL else None
    pending = thread.metadata.pending_question
    return DelegationSummary(
        sub_session_id=thread.session_id,
        parent_session_id=thread.metadata.delegated_by_session_id,
        expert=expert,
        title=title_of(brief),
        brief=brief,
        status=status,
        created_at=thread.started_at,
        finished_at=finished_at,
        elapsed_seconds=_elapsed(thread.started_at, finished_at),
        cost_usd=(cost_microdollars or 0) / _MICRODOLLARS,
        files_count=files_count,
        question=pending.text if pending else None,
        question_options=pending.options if pending else [],
        asked_at=pending.asked_at if pending else None,
        delegated_by_expert_id=thread.metadata.delegated_by_expert_id,
    )


async def _proposed(
    user_id: str, expert_id: str | None, parent_session_id: str | None
) -> list[DelegationSummary]:
    held = await delegation_list_db.list_held_handoffs(user_id, parent_session_id)
    rows = [_proposal(h) for h in held]
    return [
        r
        for r in rows
        if r is not None
        and (expert_id is None or (r.expert and r.expert.id == expert_id))
    ]


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
        created_at=held.created_at,
        finished_at=None,
        elapsed_seconds=None,
        cost_usd=None,
        files_count=0,
        question=None,
        question_options=[],
    )


def _counts(rows: list[DelegationSummary], spent_microdollars: int) -> DelegationCounts:
    def count(*statuses: str) -> int:
        return sum(1 for r in rows if r.status in statuses)

    return DelegationCounts(
        working=count("running", "queued"),
        needs_you=count("needs_input", "proposed"),
        completed=count("completed"),
        failed=count("failed"),
        spent_today_usd=round(spent_microdollars / _MICRODOLLARS, 6),
    )


def _elapsed(started: datetime, finished: datetime | None) -> float:
    end = finished or datetime.now(UTC)
    return round(max(0.0, (end - started).total_seconds()), 2)
