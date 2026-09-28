"""Facts a child-session status reports beyond its outcome: cost, timing, and
the question it stopped on.

Kept apart from ``run_sub_session`` so the spawn tools and the poll tool read
them the same way, and so the outcome mapping there stays a pure translation.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.copilot.model import ChatSession, PendingQuestion
from backend.copilot.sdk.session_waiter import SessionOutcome
from backend.copilot.sdk.stream_accumulator import ToolCallEntry
from backend.data.db_accessors import delegation_db

logger = logging.getLogger(__name__)

MICRODOLLARS_PER_USD = 1_000_000
ASK_QUESTION_TOOL = "ask_question"

# Outcomes after which the child's turn is over: ``finished_at`` is set.
TERMINAL_OUTCOMES: frozenset[SessionOutcome] = frozenset(
    {"completed", "failed", "refused", "rejected_concurrent_turn_cap"}
)


class RunFacts(BaseModel):
    cost_usd: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class PendingAsk(BaseModel):
    """What the child asked the user; ``options`` are the first question's."""

    question: str
    options: list[str] = []


class _AskItem(BaseModel):
    question: str = ""
    options: list[str] = []


class _AskArgs(BaseModel):
    questions: list[_AskItem] = []


async def sub_session_cost_usd(user_id: str, session_id: str) -> float | None:
    """Logged spend of the child session so far, or None when unreadable."""
    try:
        costs = await delegation_db().get_session_costs(user_id, [session_id])
    except Exception:
        logger.warning(
            f"Could not read cost of sub-session {session_id}", exc_info=True
        )
        return None
    return round(costs.get(session_id, 0) / MICRODOLLARS_PER_USD, 6)


async def run_facts(
    user_id: str,
    session_id: str,
    outcome: SessionOutcome,
    started_at: datetime | None,
) -> RunFacts:
    """Cost so far, plus the turn's end when *outcome* says it is over."""
    return RunFacts(
        cost_usd=await sub_session_cost_usd(user_id, session_id),
        started_at=started_at,
        finished_at=datetime.now(UTC) if outcome in TERMINAL_OUTCOMES else None,
    )


def turn_started_at(sub: ChatSession) -> datetime | None:
    """When the child's latest turn began: its last user message."""
    return next(
        (m.created_at for m in reversed(sub.messages) if m.role == "user"),
        None,
    )


def turn_finished_at(sub: ChatSession) -> datetime | None:
    """When the child's persisted turn ended: its last message."""
    return sub.messages[-1].created_at if sub.messages else None


def ask_from_tool_calls(tool_calls: list[ToolCallEntry]) -> PendingAsk | None:
    """The question a turn ended on, when its last tool call was ``ask_question``."""
    if not tool_calls:
        return None
    last = tool_calls[-1]
    if last.tool_name != ASK_QUESTION_TOOL or last.success is False:
        return None
    items = [q for q in _ask_args(last.input).questions if q.question.strip()]
    if not items:
        return None
    return PendingAsk(
        question="; ".join(q.question.strip() for q in items),
        options=items[0].options,
    )


def ask_from_pending(pending: PendingQuestion | None) -> PendingAsk | None:
    """The question parked on the child session, if it is still unanswered."""
    if pending is None or not pending.text.strip():
        return None
    return PendingAsk(question=pending.text, options=pending.options)


def _ask_args(raw: Any) -> _AskArgs:
    """Tool input arrives as a dict live and as a JSON string when replayed."""
    try:
        if isinstance(raw, str):
            return _AskArgs.model_validate(json.loads(raw))
        return _AskArgs.model_validate(raw or {})
    except (ValidationError, ValueError):
        return _AskArgs()
