"""Facts a child-session status reports beyond its outcome: cost and timing.

Kept apart from ``run_sub_session`` so the spawn tools and the poll tool read
them the same way, and so the outcome mapping there stays a pure translation.
"""

import logging
from datetime import UTC, datetime

from pydantic import BaseModel

from backend.copilot.model import ChatSession
from backend.copilot.sdk.session_waiter import SessionOutcome
from backend.data.db_accessors import delegation_db

logger = logging.getLogger(__name__)

MICRODOLLARS_PER_USD = 1_000_000

# Outcomes after which the child's turn is over: ``finished_at`` is set.
TERMINAL_OUTCOMES: frozenset[SessionOutcome] = frozenset(
    {"completed", "failed", "refused", "rejected_concurrent_turn_cap"}
)


class RunFacts(BaseModel):
    cost_usd: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


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
