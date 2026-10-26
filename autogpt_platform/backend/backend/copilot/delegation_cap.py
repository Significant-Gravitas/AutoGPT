"""The question a delegated thread stops on when it reaches its cap.

With ``ask_before_over_cap`` on, a thread that spends its per-delegation cap
is stopped and parked on a question to the user instead of failing: raise the
cap by a fixed step and carry on, or stop there. The answer arrives through
``answer_session`` like any other; this module recognises it and applies it.

Every write here goes to the thread's row alone and then drops the cached
session, so the next turn and every reader see it: the turn that landed on
the cap cached its own copy just before.
"""

import logging
from datetime import UTC, datetime

from pydantic import BaseModel

from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import PendingQuestion, invalidate_session_cache
from backend.data.db_accessors import chat_db, delegation_db

logger = logging.getLogger(__name__)

CAP_STOP = "Stop"
_RAISES: dict[str, float] = {"Raise by $1": 1.0, "Raise by $5": 5.0}
CAP_OPTIONS: list[str] = [*_RAISES, CAP_STOP]
_QUESTION_PREFIX = "This hand-off has reached its $"


class CapAnswer(BaseModel):
    """The user's answer to the cap question; ``raise_usd`` None means stop."""

    raise_usd: float | None


def cap_question(cap_usd: float) -> str:
    return f"{_QUESTION_PREFIX}{cap_usd:.2f} cap. Raise it and continue?"


def is_cap_question(pending: PendingQuestion | None) -> bool:
    return (
        pending is not None
        and pending.text.startswith(_QUESTION_PREFIX)
        and pending.options == CAP_OPTIONS
    )


def parse_cap_answer(pending: PendingQuestion | None, message: str) -> CapAnswer | None:
    """The answer, if *message* picks one of the cap question's options."""
    if not is_cap_question(pending):
        return None
    choice = message.strip()
    if choice in _RAISES:
        return CapAnswer(raise_usd=_RAISES[choice])
    if choice.lower() == CAP_STOP.lower():
        return CapAnswer(raise_usd=None)
    return None


async def park_at_cap(session_id: str, user_id: str, cap_usd: float) -> None:
    await chat_db().set_session_pending_question(
        session_id,
        user_id,
        cap_question(cap_usd),
        datetime.now(UTC),
        options=CAP_OPTIONS,
    )
    await invalidate_session_cache(session_id)


async def hold_at_cap(session_id: str, user_id: str, cap_usd: float) -> bool:
    """Hold a thread that reached its cap for the user's word: parked on the
    raise-or-stop question while ``ask_before_over_cap`` is on, else stopped
    there. True when it parked. Cancelling a turn still running is the
    caller's: a turn that landed on the cap has already ended.
    """
    if await asks_over_cap(user_id):
        await park_at_cap(session_id, user_id, cap_usd)
        return True
    await delegation_db().stop_delegation_at_cap(session_id, user_id)
    await invalidate_session_cache(session_id)
    return False


async def asks_over_cap(user_id: str) -> bool:
    """Unreadable settings ask: a question costs less than a lost thread."""
    try:
        settings = await delegation_db().get_delegation_settings(user_id)
    except Exception:
        logger.warning(f"Delegation settings unreadable for {user_id}", exc_info=True)
        return True
    return settings.ask_before_over_cap


async def stop_at_cap(session_id: str, user_id: str) -> None:
    """The user chose "Stop": record it, and make sure nothing still runs."""
    await delegation_db().stop_delegation_at_cap(session_id, user_id)
    await invalidate_session_cache(session_id)
    await enqueue_cancel_task(session_id)


async def raise_cap(
    session_id: str, user_id: str, by_usd: float, question: PendingQuestion
) -> str:
    """Lift the thread's cap once per question; the message that resumes it.

    The raise is keyed to the question it answers, so a retried answer (the
    turn could not start, the client sent it again) raises nothing twice.
    """
    cap = await delegation_db().raise_delegation_cap(
        session_id, user_id, by_usd, question.asked_at.isoformat()
    )
    # Before the turn it resumes starts: that turn reads the session from the
    # cache, where the turn that landed on the cap left the old cap.
    await invalidate_session_cache(session_id)
    return (
        f"[The user raised this hand-off's budget by ${by_usd:.2f} to "
        f"${cap:.2f}. Carry on with the task where you left off.]"
    )
