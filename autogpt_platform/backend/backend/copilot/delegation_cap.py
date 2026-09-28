"""The question a delegated thread stops on when it reaches its cap.

With ``ask_before_over_cap`` on, a thread that spends its per-delegation cap
is stopped and parked on a question to the user instead of failing: raise the
cap by a fixed step and carry on, or stop there. The answer arrives through
``answer_session`` like any other; this module recognises it and applies it.
"""

from datetime import UTC, datetime

from pydantic import BaseModel

from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import PendingQuestion
from backend.data.db_accessors import chat_db, delegation_db

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


async def apply_cap_answer(
    session_id: str, user_id: str, answer: CapAnswer
) -> str | None:
    """Raise the thread's cap and return the message that resumes it, or stop
    the thread and return None (nothing to send)."""
    if answer.raise_usd is None:
        await delegation_db().stop_delegation_at_cap(session_id, user_id)
        await enqueue_cancel_task(session_id)
        return None
    cap = await delegation_db().raise_delegation_cap(
        session_id, user_id, answer.raise_usd
    )
    return (
        f"[The user raised this hand-off's budget by ${answer.raise_usd:.2f} to "
        f"${cap:.2f}. Carry on with the task where you left off.]"
    )
