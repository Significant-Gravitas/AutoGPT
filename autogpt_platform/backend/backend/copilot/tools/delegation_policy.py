"""The user's delegation settings, applied to one hand-off.

Two limits, set on Otto's Settings tab: a daily budget across every
delegation (checked before a new hand-off starts) and a per-delegation cap
(stored on the delegated thread, and checked whenever its status is read,
since the tree ledger has no per-child ceiling to hand it). The settings'
mode is the approval mode a thread Otto opens starts in, and Ask first for a
teammate hired this week when ``new_experts_ask_first`` is on.
"""

import logging
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from backend.copilot.delegation_cap import (
    CAP_OPTIONS,
    cap_question,
    is_cap_question,
    park_at_cap,
)
from backend.copilot.delegation_settings import DelegationSettings, start_of_utc_day
from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import AutopilotMode, ChatSession, ChatSessionMetadata
from backend.data.db_accessors import delegation_db

from .models import SubSessionStatusResponse
from .sub_session_facts import MICRODOLLARS_PER_USD

logger = logging.getLogger(__name__)

CAP_REACHED = "cap reached"
# How long a new hire's hand-offs start in Ask first, when the user asks for it.
NEW_HIRE_WINDOW = timedelta(days=7)
_UNCHECKED = (
    "Could not check today's delegation budget, so nothing was handed off. "
    "Try again in a moment."
)


class DelegationTerms(BaseModel):
    """What a new delegated thread starts with."""

    mode: AutopilotMode | None
    cap_usd: float


async def delegation_terms(
    user_id: str, session: ChatSession, target_id: str
) -> DelegationTerms | str:
    """The new thread's mode and cap, or why the hand-off may not start.

    Fails closed: a budget that cannot be read is not a budget with room.
    """
    try:
        settings = await delegation_db().get_delegation_settings(user_id)
        spent = await delegation_db().get_delegation_spend_since(
            user_id, start_of_utc_day()
        )
        new_hire = settings.new_experts_ask_first and await _hired_lately(
            user_id, target_id
        )
    except Exception:
        logger.warning(f"Delegation budget unreadable for {user_id}", exc_info=True)
        return _UNCHECKED
    if spent >= settings.daily_budget_usd * MICRODOLLARS_PER_USD:
        return _budget_spent(spent, settings)
    # Otto's hand-offs follow the Settings tab; an expert's keep its own mode.
    mode = (
        settings.mode if session.expert_id is None else session.metadata.autopilot_mode
    )
    return DelegationTerms(
        mode="ask_first" if new_hire else mode,
        cap_usd=settings.per_delegation_cap_usd,
    )


async def _hired_lately(user_id: str, expert_id: str) -> bool:
    """Hired less than a week ago. An unknown hire date counts as new."""
    hired_at = await delegation_db().get_expert_hired_at(user_id, expert_id)
    return hired_at is None or datetime.now(UTC) - hired_at < NEW_HIRE_WINDOW


class CapState(BaseModel):
    """A delegated thread's cap, as read fresh from its row."""

    cap_usd: float | None
    # The user answered the cap question with "Stop".
    stopped: bool = False
    # The thread is already parked on the cap question.
    parked: bool = False


def cap_state(meta: ChatSessionMetadata) -> CapState:
    return CapState(
        cap_usd=meta.delegation_cap_usd,
        stopped=meta.delegation_cap_stopped,
        parked=is_cap_question(meta.pending_question),
    )


async def enforce_cap(
    response: SubSessionStatusResponse, state: CapState, actor: str, user_id: str
) -> SubSessionStatusResponse:
    """Stop a still-working thread that has spent its cap, and say so.

    With ``ask_before_over_cap`` on, the thread is parked on a question (raise
    the cap, or stop) and reads ``needs_input`` until the user answers; off,
    or once they answered "Stop", it reads ``error`` / "cap reached". A
    thread that finished within its run keeps its result: there is nothing
    left to stop, and discarding work already paid for helps nobody.
    """
    cap = state.cap_usd
    if cap is None or response.cost_usd is None or response.cost_usd < cap:
        return response
    working = response.status in ("running", "queued")
    if not (working or state.parked or state.stopped):
        return response
    if working:
        await enqueue_cancel_task(response.sub_session_id)
    if not state.stopped and await _asks_over_cap(user_id):
        if not state.parked:
            await park_at_cap(response.sub_session_id, user_id, cap)
        return _awaiting_raise(response, cap, actor)
    return _cap_reached(response, cap, actor)


async def _asks_over_cap(user_id: str) -> bool:
    """Unreadable settings ask: a question costs less than a lost thread."""
    try:
        settings = await delegation_db().get_delegation_settings(user_id)
    except Exception:
        logger.warning(f"Delegation settings unreadable for {user_id}", exc_info=True)
        return True
    return settings.ask_before_over_cap


def _awaiting_raise(
    response: SubSessionStatusResponse, cap: float, actor: str
) -> SubSessionStatusResponse:
    return response.model_copy(
        update={
            "status": "needs_input",
            "question": cap_question(cap),
            "question_options": list(CAP_OPTIONS),
            "message": (
                f"{actor} reached the ${cap:.2f} per-delegation cap and is paused "
                "until the user raises it or stops the task. Tell the user and "
                "wait; do not answer it yourself."
            ),
        }
    )


def _cap_reached(
    response: SubSessionStatusResponse, cap: float, actor: str
) -> SubSessionStatusResponse:
    spent = response.cost_usd or 0.0
    return response.model_copy(
        update={
            "status": "error",
            "error": CAP_REACHED,
            "message": (
                f"{actor} reached the ${cap:.2f} per-delegation cap after "
                f"${spent:.2f} and was stopped. Tell the user; they can raise the "
                "cap in Otto's settings and ask again."
            ),
        }
    )


def _budget_spent(spent: int, settings: DelegationSettings) -> str:
    return (
        f"Today's delegation budget is used up (${spent / MICRODOLLARS_PER_USD:.2f} "
        f"of ${settings.daily_budget_usd:.2f}), so nothing was handed off. Do the "
        "work yourself or tell the user; they can raise the daily budget in "
        "Otto's settings."
    )
