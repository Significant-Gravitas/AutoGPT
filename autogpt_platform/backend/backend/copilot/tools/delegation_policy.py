"""The user's delegation settings, applied to one hand-off.

Two limits, set on Otto's Settings tab: a daily budget across every
delegation (checked before a new hand-off starts) and a per-delegation cap,
stored on the delegated thread. Each engine stops the thread's turn at what
the cap has left and the executor holds the thread there
(``delegation_cap.hold_at_cap``); every read of the thread reports that
hold, and stops a turn still running past the cap as a backstop. The settings'
mode tightens the approval mode a thread Otto opens starts in, never looser
than what the delegating chat's gate enforces, and Ask first for a teammate
hired this week when ``new_experts_ask_first`` is on.
"""

import logging

from pydantic import BaseModel

from backend.api.features.experts.models import Expert
from backend.copilot.delegation_cap import (
    CAP_OPTIONS,
    cap_question,
    hold_at_cap,
    is_cap_question,
)
from backend.copilot.delegation_settings import DelegationSettings, start_of_utc_day
from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.gate import child_autopilot_mode
from backend.copilot.gate.delegation_rules import hired_lately
from backend.copilot.gate.policy import strictest_mode
from backend.copilot.model import (
    AutopilotMode,
    ChatSession,
    ChatSessionMetadata,
    get_chat_session_metadata,
)
from backend.data.db_accessors import delegation_db

from .models import SubSessionStatusResponse
from .sub_session_facts import MICRODOLLARS_PER_USD

logger = logging.getLogger(__name__)

CAP_REACHED = "cap reached"
_UNCHECKED = (
    "Could not check today's delegation budget, so nothing was handed off. "
    "Try again in a moment."
)


class DelegationTerms(BaseModel):
    """What a new delegated thread starts with."""

    mode: AutopilotMode
    cap_usd: float


async def delegation_terms(
    user_id: str, session: ChatSession, target: Expert
) -> DelegationTerms | str:
    """The new thread's mode and cap, or why the hand-off may not start.

    Fails closed: a budget that cannot be read is not a budget with room.
    """
    try:
        settings = await delegation_db().get_delegation_settings(user_id)
        spent = await delegation_db().get_delegation_spend_since(
            user_id, start_of_utc_day()
        )
        new_hire = settings.new_experts_ask_first and await hired_lately(
            user_id, target.id
        )
    except Exception:
        logger.warning(f"Delegation budget unreadable for {user_id}", exc_info=True)
        return _UNCHECKED
    if spent >= settings.daily_budget_usd * MICRODOLLARS_PER_USD:
        return _budget_spent(spent, settings)
    return DelegationTerms(
        mode=_child_mode(session, target, settings, new_hire),
        cap_usd=settings.per_delegation_cap_usd,
    )


def _child_mode(
    session: ChatSession,
    target: Expert,
    settings: DelegationSettings,
    new_hire: bool,
) -> AutopilotMode:
    """The strictest of what the delegating chat's gate enforces (tightened by
    the expert's own default), Otto's Settings tab for Otto's hand-offs, and
    Ask first for a new hire. Each can only tighten: a setting never lets a
    thread run looser than the chat that opened it."""
    modes: list[AutopilotMode] = [child_autopilot_mode(session, target.autopilot_mode)]
    if session.expert_id is None:
        modes.append(settings.mode)
    if new_hire:
        modes.append("ask_first")
    return strictest_mode(*modes)


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


async def read_cap_state(session_id: str, known: CapState) -> CapState:
    """The thread's cap as its row holds it now.

    Read past the session cache: a raise, a parked cap question or a stop is
    written to the row alone, after the turn that cached the session.
    ``known`` stands in when the row can't be found.
    """
    fresh = await get_chat_session_metadata(session_id)
    return cap_state(fresh.metadata) if fresh else known


async def enforce_cap(
    response: SubSessionStatusResponse, state: CapState, actor: str, user_id: str
) -> SubSessionStatusResponse:
    """Apply the thread's cap to what a read of it reports.

    A thread held at its cap (its turn landed there, or a read stopped it)
    reads so whatever else the read saw: ``needs_input`` on the raise-or-stop
    question while it is parked, ``error`` / "cap reached" once stopped. The
    cost log runs behind the turn, so the hold, not the logged spend, says
    the cap was reached.

    A thread still working past its cap is stopped here too, a backstop for
    a turn that ran on past it. A thread that finished within its run keeps
    its result: there is nothing left to stop, and discarding work already
    paid for helps nobody.
    """
    cap = state.cap_usd
    if cap is None:
        return response
    # A question parked after a stop is the newer state: the thread landed on
    # its cap again (the user wrote in it) and asks once more.
    if state.parked:
        return _awaiting_raise(response, cap, actor)
    if state.stopped:
        return _cap_reached(response, cap, actor)
    if response.cost_usd is None or response.cost_usd < cap:
        return response
    if response.status not in ("running", "queued"):
        return response
    # The same durable hold a turn that lands on its cap gets, so later reads
    # see the cap, not an ordinary cancellation.
    await enqueue_cancel_task(response.sub_session_id)
    if await hold_at_cap(response.sub_session_id, user_id, cap):
        return _awaiting_raise(response, cap, actor)
    return _cap_reached(response, cap, actor)


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
