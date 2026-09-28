"""The user's delegation settings, applied to one hand-off.

Two limits, set on Otto's Settings tab: a daily budget across every
delegation (checked before a new hand-off starts) and a per-delegation cap
(stored on the delegated thread, and checked whenever its status is read,
since the tree ledger has no per-child ceiling to hand it). The settings'
mode is the approval mode a thread Otto opens starts in.
"""

import logging
from datetime import UTC, datetime

from pydantic import BaseModel

from backend.copilot.delegation_settings import DelegationSettings
from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import AutopilotMode, ChatSession
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

    mode: AutopilotMode | None
    cap_usd: float


async def delegation_terms(user_id: str, session: ChatSession) -> DelegationTerms | str:
    """The new thread's mode and cap, or why the hand-off may not start.

    Fails closed: a budget that cannot be read is not a budget with room.
    """
    try:
        settings = await delegation_db().get_delegation_settings(user_id)
        spent = await delegation_db().get_delegation_spend_since(
            user_id, _start_of_day()
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
    return DelegationTerms(mode=mode, cap_usd=settings.per_delegation_cap_usd)


async def enforce_cap(
    response: SubSessionStatusResponse, cap_usd: float | None, actor: str
) -> SubSessionStatusResponse:
    """Stop a still-working thread that has spent its cap, and say so.

    A finished thread keeps its result: there is nothing left to stop, and
    discarding work already paid for helps nobody.
    """
    if (
        cap_usd is None
        or response.status not in ("running", "queued")
        or response.cost_usd is None
        or response.cost_usd < cap_usd
    ):
        return response
    await enqueue_cancel_task(response.sub_session_id)
    return response.model_copy(
        update={
            "status": "error",
            "error": CAP_REACHED,
            "message": (
                f"{actor} reached the ${cap_usd:.2f} per-delegation cap after "
                f"${response.cost_usd:.2f} and was stopped. Tell the user; they "
                "can raise the cap in Otto's settings and ask again."
            ),
        }
    )


def _start_of_day() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _budget_spent(spent: int, settings: DelegationSettings) -> str:
    return (
        f"Today's delegation budget is used up (${spent / MICRODOLLARS_PER_USD:.2f} "
        f"of ${settings.daily_budget_usd:.2f}), so nothing was handed off. Do the "
        "work yourself or tell the user; they can raise the daily budget in "
        "Otto's settings."
    )
