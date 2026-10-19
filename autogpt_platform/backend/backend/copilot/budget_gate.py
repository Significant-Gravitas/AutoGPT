"""The allowance check a platform-billed turn passes before it starts.

The chat route checks the allowance before it accepts a message, but much of
what runs never passes through it: scheduled follow-ups and routines, the
AutoPilot block, sub-sessions, approval wakes, chat bots, and the turn an
engine starts by itself to answer messages queued while it ran. So the
executor checks every platform turn it picks up, and the SDK engine checks
before it continues. A turn the allowance can't cover doesn't start: it ends
as a "not started" pause naming the limit and when it refills, recorded in
the chat like any other pause. The one exception is the AutoPilot block's
turn, which the executor runs on the credit balance instead when the balance
can pay for it (see ``backend.copilot.credit_billing``).

ChatGPT and Microsoft 365 turns are paid for by the user's own subscription,
so they never come here.
"""

import logging

from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession, upsert_chat_session
from backend.copilot.pending_messages import peek_pending_count
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    SubscriptionTier,
    check_rate_limit,
    get_global_rate_limits,
)
from backend.copilot.turn_pause import TurnPause, append_pause_marker
from backend.copilot.usage_activation import get_ready_usage_state

logger = logging.getLogger(__name__)

config = ChatConfig()


class AutoContinueGate(BaseModel):
    """Whether a finished turn may go on to the messages queued during it."""

    proceed: bool
    # Set when the allowance refused them: the pause to send in their place.
    pause: TurnPause | None = None


async def refusal_for(user_id: str | None) -> TurnPause | None:
    """The not-started pause for a platform turn starting now, or None to run.

    Raises ``RateLimitUnavailable`` when the allowance can't be read: callers
    fail closed rather than run a turn nobody metered.
    """
    if not user_id:
        return None
    daily, weekly, tier = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    if daily == 0 and weekly == 0:
        # No plan, or the trial is over: nothing to spend, and nothing that
        # refills. Settled here because an ended trial's usage state still
        # reads TRIAL, which check_rate_limit's tier guard would take for a
        # subscription change and report as unavailable -- and which is
        # what says the pause is the trial's rather than a missing plan's.
        state = await get_ready_usage_state(user_id)
        ended_trial = state.tier == SubscriptionTier.TRIAL
        return TurnPause(
            reason="trial" if ended_trial else "weekly",
            resets_at=None,
            checkpoint_saved=False,
            started=False,
        )
    try:
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily,
            weekly_cost_limit=weekly,
            expected_tier=tier,
        )
    except RateLimitExceeded as exc:
        # A trial's allowance doesn't refill; its end is not a reset. Nor
        # does a window that allows nothing (no plan, or the trial is over).
        limit = {"daily": daily, "weekly": weekly}.get(exc.window, 0)
        return TurnPause(
            reason=exc.window,
            resets_at=exc.resets_at if limit > 0 else None,
            checkpoint_saved=False,
            started=False,
        )
    return None


async def record_refusal(session: ChatSession, pause: TurnPause) -> None:
    """Leave the pause in the chat so its card survives a reload.

    Best effort: the live stream carries the pause even if this fails.
    """
    if not append_pause_marker(session, pause):
        return
    try:
        await upsert_chat_session(session)
    except Exception:
        logger.warning(
            f"Could not save the not-started marker in session "
            f"{session.session_id[:12]}",
            exc_info=True,
        )


async def gate_auto_continue(
    session: ChatSession, user_id: str | None, *, platform_billed: bool
) -> AutoContinueGate:
    """Check the allowance before a finished turn answers queued messages.

    Only when something is queued. Refused, or with the allowance unreadable,
    the messages stay queued for the next turn instead of being drained; a
    refusal is also recorded in the chat, as the executor records its own.
    """
    if not platform_billed or not user_id:
        return AutoContinueGate(proceed=True)
    try:
        if not await peek_pending_count(session.session_id):
            return AutoContinueGate(proceed=True)
        pause = await refusal_for(user_id)
    except Exception:
        logger.warning(
            f"Queued messages in session {session.session_id[:12]} wait for "
            "the next turn: the allowance couldn't be checked",
            exc_info=True,
        )
        return AutoContinueGate(proceed=False)
    if pause is None:
        return AutoContinueGate(proceed=True)
    await record_refusal(session, pause)
    return AutoContinueGate(proceed=False, pause=pause)
