"""Which billing clock an account's usage limits run on, and its windows now.

Every account has exactly one clock:

- an active **trial**: the trial's own start and end;
- a **Stripe-billed** plan (BASIC, PRO, MAX, BUSINESS on an account with a
  Stripe customer) with a stored billing period: that period;
- everything else with an allowance (ENTERPRISE, a plan an admin granted
  without Stripe, an install with payment off): monthly on the day and time
  the **account** was created, which every account has.

Stripe is the source of truth for a billing period and our copy can be late.
When the copy has ended, or a Stripe-billed plan has none, it is re-read from
Stripe, at most once a minute per user. Until Stripe confirms the next period
the last weekly window is **held** open: no new weekly allowance on a guess,
and no paying customer blocked because our copy is stale (they keep what the
window had left, and each day's reserve).
"""

import asyncio
import logging
from collections import OrderedDict
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.copilot.trial_reads import forget_usage_state
from backend.copilot.usage_activation import get_ready_usage_state
from backend.copilot.usage_windows import (
    UsageWindows,
    windows_for_period,
    windows_from_anchor,
)
from backend.data.db_accessors import usage_clock_db
from backend.data.pro_activation import UsageActivationState
from backend.data.redis_client import get_redis_async
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)

ClockSource = Literal["stripe", "held", "trial", "account"]

STRIPE_BILLED_TIERS = frozenset({"BASIC", "PRO", "MAX", "BUSINESS"})

REFRESH_GATE_PREFIX = "usage:clock_refresh"
# One Stripe read per user per minute while a period is unconfirmed.
REFRESH_RETRY_SECONDS = 60
# How long a turn's limit check waits for that read: with Stripe slow or down
# it gives up and holds the last window rather than stall the turn.
REFRESH_TIMEOUT_SECONDS = 8.0
# Stripe said there is no live subscription: nothing to re-read until the
# periodic tier sweep has had its turn.
NO_SUBSCRIPTION_RETRY_SECONDS = 6 * 3600

# The last usage state seen per user, for recording a cost without a database
# read: used for a minute at most and never past its billing period's end.
RECENT_STATE_SECONDS = 60
RECENT_STATE_MAX_USERS = 10_000


class ResolvedClock(BaseModel):
    model_config = ConfigDict(frozen=True)

    windows: UsageWindows
    source: ClockSource


def clock_from_state(state: UsageActivationState, now: int) -> ResolvedClock | None:
    """The account's windows from what the usage state holds; None when the
    state carries no clock at all (a database manager on the previous release)."""
    if state.tier.value == "TRIAL" and _has_period(
        state.trial_started_at, state.trial_ends_at
    ):
        trial = windows_for_period(
            state.trial_started_at or 0, state.trial_ends_at or 0, None, now
        )
        return ResolvedClock(windows=trial, source="trial")
    if is_stripe_billed(state) and _has_period(state.period_start, state.period_end):
        stripe = windows_for_period(
            state.period_start or 0, state.period_end or 0, state.billing_anchor, now
        )
        return ResolvedClock(windows=stripe, source="held" if stripe.held else "stripe")
    if state.created_at is None:
        return None
    return ResolvedClock(
        windows=windows_from_anchor(state.created_at, now), source="account"
    )


async def resolve_clock(
    state: UsageActivationState, now: int
) -> tuple[UsageActivationState, ResolvedClock | None]:
    """The account's clock before a turn, re-reading Stripe when our copy of a
    Stripe-billed plan's period has ended or is missing. Returns the state the
    clock was read from, which is a fresh one after a re-read."""
    # "busy": another check is re-reading Stripe right now, or just did; its
    # answer may already be saved, so the state is read again either way.
    if needs_refresh(state, now) and await _refresh_once(state, now) != "unconfirmed":
        forget_usage_state(state.user_id)
        state = await get_ready_usage_state(state.user_id)
    remember_state(state)
    return state, clock_from_state(state, now)


def is_stripe_billed(state: UsageActivationState) -> bool:
    return state.has_stripe_customer and state.tier.value in STRIPE_BILLED_TIERS


def needs_refresh(state: UsageActivationState, now: int) -> bool:
    if not is_stripe_billed(state):
        return False
    if not _has_period(state.period_start, state.period_end):
        return True
    return now >= (state.period_end or 0)


def remember_state(state: UsageActivationState) -> None:
    _recent_states[state.user_id] = (monotonic(), state)
    _recent_states.move_to_end(state.user_id)
    while len(_recent_states) > RECENT_STATE_MAX_USERS:
        _recent_states.popitem(last=False)


def recent_state(user_id: str, now: int) -> UsageActivationState | None:
    """The state last seen for the user, while it is still safe to record a
    cost against: under a minute old and not past its billing period."""
    seen = _recent_states.get(user_id)
    if seen is None:
        return None
    seen_at, state = seen
    expired = monotonic() - seen_at > RECENT_STATE_SECONDS
    period_over = state.period_end is not None and now >= state.period_end
    if expired or period_over:
        _recent_states.pop(user_id, None)
        return None
    return state


def forget_recent_states() -> None:
    _recent_states.clear()


_recent_states: OrderedDict[str, tuple[float, UsageActivationState]] = OrderedDict()


async def _refresh_once(
    state: UsageActivationState, now: int
) -> Literal["saved", "busy", "unconfirmed"]:
    """Re-read the period from Stripe unless that was tried in the last
    minute ("busy"). "saved" when a newer period was stored. Never raises:
    without an answer the caller holds the last window (or uses the account's
    clock)."""
    user_id = state.user_id
    gate = f"{REFRESH_GATE_PREFIX}:{user_id}"
    try:
        redis = await get_redis_async()
        if not await redis.set(gate, "1", nx=True, ex=REFRESH_RETRY_SECONDS):
            return "busy"
    except Exception:
        # No gate, no Stripe read: the check that follows fails closed on the
        # same Redis outage anyway.
        return "unconfirmed"
    try:
        outcome = await asyncio.wait_for(
            usage_clock_db().refresh_billing_clock(user_id),
            timeout=REFRESH_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning(
            f"Could not re-read the billing period of user {user_id} from Stripe",
            exc_info=True,
        )
        outcome = "stripe_error"
    logger.info(f"Billing clock re-read for user {user_id}: {outcome}")
    if outcome == "saved":
        return "saved"
    if outcome in ("no_subscription", "not_stripe_billed"):
        try:
            await redis.expire(gate, NO_SUBSCRIPTION_RETRY_SECONDS)
        except Exception:
            logger.warning("Could not extend the billing clock re-read gate")
    _track_unconfirmed(state, now, outcome)
    return "unconfirmed"


def _track_unconfirmed(state: UsageActivationState, now: int, outcome: str) -> None:
    """One event per re-read that didn't confirm a period (so at most one a
    minute per user, and one per six hours once Stripe says there is none)."""
    missing = not _has_period(state.period_start, state.period_end)
    posthog_client.capture(
        state.user_id,
        PostHogEvent.USAGE_CLOCK_MISSING if missing else PostHogEvent.USAGE_CLOCK_HELD,
        {
            "tier": state.tier.value,
            "outcome": outcome,
            "seconds_past_period_end": (
                None if missing else max(0, now - (state.period_end or now))
            ),
        },
    )


def _has_period(start: int | None, end: int | None) -> bool:
    return start is not None and end is not None and end > start
