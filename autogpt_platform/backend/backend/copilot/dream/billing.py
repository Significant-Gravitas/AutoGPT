"""Dream-pass billing — pre-flight check + per-phase cost row.

Dream-pass spend rolls into the user's existing daily/weekly USD budget
(``backend.copilot.rate_limit``) — there is no dedicated dream-pass
counter. This module is the seam between the dream and the shared billing
primitives:

* :func:`check_dream_budget` — pre-flight gate. Called once after the
  Redis lock is acquired, before phase 1 runs. Refuses the pass when
  the user is paywalled (``NO_TIER`` + ``ENABLE_PLATFORM_PAYMENT``) or
  has already exhausted their daily/weekly cap.
* :func:`record_phase_cost` — per-phase charge, through the one record
  every background call uses (``backend/copilot/inference/record.py``).
  Called after each of consolidate / recombine / sanitize completes, on
  the sync path and from the batch callbacks alike. The record prices a
  phase the provider did not price from the catalog price card, at the
  path's batch discount, and labels the row with the route's provider
  (so cost dashboards don't have to back-correlate dream rows to
  OpenRouter vs Anthropic vs a local backend).

Per-LLM-call rows (not one row per pass) match the chat convention so
the existing per-block / per-provider rollups in the admin dashboard
keep working without dream-specific code paths.

Failure semantics mirror the chat path: a partial pass still charges
for the phases that completed before the error, because we already
paid the provider for those tokens.
"""

from __future__ import annotations

import logging
from typing import Literal

from backend.copilot.config import ChatConfig
from backend.copilot.inference.context import InferenceContext, InferenceUsage
from backend.copilot.inference.record import record
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    get_global_rate_limits,
    is_user_paywalled,
)

logger = logging.getLogger(__name__)


DreamBudgetSkipReason = Literal[
    "insufficient_credits",
    "rate_limit_unavailable",
]


async def check_dream_budget(
    user_id: str, config: ChatConfig | None = None
) -> tuple[bool, DreamBudgetSkipReason | None]:
    """Pre-flight: is the user allowed to spend on a dream pass right now?

    Returns ``(True, None)`` when the pass may proceed. Returns
    ``(False, reason)`` when the orchestrator should bail with a
    skipped result. Background-callable — never raises.

    Reasons:
        ``"insufficient_credits"``  — user is paywalled (NO_TIER +
            ``ENABLE_PLATFORM_PAYMENT`` on) OR already over their
            daily/weekly cap. Treated as a soft skip so the next
            scheduler tick retries naturally.
        ``"rate_limit_unavailable"`` — Redis is unreadable. Fail
            closed: a brown-out must not let dream passes bypass the
            user's USD cap. Surfaces as ``error`` on the result so the
            admin endpoint reports it; the scheduler retries next tick.
    """
    config = config or ChatConfig()

    try:
        paywalled = await is_user_paywalled(user_id)
    except Exception as exc:
        # Tier lookup failure during a background job — fail closed so
        # we don't run a paywalled user's dream pass on a transient
        # Supabase blip. Scheduler will retry next tick.
        logger.warning(
            "dream billing: paywall check failed for user=%s: %s", user_id[:8], exc
        )
        return False, "rate_limit_unavailable"
    if paywalled:
        logger.info("dream billing: skipping paywalled user=%s", user_id[:8])
        return False, "insufficient_credits"

    try:
        daily_limit, weekly_limit, _tier = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
            skip_daily=True,
        )
    except RateLimitExceeded as exc:
        logger.info(
            "dream billing: user=%s over %s cap, skipping", user_id[:8], exc.window
        )
        return False, "insufficient_credits"
    except RateLimitUnavailable:
        logger.warning(
            "dream billing: rate-limit state unreadable for user=%s; failing closed",
            user_id[:8],
        )
        return False, "rate_limit_unavailable"

    return True, None


async def record_phase_cost(
    ctx: InferenceContext, usage: InferenceUsage
) -> InferenceUsage:
    """Charge one phase's spend against the user's window and log its row.

    A thin adapter on ``inference.record``. The row keeps the dream's
    shape: ``block_name=copilot:dream:<phase>`` so the per-block rollup
    separates dream spend from chat spend, the pass id as
    ``graph_exec_id`` so a pass's three rows join up, and the
    ``dream_pass_id`` / ``dream_phase`` metadata keys dashboards read. The
    record adds the provider label, the execution path, the discount and
    the expert.

    Returns *usage* as priced. No row and no charge for a phase with
    neither tokens nor a cost; tokens without a cost (a model with no
    catalog price) still log but don't charge the rate-limit counter.
    """
    return await record(
        ctx,
        usage,
        block_name=f"copilot:dream:{ctx.job.phase}",
        metadata={
            "dream_pass_id": ctx.job.correlation_id,
            "dream_phase": ctx.job.phase,
        },
    )
