"""Dream-pass billing — pre-flight check + per-step cost log.

Dream-pass spend rolls into the user's existing daily/weekly USD budget
(``backend.copilot.rate_limit``) — there is no dedicated dream-pass
counter. This module is the seam between the orchestrator and the
shared billing primitives:

* :func:`check_dream_budget` — pre-flight gate. Called once after the
  Redis lock is acquired, before phase 1 runs. Refuses the pass when
  the user is paywalled (``NO_TIER`` + ``ENABLE_PLATFORM_PAYMENT``) or
  has already exhausted their daily/weekly cap.
* :func:`record_phase_cost` — per-phase charge. Called after each of
  consolidate / recombine / sanitize completes, charging the real LLM
  spend against the user's window and writing a ``PlatformCostLog``
  row with ``provider`` set to the actual LLM provider (so downstream
  cost dashboards don't have to back-correlate dream rows to OpenRouter
  vs Anthropic vs OpenAI).

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
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    get_global_rate_limits,
    is_user_paywalled,
)
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.transport_routing import routing_kwargs_for_chat_transport

from .routing import ExecutionPath
from .schemas import PhaseUsage

logger = logging.getLogger(__name__)


# Provider-locked batch paths — these dispatch through a specific
# provider's batch API regardless of which chat transport is active,
# so the cost-log label is fixed at the table level. The
# ``sync_baseline`` path is NOT in this table because its provider
# follows the active ``ChatConfig.transport`` (see
# :func:`_provider_for_execution_path` below).
_PROVIDER_BY_BATCH_PATH: dict[ExecutionPath, str] = {
    "anthropic_batch": "anthropic",
    "openai_batch": "openai",
}


def _provider_for_execution_path(execution_path: ExecutionPath) -> str:
    """Resolve the ``PlatformCostLog.provider`` label for a phase row.

    Batch paths are pinned to their provider (Anthropic / OpenAI). The
    sync_baseline path follows the active chat transport — so a
    local-Ollama install logs ``provider="ollama"``, a subscription
    or direct-Anthropic install logs ``"anthropic"``, and the cloud
    OpenRouter default logs ``"open_router"``. Centralizes what was
    a static ``"sync_baseline" → "open_router"`` mapping that
    misattributed local + subscription dream rows as OR spend on the
    admin dashboard.
    """
    if execution_path in _PROVIDER_BY_BATCH_PATH:
        return _PROVIDER_BY_BATCH_PATH[execution_path]
    return routing_kwargs_for_chat_transport().cost_log_provider


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
    *,
    user_id: str,
    pass_id: str,
    phase_usage: PhaseUsage,
    execution_path: ExecutionPath,
) -> None:
    """Charge one phase's spend against the user's window + log a row.

    Writes a ``PlatformCostLog`` row tagged ``provider=<llm_provider>``
    (not ``provider="dream_pass"``) so existing per-provider dashboards
    aggregate naturally. The dream-specific shape is carried in
    ``metadata`` (dream_pass_id, dream_phase, execution_path,
    discount_applied) and ``block_name=copilot:dream:<phase>`` so the
    per-block rollup separates dream spend from chat spend.

    ``graph_exec_id`` is set to the ``pass_id`` so each phase row joins
    back to its parent dream pass — the same join key the future P9
    inline ``dream.operations`` SSE event will reference.

    No-ops when the phase has no cost (skipped phase / unknown rate
    card). Matches ``persist_and_record_usage``'s contract: tokens
    without cost still log but don't charge the rate-limit counter.
    """
    if (
        phase_usage.cost_usd is None
        and (
            phase_usage.input_tokens
            + phase_usage.output_tokens
            + phase_usage.cache_read_tokens
            + phase_usage.cache_creation_tokens
        )
        == 0
    ):
        return

    provider = _provider_for_execution_path(execution_path)

    await persist_and_record_usage(
        session=None,
        user_id=user_id,
        prompt_tokens=phase_usage.input_tokens,
        completion_tokens=phase_usage.output_tokens,
        cache_read_tokens=phase_usage.cache_read_tokens,
        cache_creation_tokens=phase_usage.cache_creation_tokens,
        log_prefix=f"[dream:{phase_usage.phase}]",
        cost_usd=phase_usage.cost_usd,
        model=phase_usage.model,
        provider=provider,
        block_name_override=f"copilot:dream:{phase_usage.phase}",
        graph_exec_id_override=pass_id,
        extra_metadata={
            "source": "dream_pass",
            "dream_pass_id": pass_id,
            "dream_phase": phase_usage.phase,
            "execution_path": execution_path,
        },
        # Dream is background work — it rolls up under the user's
        # weekly cap but must not eat the interactive daily budget.
        skip_daily=True,
    )
