"""What a batch dream pass cost: one PlatformCostLog row per phase that
landed, each charged at most once however many times the pass's end or its
cleanup runs, and the usage of those phases for the pass's DreamPass record.

Every such charge goes by the usage attribution (#15208) the pass started
under, read off its row: batch settlement is deferred, made in another
process and possibly much later, so the row is the one place that knows it.
A sync pass charges each phase in-process as it lands
(``billing.record_phase_cost``), under the attribution it captured as it
started (``orchestrator``'s ``attributed_usage``), the same value its row
keeps; the row rule here is for batch settlement only."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel

from backend.copilot.inference.context import InferenceContext, InferenceScope
from backend.copilot.inference.routing import anthropic_batch_route
from backend.copilot.trial_cost_context import restore_cost_context

from .batch_state import (
    claim_per_phase_charging,
    claim_phase_charge,
    read_state_or_none,
)
from .billing import record_phase_cost
from .phase_jobs import PHASE_TIERS, phase_job
from .schemas import DreamPassUsage, DreamPhase
from .store import read_cost_context
from .usage import batch_pass_usage, landed_phase_usage

logger = logging.getLogger(__name__)

# How one landed phase's charge went: claimed by this call and charged, found
# with nothing to charge (an errored phase, no model, no usage), or its one
# attempt failed (logged, not retried); claimed by an earlier call; or
# unknown, its claim neither made nor refused, so the next cleanup tries.
PhaseCharge = Literal["charged", "nothing", "failed", "claimed", "unknown"]

# How long a pass's charge admits claims, from its start: a phase not
# claimed by then is left for the next cleanup, as is the pass whose gate
# check is still waiting on Redis (cutting that check claims nothing). A
# claim issued in time is never cut by the window.
CLAIM_WINDOW_SECONDS = 30.0
# How long one phase's claim, once issued, is waited for, the Redis client's
# retries included, as the cleanup waits on a state read. A claim Redis took
# whose reply has not come by then leaves its phase claimed and uncharged:
# the at-most-once of any Redis command whose reply is lost.
CLAIM_TIMEOUT_SECONDS = 5.0

# The passes' charges under way. Each runs on whatever happens to the caller
# that started it (``charge_landed_phases``), and is kept here until it ends.
_CHARGES_IN_FLIGHT: set[asyncio.Task[PhaseCharges]] = set()


class PhaseCharges(BaseModel):
    """What charging a pass's landed phases did: the phases this call
    charged, and whether every landed phase is settled (charged, or its one
    attempt made, by this call or an earlier one)."""

    charged: list[DreamPhase] = []
    settled: bool


async def charge_landed_phases(
    *,
    user_id: str,
    expert_id: str | None,
    pass_id: str,
    state: dict[str, dict[str, Any]],
    phase_models: dict[str, str],
) -> PhaseCharges:
    """One PlatformCostLog row per phase landed in *state*, on the
    ``anthropic_batch`` route, each phase charged at most once whichever
    terminal path the pass takes and however often its cleanup runs.

    The charge (the pass's gate first, ``claim_per_phase_charging``, so a
    pass an earlier build charged whole is not charged again; then each
    landed phase claimed, ``claim_phase_charge``, and charged in turn) runs
    in a task of its own, which the caller's cancellation does not reach (a
    reaper run out of budget, the executor's bound on a drop hook): a cleanup
    that has read the state makes the charge even when it is cut while Redis
    answers slowly, and no cut lands between a claim and its charge. The task
    issues no claim ``CLAIM_WINDOW_SECONDS`` after it starts, and never cuts
    one it issued for the window (each waits up to ``CLAIM_TIMEOUT_SECONDS``
    of its own); a phase it claims is charged to its end. Each phase is
    claimed on its own,
    so a charge stopped part-way resumes with the rest: the claimed ones are
    skipped, the others charged. Not *settled* when the gate or a claim could
    not be made or checked: the next cleanup tries again.

    At most once, not exactly once. A process that dies between a phase's
    claim and its cost-log write leaves that phase uncharged, as does a claim
    Redis took whose reply was lost or outlasted ``CLAIM_TIMEOUT_SECONDS``;
    and a charge that fails part-way is not retried: the trial ledger and the weekly
    counter can move before the cost-log write raises, so a second attempt
    could charge them twice. Both under-account the pass rather than risk a
    second charge for a phase; the failure is logged, the crash is not.

    Each phase is recorded through ``billing.record_phase_cost`` like a
    sync phase, attributed to the pass's expert, and priced from its
    model's catalog price card (``backend/copilot/price_card.py``):
    Anthropic's additive cache buckets, less the batch path's half-price
    discount.

    Every charge is made under the usage attribution the pass started under
    (#15208), read off its row (``store.read_cost_context``) whoever charges
    and however late: the callback, a failed pass's end, a closed pass's
    dropped batch, the reaper. Never under the attribution the account has
    now: a conversion after the pass started does not move what it already
    ran. A pass whose row carries no attribution is never charged, settled
    all the same; one whose row cannot be read is left for the next cleanup.
    This is the batch route's settlement; a sync pass charges in-process,
    under the attribution it captured at its start (see the module).
    """
    try:
        context = await read_cost_context(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read the usage attribution to "
            "charge it under; left for the next cleanup",
            exc_info=True,
        )
        return PhaseCharges(settled=False)
    if context is None or context.user_id != user_id:
        logger.warning(
            f"Dream pass {pass_id}: its record carries no usage attribution "
            f"for user {user_id[:12]}; its landed phases are not charged"
        )
        return PhaseCharges(settled=True)
    scope = InferenceScope(user_id=user_id, expert_id=expert_id)
    with restore_cost_context(user_id, context):
        # The task takes the attribution with it: it copies the context it
        # is created in, and goes on under it after this returns.
        task = asyncio.ensure_future(_charge_pass(scope, pass_id, state, phase_models))
    _CHARGES_IN_FLIGHT.add(task)
    task.add_done_callback(_CHARGES_IN_FLIGHT.discard)
    return await asyncio.shield(task)


async def _charge_pass(
    scope: InferenceScope,
    pass_id: str,
    state: dict[str, dict[str, Any]],
    phase_models: dict[str, str],
) -> PhaseCharges:
    """The gate, then each landed phase claimed and charged; admitting
    claims until ``CLAIM_WINDOW_SECONDS`` after it starts."""
    deadline = asyncio.get_running_loop().time() + CLAIM_WINDOW_SECONDS
    try:
        async with asyncio.timeout_at(deadline):
            per_phase = await claim_per_phase_charging(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not claim its charging; left for the "
            "next cleanup",
            exc_info=True,
        )
        return PhaseCharges(settled=False)
    if not per_phase:
        logger.info(f"Dream pass {pass_id}: charged whole by an earlier build")
        return PhaseCharges(settled=True)
    charges: dict[DreamPhase, PhaseCharge] = {}
    for phase in PHASE_TIERS:
        row = state.get(phase)
        if row is not None:
            charges[phase] = await _claim_and_charge(
                scope, pass_id, phase, row, phase_models, deadline
            )
    return PhaseCharges(
        charged=[phase for phase, how in charges.items() if how == "charged"],
        settled="unknown" not in charges.values(),
    )


async def recorded_usage(
    pass_id: str, phase_models: dict[str, str]
) -> DreamPassUsage | None:
    """What the pass's landed phases used, read off its Redis state; ``None``
    when the state cannot be read."""
    return landed_usage(await read_state_or_none(pass_id), phase_models, pass_id)


def landed_usage(
    state: dict[str, dict[str, Any]] | None,
    phase_models: dict[str, str],
    pass_id: str,
) -> DreamPassUsage | None:
    """What the pass's landed phases used, for its record; ``None``
    (unknown) when its state could not be read. A state row that will not
    price costs the record its usage, never the pass its outcome."""
    if state is None:
        return None
    try:
        return batch_pass_usage(state, phase_models)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not price its landed phases for the record",
            exc_info=True,
        )
        return None


async def _claim_and_charge(
    scope: InferenceScope,
    pass_id: str,
    phase: DreamPhase,
    row: dict[str, Any],
    phase_models: dict[str, str],
    deadline: float,
) -> PhaseCharge:
    """Claim one phase, then charge it. The claim is issued only before
    *deadline* (loop time), the charge's claim window, else the phase is left
    for the next cleanup; once issued it is waited for up to
    ``CLAIM_TIMEOUT_SECONDS`` whatever the window, which must not cut a claim
    Redis may have taken."""
    if asyncio.get_running_loop().time() >= deadline:
        logger.warning(
            f"Dream pass {pass_id}: its charge stopped claiming before its "
            f"{phase} phase; left for the next cleanup"
        )
        return "unknown"
    try:
        async with asyncio.timeout(CLAIM_TIMEOUT_SECONDS):
            claimed = await claim_phase_charge(pass_id, phase)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not claim the charge of its {phase} "
            "phase; left for the next cleanup",
            exc_info=True,
        )
        return "unknown"
    if not claimed:
        return "claimed"
    try:
        if not await _log_phase_cost(scope, pass_id, phase, row, phase_models):
            return "nothing"
    except Exception:
        # Apply already wrote the user-facing memory operations; a cost-log
        # blip shouldn't take that down, and a retry could charge twice.
        logger.exception(
            "Failed to log batch cost for pass=%s phase=%s; not retried",
            pass_id,
            phase,
        )
        return "failed"
    return "charged"


async def _log_phase_cost(
    scope: InferenceScope,
    pass_id: str,
    phase: DreamPhase,
    row: dict[str, Any],
    phase_models: dict[str, str],
) -> bool:
    """Charge one landed phase; ``False`` when there is nothing to charge."""
    if row.get("error"):
        return False
    phase_model = phase_models.get(phase)
    if not phase_model:
        logger.warning(
            "No model recorded for pass=%s phase=%s — skipping cost log",
            pass_id,
            phase,
        )
        return False
    usage = landed_phase_usage(row, phase_model)
    if usage is None:
        return False
    ctx = InferenceContext(
        scope=scope,
        job=phase_job(phase, pass_id, timeout_seconds=None, pinned_model=phase_model),
        route=anthropic_batch_route(phase_model),
    )
    await record_phase_cost(ctx, usage)
    return True
