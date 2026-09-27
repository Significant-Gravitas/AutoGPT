"""What a batch dream pass cost: one PlatformCostLog row per phase that
landed, each charged at most once however many times the pass's end or its
cleanup runs, and the usage of those phases for the pass's DreamPass record."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel

from backend.copilot.inference.context import InferenceContext, InferenceScope
from backend.copilot.inference.routing import anthropic_batch_route

from .batch_state import (
    claim_per_phase_charging,
    claim_phase_charge,
    read_state_or_none,
)
from .billing import record_phase_cost
from .phase_jobs import PHASE_TIERS, phase_job
from .schemas import DreamPassUsage, DreamPhase
from .usage import batch_pass_usage, landed_phase_usage

logger = logging.getLogger(__name__)

# How one landed phase's charge went: claimed by this call and charged, found
# with nothing to charge (an errored phase, no model, no usage), or its one
# attempt failed (logged, not retried); claimed by an earlier call; or
# unknown, its claim neither made nor refused, so the next cleanup tries.
PhaseCharge = Literal["charged", "nothing", "failed", "claimed", "unknown"]

# The phase charges under way. Each runs to its end whatever happens to the
# caller that started it (``_charge_phase``), and is kept here until it does.
_CHARGES_IN_FLIGHT: set[asyncio.Task[PhaseCharge]] = set()


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

    Each phase is claimed on its own (``claim_phase_charge``) right before it
    is charged, so a cleanup cut short after some phases resumes with the
    rest: the claimed ones are skipped, the others charged. A phase's claim
    and charge run to their end even when the caller is cancelled (a reaper
    run out of budget, the executor's bound on a drop hook), so no cut lands
    between them. A pass an earlier build charged whole is not charged again
    (``claim_per_phase_charging``). Not *settled* when a claim could not be
    made or checked: the next cleanup tries again.

    At most once, not exactly once. A process that dies between a phase's
    claim and its cost-log write leaves that phase uncharged, and a charge
    that fails part-way is not retried: the trial ledger and the weekly
    counter can move before the cost-log write raises, so a second attempt
    could charge them twice. Both under-account the pass rather than risk a
    second charge for a phase; the failure is logged, the crash is not.

    Each phase is recorded through ``billing.record_phase_cost`` like a
    sync phase, attributed to the pass's expert, and priced from its
    model's catalog price card (``backend/copilot/price_card.py``):
    Anthropic's additive cache buckets, less the batch path's half-price
    discount.
    """
    try:
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
    scope = InferenceScope(user_id=user_id, expert_id=expert_id)
    charges: dict[DreamPhase, PhaseCharge] = {}
    for phase in PHASE_TIERS:
        row = state.get(phase)
        if row is not None:
            charges[phase] = await _charge_phase(
                scope, pass_id, phase, row, phase_models
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


async def _charge_phase(
    scope: InferenceScope,
    pass_id: str,
    phase: DreamPhase,
    row: dict[str, Any],
    phase_models: dict[str, str],
) -> PhaseCharge:
    """Claim and charge one phase in a task of its own, which the caller's
    cancellation does not reach: a claimed phase is charged even when its
    caller stops waiting."""
    task = asyncio.ensure_future(
        _claim_and_charge(scope, pass_id, phase, row, phase_models)
    )
    _CHARGES_IN_FLIGHT.add(task)
    task.add_done_callback(_CHARGES_IN_FLIGHT.discard)
    return await asyncio.shield(task)


async def _claim_and_charge(
    scope: InferenceScope,
    pass_id: str,
    phase: DreamPhase,
    row: dict[str, Any],
    phase_models: dict[str, str],
) -> PhaseCharge:
    try:
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
        # Phase errored — don't record usage for a phase that didn't
        # complete; downstream phases never ran either.
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
