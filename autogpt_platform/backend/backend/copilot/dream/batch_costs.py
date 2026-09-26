"""What a batch dream pass cost: one PlatformCostLog row per phase that
landed, charged at most once whichever way the pass ends, and the usage
of those phases for the pass's DreamPass record."""

from __future__ import annotations

import logging
from typing import Any

from backend.copilot.inference.context import InferenceContext, InferenceScope
from backend.copilot.inference.routing import anthropic_batch_route

from .batch_state import claim_costs_logged_gate, read_state_or_none
from .billing import record_phase_cost
from .phase_jobs import PHASE_TIERS, phase_job
from .schemas import DreamPassUsage, DreamPhase
from .usage import batch_pass_usage, landed_phase_usage

logger = logging.getLogger(__name__)


async def log_all_phase_costs(
    *,
    user_id: str,
    expert_id: str | None,
    pass_id: str,
    state: dict[str, dict[str, Any]],
    phase_models: dict[str, str],
) -> None:
    """One PlatformCostLog row per phase, on the ``anthropic_batch`` route.

    Idempotent via a Redis SETNX gate keyed on ``pass_id``: a pass logs its
    costs from whichever terminal path it takes, success or failure, and a
    repeated delivery of a finished batch must not charge it twice. The gate
    is claimed once, before the loop, and never released, so each phase is
    charged at most once: a phase whose charge fails is logged and stays
    uncharged, because no later delivery gets past the gate to retry it,
    while the other landed phases are still charged. A partial failure
    under-charges the pass rather than risk charging a phase twice.

    Each phase is recorded through ``billing.record_phase_cost`` like a
    sync phase, attributed to the pass's expert, and priced from its
    model's catalog price card (``backend/copilot/price_card.py``):
    Anthropic's additive cache buckets, less the batch path's half-price
    discount.

    No-ops on per-phase failure — apply already wrote the user-facing
    memory operations; a cost-log blip shouldn't take that down.
    """
    if not await claim_costs_logged_gate(pass_id):
        logger.info(
            "Skipping batch cost log for pass=%s — already charged",
            pass_id,
        )
        return

    for phase in PHASE_TIERS:
        row = state.get(phase)
        if row is None:
            continue
        try:
            scope = InferenceScope(user_id=user_id, expert_id=expert_id)
            await _log_phase_cost(scope, pass_id, phase, row, phase_models)
        except Exception:
            logger.exception(
                "Failed to log batch cost for pass=%s phase=%s", pass_id, phase
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


async def _log_phase_cost(
    scope: InferenceScope,
    pass_id: str,
    phase: DreamPhase,
    row: dict[str, Any],
    phase_models: dict[str, str],
) -> None:
    if row.get("error"):
        # Phase errored — don't record usage for a phase that didn't
        # complete; downstream phases never ran either.
        return
    phase_model = phase_models.get(phase)
    if not phase_model:
        logger.warning(
            "No model recorded for pass=%s phase=%s — skipping cost log",
            pass_id,
            phase,
        )
        return
    usage = landed_phase_usage(row, phase_model)
    if usage is None:
        return
    ctx = InferenceContext(
        scope=scope,
        job=phase_job(phase, pass_id, timeout_seconds=None, pinned_model=phase_model),
        route=anthropic_batch_route(phase_model),
    )
    await record_phase_cost(ctx, usage)
