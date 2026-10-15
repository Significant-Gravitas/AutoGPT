"""What a dream pass used, per phase and in total, on either route.

The sync orchestrator gets each phase's usage back priced from the call it
made. A batch pass's phases are priced here from the rows its callbacks kept
in Redis, the way ``inference.record.record`` prices their cost rows: on the
Anthropic batch route, at the batch discount, and a phase that reported no
tokens keeps an unknown cost rather than a known zero.
"""

from typing import Any

from backend.copilot.inference.context import InferenceUsage
from backend.copilot.inference.record import price
from backend.copilot.inference.routing import anthropic_batch_route

from .phase_jobs import PHASE_TIERS
from .routing import ExecutionPath, batch_discount
from .schemas import DreamPassUsage, DreamPhase, PhaseUsage


def batch_pass_usage(
    state: dict[str, dict[str, Any]], phase_models: dict[str, str]
) -> DreamPassUsage | None:
    """The usage of every phase of a batch pass whose result landed without an
    error, priced on the model it ran on; ``None`` when none did."""
    phases = [
        phase_usage(phase, _priced(usage))
        for phase in PHASE_TIERS
        if (usage := landed_phase_usage(state.get(phase), phase_models.get(phase)))
        is not None
    ]
    return aggregate_usage(phases, "anthropic_batch") if phases else None


def landed_phase_usage(
    row: dict[str, Any] | None, model: str | None
) -> InferenceUsage | None:
    """The tokens a batch phase's state row records, on *model*. ``None`` for a
    phase that has not landed, came back errored, or has no model recorded."""
    if row is None or row.get("error") or not model:
        return None
    return InferenceUsage(
        model=model,
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        cache_read_tokens=int(row.get("cache_read_tokens") or 0),
        cache_creation_tokens=int(row.get("cache_creation_tokens") or 0),
        payer=anthropic_batch_route(model).payer,
    )


def phase_usage(phase: DreamPhase, usage: InferenceUsage) -> PhaseUsage:
    return PhaseUsage(
        phase=phase,
        model=usage.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_creation_tokens,
        cost_usd=usage.cost_usd,
    )


def aggregate_usage(
    phases: list[PhaseUsage], execution_path: ExecutionPath
) -> DreamPassUsage:
    """Roll up per-phase usage into a ``DreamPassUsage``.

    ``total_cost_usd`` is None when any single phase had unknown cost
    so we never silently bill at a partial figure.
    """
    total_cost: float | None = 0.0
    for p in phases:
        if p.cost_usd is None:
            total_cost = None
            break
        total_cost += p.cost_usd
    return DreamPassUsage(
        phases=phases,
        total_input_tokens=sum(p.input_tokens for p in phases),
        total_output_tokens=sum(p.output_tokens for p in phases),
        total_cache_read_tokens=sum(p.cache_read_tokens for p in phases),
        total_cache_creation_tokens=sum(p.cache_creation_tokens for p in phases),
        total_cost_usd=total_cost,
        discount_applied=batch_discount(execution_path),
    )


def _priced(usage: InferenceUsage) -> InferenceUsage:
    reported = (
        usage.input_tokens
        or usage.output_tokens
        or usage.cache_read_tokens
        or usage.cache_creation_tokens
    )
    return price(usage, anthropic_batch_route(usage.model)) if reported else usage
