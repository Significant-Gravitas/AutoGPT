"""What a dream phase is to the inference layer.

The sync orchestrator and the batch callbacks describe a phase the same way:
a ``dream`` job correlated by the pass id, deferred (nobody waits on a
dream), on the tier the phase has always run on.
"""

from backend.copilot.inference.context import InferenceJob, InferenceTier

from .schemas import DreamPhase

# Consolidate and sanitize run on the fast standard model, recombine on the
# fast advanced one; the batch submit picks its per-phase models on the same
# split (``batch_submit.phase_models_for_config``).
PHASE_TIERS: dict[DreamPhase, InferenceTier] = {
    "consolidate": "standard",
    "recombine": "advanced",
    "sanitize": "standard",
}


def phase_job(
    phase: DreamPhase,
    pass_id: str,
    *,
    timeout_seconds: float | None,
    pinned_model: str | None = None,
) -> InferenceJob:
    """The job of one phase of pass *pass_id*. *pinned_model* is for a phase
    whose model was already fixed, as a batch submit fixes it."""
    return InferenceJob(
        kind="dream",
        phase=phase,
        correlation_id=pass_id,
        latency_class="deferred",
        tier=PHASE_TIERS[phase],
        timeout_seconds=timeout_seconds,
        pinned_model=pinned_model,
    )
