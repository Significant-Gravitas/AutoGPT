"""Apply previewed hires or raises by their one-time confirmation ids.

Step 2 of the confirm-gated team-change flow. One confirm tool serves every
kind: it takes nothing but the ids, so the applied change is exactly what the
user saw. Each id is single-use and bound to the Autopilot session that
produced it. A user who approves several previews in one breath is one call,
not one round-trip each — pass ``confirmation_ids``.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.copilot.model import ChatSession
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .expert_proposal import (
    ExpertChangeProposal,
    apply_proposal,
    autopilot_session_guard,
    load_bound_proposal,
)
from .models import (
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertChangeBatchAppliedResponse,
    ExpertChangeResult,
    ExpertSummary,
    ToolResponseBase,
)

_PROPOSAL_FIELDS = (
    "template_id",
    "name",
    "role",
    "about",
    "boundaries",
    "voice_preferences",
    "weekly_budget",
)

MAX_BATCH_CONFIRMATIONS = 20


class _BatchParams(BaseModel):
    """Bounds on the batch itself, checked before anything is consumed."""

    confirmation_ids: list[str] = Field(
        min_length=1, max_length=MAX_BATCH_CONFIRMATIONS
    )


class ConfirmExpertChangeTool(BaseTool):
    """Create the experts previewed by hire_expert or raise_expert."""

    @property
    def name(self) -> str:
        return "confirm_expert_change"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Apply a hire_expert or raise_expert proposal after the user has "
            "approved it. Takes only confirmation ids and creates exactly the "
            "previewed experts; each id is single-use. When the user approves "
            "several previews at once, confirm them in a single call by "
            "passing confirmation_ids."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "confirmation_id": {
                    "type": "string",
                    "description": (
                        "One id returned by hire_expert/raise_expert. Use "
                        "this or confirmation_ids, never both."
                    ),
                },
                "confirmation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_BATCH_CONFIRMATIONS,
                    "description": (
                        "Every id the user approved together, applied in one "
                        f"call (max {MAX_BATCH_CONFIRMATIONS})."
                    ),
                },
            },
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        confirmation_id: str = "",
        confirmation_ids: list[str] | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if error := autopilot_session_guard(user_id, session):
            return error
        assert user_id is not None

        if any(field in kwargs for field in _PROPOSAL_FIELDS):
            return ErrorResponse(
                message=(
                    "confirm_expert_change creates exactly the previewed "
                    "expert and does not accept new values. Call hire_expert "
                    "or raise_expert to propose something different."
                ),
                session_id=session_id,
            )

        single = confirmation_id.strip()
        # An empty list is "no batch", not "a batch of nothing", so a model
        # that fills both keys but only means one of them still works.
        batch = confirmation_ids or None
        if single and batch:
            return ErrorResponse(
                message=(
                    "Pass confirmation_id for one approved preview or "
                    "confirmation_ids for several, never both."
                ),
                session_id=session_id,
            )
        if batch:
            return await _confirm_batch(user_id, session, batch)
        if not single:
            return ErrorResponse(
                message=(
                    "A confirmation_id from hire_expert or raise_expert is "
                    "required, or confirmation_ids to apply several at once."
                ),
                session_id=session_id,
            )
        return await _confirm_one(user_id, session, single)


async def _confirm_one(
    user_id: str,
    session: ChatSession,
    confirmation_id: str,
) -> ToolResponseBase:
    proposal = await load_bound_proposal(
        await get_redis_async(),
        confirmation_id,
        user_id,
        session,
    )
    if isinstance(proposal, ErrorResponse):
        return proposal
    return await apply_proposal(user_id, session.session_id, proposal)


async def _confirm_batch(
    user_id: str,
    session: ChatSession,
    confirmation_ids: list[str],
) -> ToolResponseBase:
    """Apply every approved id, reporting each one's outcome separately.

    ``capacity_error`` only runs at preview time, so N previews that were
    each under the active-expert cap can exceed it together. Nothing is
    pre-computed here to stop that: the creation transaction is the cap's
    real enforcement point, so the overflowing hires come back as ordinary
    per-id limit errors and the team can never actually exceed the cap.
    """
    session_id = session.session_id
    try:
        params = _BatchParams(confirmation_ids=confirmation_ids)
    except ValidationError:
        return ErrorResponse(
            message=(
                "confirmation_ids must be a list of 1 to "
                f"{MAX_BATCH_CONFIRMATIONS} confirmation ids."
            ),
            session_id=session_id,
        )
    ids = [candidate.strip() for candidate in params.confirmation_ids]
    if not all(ids):
        return ErrorResponse(
            message="confirmation_ids must not contain a blank id.",
            session_id=session_id,
        )

    # Every id is resolved and bound-checked before a single write happens,
    # so a stale or foreign id in the batch is reported instead of deciding
    # what the ids beside it are allowed to do.
    #
    # ``load_bound_proposal`` consumes as it checks, so this pass burns the
    # ids it accepts. That is deliberate: it makes one batch behave exactly
    # like N sequential single-id confirms, where a proposal that fails at
    # apply time is likewise discarded and re-previewed. Splitting the check
    # from the consume would give the batch path its own, weaker single-use
    # guarantee than the single-id path it has to match.
    redis = await get_redis_async()
    loaded = [
        await load_bound_proposal(redis, candidate, user_id, session)
        for candidate in ids
    ]
    results = [
        await _apply_one(user_id, session_id, candidate, proposal)
        for candidate, proposal in zip(ids, loaded)
    ]
    experts = [
        result.expert
        for result in results
        if result.applied and result.expert is not None
    ]
    return ExpertChangeBatchAppliedResponse(
        message=_batch_message(results),
        session_id=session_id,
        applied=bool(experts),
        results=results,
        experts=experts,
    )


async def _apply_one(
    user_id: str,
    session_id: str,
    confirmation_id: str,
    proposal: ExpertChangeProposal | ErrorResponse,
) -> ExpertChangeResult:
    if isinstance(proposal, ErrorResponse):
        return ExpertChangeResult(
            confirmation_id=confirmation_id,
            applied=False,
            error=proposal.message,
        )
    applied = await apply_proposal(user_id, session_id, proposal)
    if isinstance(applied, ExpertChangeAppliedResponse):
        return ExpertChangeResult(
            confirmation_id=confirmation_id,
            applied=True,
            kind=applied.kind,
            expert=applied.expert,
            failed_workflows=applied.failed_workflows,
        )
    return ExpertChangeResult(
        confirmation_id=confirmation_id,
        applied=False,
        error=applied.message,
    )


def _batch_message(results: list[ExpertChangeResult]) -> str:
    """Read the batch back the way the user has to hear it.

    A partial batch is the dangerous case: without naming both halves the
    model announces the whole approval as done, so every failure and every
    half-installed hire gets its own sentence.
    """
    created: list[ExpertSummary] = [
        result.expert
        for result in results
        if result.applied and result.expert is not None
    ]
    failures = [result for result in results if not result.applied]
    names = ", ".join(expert.name for expert in created)

    if not created:
        parts = [f"Nothing was applied — all {len(results)} confirmations failed."]
    elif failures:
        parts = [f"{len(created)} of {len(results)} applied: {names}."]
    else:
        parts = [f"All {len(results)} approved changes applied: {names}."]

    parts.extend(
        f"{result.confirmation_id} failed: {result.error}" for result in failures
    )
    parts.extend(
        f"{result.expert.name} joined without these workflows: "
        f"{', '.join(result.failed_workflows)}."
        for result in results
        if result.expert is not None and result.failed_workflows
    )
    parts.append("Tell the user exactly who is on the team now and what did not land.")
    return " ".join(parts)
