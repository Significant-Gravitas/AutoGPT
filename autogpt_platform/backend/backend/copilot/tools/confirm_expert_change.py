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
from backend.data.redis_client import AsyncRedisClient, get_redis_async

from .base import BaseTool
from .expert_proposal import (
    apply_proposal,
    autopilot_session_guard,
    load_bound_proposal,
)
from .models import (
    EXPERT_CHANGE_LANDED_REASONS,
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertChangeBatchAppliedResponse,
    ExpertChangeError,
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


_MALFORMED_IDS_MESSAGE = (
    "confirm_expert_change takes confirmation_id as a single id string or "
    f"confirmation_ids as a list of 1 to {MAX_BATCH_CONFIRMATIONS} id "
    "strings, and nothing else. No confirmation was used up."
)


class _ConfirmParams(BaseModel):
    """Both ids as a model may actually send them, checked before anything
    is consumed.

    Filling both keys and nulling the unused one is a routine tool-call
    shape, so ``None`` is a value here rather than a crash. Anything else
    that is not a string or a list of strings is answered with the guidance
    below instead of escaping as a traceback, which ``BaseTool.execute``
    would flatten into "an error occurred" — leaving the model unable to
    tell whether the ids were consumed.
    """

    confirmation_id: str | None = None
    confirmation_ids: list[str] | None = Field(
        default=None, max_length=MAX_BATCH_CONFIRMATIONS
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
        confirmation_id: object = None,
        confirmation_ids: object = None,
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

        try:
            params = _ConfirmParams.model_validate(
                {
                    "confirmation_id": confirmation_id,
                    "confirmation_ids": confirmation_ids,
                }
            )
        except ValidationError:
            return ErrorResponse(
                message=_MALFORMED_IDS_MESSAGE,
                session_id=session_id,
            )

        single = (params.confirmation_id or "").strip()
        # An empty list is "no batch", not "a batch of nothing", so a model
        # that fills both keys but only means one of them still works.
        batch = params.confirmation_ids or None
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
    if isinstance(proposal, ExpertChangeError):
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
    ids = [candidate.strip() for candidate in confirmation_ids]
    if not all(ids):
        return ErrorResponse(
            message="confirmation_ids must not contain a blank id.",
            session_id=session_id,
        )

    redis = await get_redis_async()
    results = [
        await _confirm_and_apply(redis, user_id, session, candidate)
        for candidate in ids
    ]
    experts = _created_experts(results)
    return ExpertChangeBatchAppliedResponse(
        message=_batch_message(results),
        session_id=session_id,
        applied=any(result.outcome != "failed" for result in results),
        results=results,
        experts=experts,
    )


async def _confirm_and_apply(
    redis: AsyncRedisClient,
    user_id: str,
    session: ChatSession,
    confirmation_id: str,
) -> ExpertChangeResult:
    """Consume one id and apply it before the next id is touched.

    ``load_bound_proposal`` tombstones as it checks, so consuming all N up
    front and applying afterwards means an interrupted call — a Stop or a
    disconnect, both of which cancel the turn mid-tool — permanently burns
    every approval it never got to, with nothing created. Because no apply
    reads the ids beside it, interleaving costs nothing and leaves the ids
    past the interruption still valid and re-confirmable.
    """
    proposal = await load_bound_proposal(redis, confirmation_id, user_id, session)
    if isinstance(proposal, ExpertChangeError):
        return _unapplied_result(confirmation_id, proposal)
    applied = await apply_proposal(user_id, session.session_id, proposal)
    if isinstance(applied, ExpertChangeAppliedResponse):
        return ExpertChangeResult(
            confirmation_id=confirmation_id,
            outcome="applied",
            kind=applied.kind,
            expert=applied.expert,
            failed_workflows=applied.failed_workflows,
        )
    return _unapplied_result(confirmation_id, applied)


def _unapplied_result(
    confirmation_id: str, error: ExpertChangeError
) -> ExpertChangeResult:
    """Not every refusal is a failure.

    ``already_applied`` and ``applied_but_expert_gone`` both mean the change
    is on the team — the first from an earlier confirm, the second from this
    one — so folding them in with the genuine failures makes the batch tell
    the user a teammate they have was never added.
    """
    landed = error.reason in EXPERT_CHANGE_LANDED_REASONS
    return ExpertChangeResult(
        confirmation_id=confirmation_id,
        outcome="already_applied" if landed else "failed",
        reason=error.reason,
        error=error.message,
    )


def _created_experts(results: list[ExpertChangeResult]) -> list[ExpertSummary]:
    return [
        result.expert
        for result in results
        if result.outcome == "applied" and result.expert is not None
    ]


def _batch_message(results: list[ExpertChangeResult]) -> str:
    """Read the batch back the way the user has to hear it.

    A partial batch is the dangerous case: without naming both halves the
    model announces the whole approval as done, so every failure and every
    half-installed hire gets its own sentence.
    """
    created = _created_experts(results)
    already = [result for result in results if result.outcome == "already_applied"]
    failures = [result for result in results if result.outcome == "failed"]
    landed = len(results) - len(failures)
    names = ", ".join(expert.name for expert in created)
    named = f": {names}" if names else ""

    if not landed:
        parts = [f"Nothing was applied — all {len(results)} confirmations failed."]
    elif failures:
        parts = [f"{landed} of {len(results)} approved changes are done{named}."]
    else:
        parts = [f"All {len(results)} approved changes are done{named}."]

    if already:
        parts.append(
            f"{len(already)} of them were already applied before this call — "
            "they are done, do not re-preview or retry them."
        )
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
