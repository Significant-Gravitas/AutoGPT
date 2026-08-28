"""Apply one or several user-approved expert-change previews."""

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
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertChangeBatchAppliedResponse,
    ExpertChangeBatchResult,
    ExpertChangeError,
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
    "Choose one confirmation id, or a list of 1 to "
    f"{MAX_BATCH_CONFIRMATIONS} confirmation ids."
)


class _ConfirmParams(BaseModel):
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
            "Apply user-approved hire_expert or raise_expert previews. Pass "
            "confirmation_id for one preview or confirmation_ids for a whole "
            "approved roster. Each id is single-use and retries are idempotent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "confirmation_id": {
                    "type": "string",
                    "description": "One approved preview id; omit for a roster.",
                },
                "confirmation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_BATCH_CONFIRMATIONS,
                    "description": "All preview ids approved together in one roster.",
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
                message="The approved preview cannot be changed during confirmation.",
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
        batch = params.confirmation_ids or []
        if single and batch:
            return ErrorResponse(
                message="Choose one preview or one roster, not both.",
                session_id=session_id,
            )
        if batch:
            return await _confirm_batch(user_id, session, batch)
        if not single:
            return ErrorResponse(
                message=_MALFORMED_IDS_MESSAGE,
                session_id=session_id,
            )
        return await _confirm_one(user_id, session, single)


async def _confirm_one(
    user_id: str, session: ChatSession, confirmation_id: str
) -> ToolResponseBase:
    proposal = await load_bound_proposal(
        await get_redis_async(),
        confirmation_id,
        user_id,
        session,
    )
    if isinstance(proposal, ErrorResponse):
        return proposal
    applied = await apply_proposal(user_id, session.session_id, proposal)
    if isinstance(applied, ExpertChangeAppliedResponse):
        return applied.model_copy(update={"confirmation_id": confirmation_id})
    return applied


async def _confirm_batch(
    user_id: str, session: ChatSession, confirmation_ids: list[str]
) -> ToolResponseBase:
    ids = [candidate.strip() for candidate in confirmation_ids]
    if not ids or not all(ids):
        return ErrorResponse(
            message="Every selected preview must have a valid confirmation id.",
            session_id=session.session_id,
        )
    redis = await get_redis_async()
    results = [
        await _confirm_and_apply(redis, user_id, session, confirmation_id)
        for confirmation_id in ids
    ]
    experts = _created_experts(results)
    return ExpertChangeBatchAppliedResponse(
        message=_batch_message(results),
        session_id=session.session_id,
        applied=any(result.outcome != "failed" for result in results),
        results=results,
        experts=experts,
    )


async def _confirm_and_apply(
    redis: AsyncRedisClient,
    user_id: str,
    session: ChatSession,
    confirmation_id: str,
) -> ExpertChangeBatchResult:
    proposal = await load_bound_proposal(redis, confirmation_id, user_id, session)
    if isinstance(proposal, ExpertChangeError):
        return ExpertChangeBatchResult(
            confirmation_id=confirmation_id,
            outcome=(
                "already_applied" if proposal.reason == "already_applied" else "failed"
            ),
            reason=proposal.reason,
        )
    if isinstance(proposal, ErrorResponse):
        return ExpertChangeBatchResult(
            confirmation_id=confirmation_id,
            outcome="failed",
            reason="unexpected_failure",
        )

    applied = await apply_proposal(user_id, session.session_id, proposal)
    if isinstance(applied, ExpertChangeAppliedResponse):
        return ExpertChangeBatchResult(
            confirmation_id=confirmation_id,
            outcome="applied",
            proposed_name=proposal.preview.name,
            kind=applied.kind,
            expert=applied.expert,
            failed_workflows=applied.failed_workflows,
        )
    if not isinstance(applied, ErrorResponse):
        return ExpertChangeBatchResult(
            confirmation_id=confirmation_id,
            outcome="failed",
            proposed_name=proposal.preview.name,
            kind=proposal.preview.kind,
            reason="unexpected_failure",
        )
    return ExpertChangeBatchResult(
        confirmation_id=confirmation_id,
        outcome="failed",
        proposed_name=proposal.preview.name,
        kind=proposal.preview.kind,
        reason=_apply_failure_reason(applied),
    )


def _apply_failure_reason(error: ErrorResponse) -> str:
    message = error.message.lower()
    if "limit" in message or "maximum" in message:
        return "limit_reached"
    if "workspace" in message or "temporarily unavailable" in message:
        return "workspace_unavailable"
    if "edited somewhere else" in message:
        return "expert_moved"
    if "template" in message or "no longer exists" in message:
        return "template_gone"
    return "unexpected_failure"


def _created_experts(
    results: list[ExpertChangeBatchResult],
) -> list[ExpertSummary]:
    seen: set[str] = set()
    experts: list[ExpertSummary] = []
    for result in results:
        if result.expert is None or result.expert.id in seen:
            continue
        seen.add(result.expert.id)
        experts.append(result.expert)
    return experts


_REASON_COPY = {
    "expired": "the preview expired",
    "not_approved": "the preview was not approved",
    "unwatermarked": "the preview is too old",
    "wrong_chat": "the preview belongs to another chat",
    "limit_reached": "the team limit was reached",
    "workspace_unavailable": "the expert workspace was unavailable",
    "expert_moved": "the expert changed after preview",
    "template_gone": "the expert template is no longer available",
    "unexpected_failure": "the change could not be applied",
}


def _batch_message(results: list[ExpertChangeBatchResult]) -> str:
    applied = [result for result in results if result.outcome == "applied"]
    already = [result for result in results if result.outcome == "already_applied"]
    failed = [result for result in results if result.outcome == "failed"]
    parts: list[str] = []
    if applied:
        parts.append(
            "Team ready: "
            + ", ".join(
                result.expert.name for result in applied if result.expert is not None
            )
            + "."
        )
    if already:
        parts.append(f"{len(already)} approved change(s) were already applied.")
    parts.extend(
        f"{result.proposed_name or 'An approved expert'} was not added: "
        f"{_REASON_COPY.get(result.reason or '', 'the change could not be applied')}."
        for result in failed
    )
    return " ".join(parts) or "No team changes were applied."
