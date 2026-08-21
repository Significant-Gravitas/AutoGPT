"""Server-side storage and application for pending hire/raise proposals.

Mirrors ``soul_proposal`` for team changes: the preview tools write nothing and
park the exact proposal in Redis under a one-time ``confirmation_id``, and
``confirm_expert_change`` loads it bound to the same Autopilot session,
consumes it single-use, and applies it.
"""

import logging

from pydantic import BaseModel, ValidationError

from backend.api.features.experts.errors import (
    ACTIVE_EXPERT_LIMIT,
    LIFETIME_RAISED_EXPERT_LIMIT,
    ExpertHireUnavailableError,
    ExpertLimitExceededError,
    ExpertTemplateNotFoundError,
    RaisedExpertLifetimeLimitExceededError,
)
from backend.api.features.experts.models import Expert, HireResult, RaiseResult
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.data.redis_client import AsyncRedisClient
from backend.util.exceptions import (
    ExpertNotFoundError,
    ExpertPrivateTenancyNotFoundError,
)

from .models import (
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertChangeKind,
    ExpertChangePreview,
    ExpertSummary,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

PROPOSAL_TTL_SECONDS = 15 * 60
_PROPOSAL_KEY_PREFIX = "copilot:expert_change_proposal:"
_LOG_ID_PREFIX_LENGTH = 12


class ExpertChangeProposal(BaseModel):
    """The exact pending team change stored between preview and confirm."""

    user_id: str
    session_id: str
    preview: ExpertChangePreview


def proposal_key(confirmation_id: str) -> str:
    return f"{_PROPOSAL_KEY_PREFIX}{confirmation_id}"


def _stale_preview_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "This confirmation_id is unknown, expired, or already used. "
            "Call hire_expert or raise_expert again for a fresh preview."
        ),
        session_id=session_id,
    )


async def store_proposal(
    redis: AsyncRedisClient,
    confirmation_id: str,
    proposal: ExpertChangeProposal,
) -> None:
    await redis.setex(
        proposal_key(confirmation_id),
        PROPOSAL_TTL_SECONDS,
        proposal.model_dump_json(),
    )


def autopilot_session_guard(
    user_id: str | None, session: ChatSession
) -> ErrorResponse | None:
    """Team changes belong to Autopilot — an expert cannot staff its own team."""
    if not user_id:
        return ErrorResponse(
            message="Please sign in to change the team.",
            session_id=session.session_id,
        )
    if session.expert_id:
        return ErrorResponse(
            message=(
                "Only the user can change the team, and only from the "
                "Autopilot chat. Tell them what you'd add and let them do it "
                "there."
            ),
            session_id=session.session_id,
        )
    return None


async def capacity_error(
    user_id: str,
    session_id: str,
    kind: ExpertChangeKind,
) -> ErrorResponse | None:
    """Refuse at preview time when the team is already full.

    Advisory only — the creation transaction re-enforces both caps. Checking
    here keeps the user from approving a hire that can never land.
    """
    if await experts_db().count_active_experts(user_id) >= ACTIVE_EXPERT_LIMIT:
        return _limit_error(
            f"The team is already at its limit of {ACTIVE_EXPERT_LIMIT} active "
            "experts. Ask the user to archive someone first.",
            session_id,
        )
    if kind == "raise":
        raised = await experts_db().count_raised_experts(user_id)
        if raised >= LIFETIME_RAISED_EXPERT_LIMIT:
            return _limit_error(
                "This account has raised its lifetime maximum of "
                f"{LIFETIME_RAISED_EXPERT_LIMIT} experts. Hiring from the "
                "roster still works.",
                session_id,
            )
    return None


async def load_bound_proposal(
    redis: AsyncRedisClient,
    confirmation_id: str,
    user_id: str,
    session: ChatSession,
) -> ExpertChangeProposal | ErrorResponse:
    key = proposal_key(confirmation_id)
    raw = await redis.get(key)
    if raw is None:
        return _stale_preview_error(session.session_id)

    try:
        proposal = ExpertChangeProposal.model_validate_json(raw)
    except ValidationError:
        await redis.delete(key)
        logger.warning(
            "Discarding malformed expert-change proposal for user %s",
            user_id[:_LOG_ID_PREFIX_LENGTH],
        )
        return _stale_preview_error(session.session_id)

    if proposal.user_id != user_id or proposal.session_id != session.session_id:
        return ErrorResponse(
            message="This confirmation_id belongs to a different chat.",
            session_id=session.session_id,
        )

    # GET and DEL are intentionally separate: binding must be checked before
    # consumption so a mismatched caller cannot invalidate a legitimate
    # proposal.
    if await redis.delete(key) == 0:
        return _stale_preview_error(session.session_id)
    return proposal


async def apply_proposal(
    user_id: str,
    session_id: str,
    proposal: ExpertChangeProposal,
) -> ToolResponseBase:
    preview = proposal.preview
    if preview.kind == "hire":
        return await _apply_hire(user_id, session_id, preview)
    return await _apply_raise(user_id, session_id, preview)


async def _apply_hire(
    user_id: str,
    session_id: str,
    preview: ExpertChangePreview,
) -> ToolResponseBase:
    if preview.template_id is None:
        return _apply_failed_error(session_id)
    try:
        result: HireResult = await experts_db().hire_expert(
            user_id,
            preview.template_id,
            preview.name or None,
        )
    except Exception as e:
        return _hire_failure_response(e, session_id)
    return ExpertChangeAppliedResponse(
        message=(
            f"{result.expert.name} is hired and on the team. Tell the user "
            "who joined and what they own."
        ),
        session_id=session_id,
        kind="hire",
        expert=_summary(result.expert),
        failed_workflows=result.failed_preloads,
    )


async def _apply_raise(
    user_id: str,
    session_id: str,
    preview: ExpertChangePreview,
) -> ToolResponseBase:
    try:
        result: RaiseResult = await experts_db().create_raised_expert(
            user_id,
            preview.name,
            preview.role or None,
            preview.voice_preferences or None,
            color=preview.color or None,
            about=preview.about or None,
            boundaries=preview.boundaries or None,
            weekly_budget=preview.weekly_budget,
        )
    except Exception as e:
        return _raise_failure_response(e, session_id)
    return ExpertChangeAppliedResponse(
        message=(
            f"{result.expert.name} is raised and on the team. Tell the user "
            "who joined and what they own."
        ),
        session_id=session_id,
        kind="raise",
        expert=_summary(result.expert),
    )


def _hire_failure_response(error: Exception, session_id: str) -> ErrorResponse:
    """Map the hire path's typed failures to something the user can act on."""
    if isinstance(error, ExpertTemplateNotFoundError):
        return ErrorResponse(
            message=(
                "That expert template no longer exists. List the roster again "
                "and pick a current one."
            ),
            session_id=session_id,
        )
    if isinstance(error, ExpertLimitExceededError):
        return _limit_error(
            f"The team is already at its limit of {error.limit} active "
            "experts. Archive someone before hiring.",
            session_id,
        )
    if isinstance(
        error,
        (
            ExpertHireUnavailableError,
            ExpertPrivateTenancyNotFoundError,
            ExpertNotFoundError,
        ),
    ):
        return ErrorResponse(
            message=(
                "The expert workspace is temporarily unavailable, so nothing "
                "was hired. Try again shortly."
            ),
            session_id=session_id,
        )
    return _unexpected_failure(error, session_id, "hire_expert")


def _raise_failure_response(error: Exception, session_id: str) -> ErrorResponse:
    if isinstance(error, ExpertLimitExceededError):
        return _limit_error(
            f"The team is already at its limit of {error.limit} active "
            "experts. Archive someone before raising a new one.",
            session_id,
        )
    if isinstance(error, RaisedExpertLifetimeLimitExceededError):
        return _limit_error(
            f"This account has raised its lifetime maximum of {error.limit} "
            "experts.",
            session_id,
        )
    return _unexpected_failure(error, session_id, "raise_expert")


def _limit_error(message: str, session_id: str) -> ErrorResponse:
    return ErrorResponse(message=message, session_id=session_id)


def _unexpected_failure(
    error: Exception, session_id: str, tool_name: str
) -> ErrorResponse:
    logger.warning("%s apply failed: %s", tool_name, error, exc_info=True)
    return ErrorResponse(
        message=(
            "Couldn't complete that change, and the proposal has been "
            f"discarded. Call {tool_name} again to re-preview and retry."
        ),
        session_id=session_id,
    )


def _summary(expert: Expert) -> ExpertSummary:
    return ExpertSummary(
        id=expert.id,
        name=expert.name,
        role=expert.role,
        avatar_url=expert.avatar_url,
        color=expert.color,
    )


def _apply_failed_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "That proposal is missing the template it referred to and has "
            "been discarded. Call hire_expert again to re-preview."
        ),
        session_id=session_id,
    )
