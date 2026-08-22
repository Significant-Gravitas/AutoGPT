"""Server-side storage and application for pending team-change proposals.

Mirrors ``soul_proposal`` for team changes: the preview tools (``hire_expert``,
``raise_expert``, ``update_expert``) write nothing and park the exact proposal
in Redis under a one-time ``confirmation_id``, and ``confirm_expert_change``
loads it bound to the same Autopilot session, consumes it single-use, and
applies it.
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
from backend.api.features.experts.models import (
    Expert,
    ExpertSoulUpdate,
    HireResult,
    RaiseResult,
)
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
PROPOSAL_TTL_MINUTES = PROPOSAL_TTL_SECONDS // 60
_PROPOSAL_KEY_PREFIX = "copilot:expert_change_proposal:"
# Tombstone written when a proposal is consumed, so a repeated "yes" can be
# answered with "already done" instead of the same message an expired
# preview gets.
_CONSUMED_KEY_PREFIX = "copilot:expert_change_consumed:"
_LOG_ID_PREFIX_LENGTH = 12
_PREVIEW_TOOLS = "hire_expert, raise_expert or update_expert"


class ExpertChangeProposal(BaseModel):
    """The exact pending team change stored between preview and confirm."""

    user_id: str
    session_id: str
    preview: ExpertChangePreview
    # Only set for ``kind == "update"``: the teammate the edit rewrites.
    expert_id: str | None = None


def proposal_key(confirmation_id: str) -> str:
    return f"{_PROPOSAL_KEY_PREFIX}{confirmation_id}"


def _consumed_key(confirmation_id: str) -> str:
    return f"{_CONSUMED_KEY_PREFIX}{confirmation_id}"


def _stale_preview_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "This confirmation_id is unknown or has expired — previews last "
            f"{PROPOSAL_TTL_MINUTES} minutes. Call {_PREVIEW_TOOLS} again for "
            "a fresh preview."
        ),
        session_id=session_id,
    )


def _already_confirmed_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "You already confirmed this change, so there is nothing left to "
            "apply — tell the user it is done. Call "
            f"{_PREVIEW_TOOLS} for a fresh preview only if they want another "
            "change, or if that earlier confirmation reported an error."
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
    """Team changes belong to the user, typing in their own Autopilot chat.

    Two things have to hold, and "no ``expert_id``" only proves the first:
    an expert must not staff its own team, AND a human must actually be
    driving the conversation. A session an ``AutoPilotBlock`` opened inside
    a graph run carries no ``expert_id`` either, and its prompt can be
    assembled from data the user never read — so it is gated on the
    positive ``interactive`` origin instead.
    """
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
    if session.metadata.origin != "interactive":
        return ErrorResponse(
            message=(
                "This session was started by an automation, not by the user, "
                "so it cannot hire, raise, or edit a teammate. Report what "
                "the team would need and let the user make the change in "
                "their own Autopilot chat."
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
        if await redis.get(_consumed_key(confirmation_id)) is not None:
            return _already_confirmed_error(session.session_id)
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
    await redis.setex(_consumed_key(confirmation_id), PROPOSAL_TTL_SECONDS, "1")
    return proposal


async def apply_proposal(
    user_id: str,
    session_id: str,
    proposal: ExpertChangeProposal,
) -> ToolResponseBase:
    preview = proposal.preview
    if preview.kind == "hire":
        return await _apply_hire(user_id, session_id, preview)
    if preview.kind == "raise":
        return await _apply_raise(user_id, session_id, preview)
    if preview.kind == "update":
        return await _apply_update(user_id, session_id, proposal)
    logger.error(
        "apply_proposal received unsupported preview kind %r for user %s",
        preview.kind,
        user_id[:_LOG_ID_PREFIX_LENGTH],
    )
    return ErrorResponse(
        message=(
            "This proposal kind is not supported by confirm_expert_change. "
            f"Call {_PREVIEW_TOOLS} again for a fresh preview."
        ),
        session_id=session_id,
    )


async def _apply_hire(
    user_id: str,
    session_id: str,
    preview: ExpertChangePreview,
) -> ToolResponseBase:
    if preview.template_id is None:
        return _discarded_proposal_error(session_id, "template", "hire_expert")
    try:
        result: HireResult = await experts_db().hire_expert(
            user_id,
            preview.template_id,
            preview.name or None,
        )
    except Exception as e:
        return _hire_failure_response(e, session_id)
    return ExpertChangeAppliedResponse(
        message=_hire_message(result.expert.name, result.failed_preloads),
        session_id=session_id,
        kind="hire",
        expert=_summary(result.expert),
        failed_workflows=result.failed_preloads,
    )


def _hire_message(name: str, failed_workflows: list[str]) -> str:
    """A hire whose workflows didn't install must not read as a clean one.

    The expert exists either way, but until the listed workflows are added
    back it cannot do the part of the job they carried — so the model is
    told to name them rather than announce an unqualified success.
    """
    if not failed_workflows:
        return f"{name} is hired and on the team. Tell the user who joined and what they own."
    workflows = ", ".join(failed_workflows)
    return (
        f"{name} joined the team, but {len(failed_workflows)} of their "
        f"workflows could not be installed: {workflows}. Tell the user who "
        "joined, name the workflows that failed, and say those need to be "
        f"added from {name}'s team page before that part of the job can run."
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


async def _apply_update(
    user_id: str,
    session_id: str,
    proposal: ExpertChangeProposal,
) -> ToolResponseBase:
    """Write the soul edit previewed by ``update_expert``.

    The preview already merged the requested fields over the stored soul and
    validated the result, so this writes the previewed values verbatim — the
    user approved exactly this text.
    """
    preview = proposal.preview
    if proposal.expert_id is None:
        return _discarded_proposal_error(session_id, "expert", "update_expert")
    try:
        updated = await experts_db().update_soul(
            user_id,
            proposal.expert_id,
            ExpertSoulUpdate(
                name=preview.name,
                identity=preview.about,
                boundaries=preview.boundaries,
                voice_preferences=preview.voice_preferences,
            ),
        )
    except ExpertNotFoundError:
        return ErrorResponse(
            message=(
                "That expert vanished between the preview and the "
                "confirmation — they may have just been archived. Nothing "
                "was changed."
            ),
            session_id=session_id,
        )
    except Exception as e:
        return _unexpected_failure(e, session_id, "update_expert")
    return ExpertChangeAppliedResponse(
        message=f"{updated.name} is updated. Tell the user exactly what changed.",
        session_id=session_id,
        kind="update",
        expert=_summary(updated),
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


def _discarded_proposal_error(
    session_id: str, missing: str, tool_name: str
) -> ErrorResponse:
    return ErrorResponse(
        message=(
            f"That proposal is missing the {missing} it referred to and has "
            f"been discarded. Call {tool_name} again to re-preview."
        ),
        session_id=session_id,
    )
