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
    ExpertWriteNotReadableError,
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


class ExpertSoulSnapshot(BaseModel):
    """The stored soul exactly as ``update_expert`` read it at preview time.

    Compared field-by-field in the apply write so an edit the user made
    elsewhere between preview and confirm refuses instead of being reverted.
    Raw column values, not ``ExpertSoulUpdate`` — validating them again here
    would normalise the snapshot away from what the row actually holds.
    """

    name: str
    identity: str
    voice_preferences: str
    boundaries: str


class ExpertChangeProposal(BaseModel):
    """The exact pending team change stored between preview and confirm."""

    user_id: str
    session_id: str
    preview: ExpertChangePreview
    # Only set for ``kind == "update"``: the teammate the edit rewrites.
    expert_id: str | None = None
    # Only set for ``kind == "update"``: see ``ExpertSoulSnapshot``.
    expected_soul: ExpertSoulSnapshot | None = None
    # The human turn this preview answered — see ``user_turn_watermark``.
    # ``None`` only for a proposal parked by code that predates the field.
    # An unknown watermark cannot prove the user answered anything, so the
    # gate refuses it rather than reading it as "answered at -1", which any
    # session with one sequenced user message would clear.
    user_turn_watermark: int | None = None


def proposal_key(confirmation_id: str) -> str:
    return f"{_PROPOSAL_KEY_PREFIX}{confirmation_id}"


def _consumed_key(confirmation_id: str) -> str:
    return f"{_CONSUMED_KEY_PREFIX}{confirmation_id}"


def _stale_preview_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "This confirmation_id is unknown or has expired — previews last "
            f"{PROPOSAL_TTL_MINUTES} minutes. If you already confirmed it "
            "earlier in this conversation, that change is APPLIED and the "
            "expert exists — call list_team to check before doing anything "
            f"else. Only call {_PREVIEW_TOOLS} again for a genuinely new "
            "change."
        ),
        session_id=session_id,
    )


def _unapproved_preview_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "The user has not answered this preview yet, so there is nothing "
            "to confirm. Read the change back to them and call "
            "confirm_expert_change only after they reply approving it."
        ),
        session_id=session_id,
    )


def _unwatermarked_preview_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "This preview was created before the approval check and carries "
            "no record of the turn it answered, so it cannot be confirmed. "
            f"Call {_PREVIEW_TOOLS} again for a fresh preview and confirm "
            "that one after the user approves it."
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


def user_turn_watermark(session: ChatSession) -> int:
    """Sequence of the newest human-authored message in this session.

    The preview stores it and ``load_bound_proposal`` demands a strictly
    higher one, so the approval the confirm claims to act on has to be a real
    user turn. Without it the same assistant turn that previewed a change can
    call ``confirm_expert_change`` on the id it just received, and the whole
    preview/confirm seam collapses into one uninterrupted model decision.
    """
    return max(
        (
            message.sequence
            for message in session.messages
            if message.role == "user" and message.sequence is not None
        ),
        default=-1,
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
    an expert must not staff its own team, AND the conversation has to be one
    a human drives. A session an ``AutoPilotBlock`` opened inside a graph run
    carries no ``expert_id`` either, and its prompt can be assembled from data
    the user never read — so it is gated on the positive ``interactive``
    origin instead, and every machine entry point (block, sub-session,
    scheduled turn) stamps ``automation`` at creation.

    ``origin`` is a property of the session, not of the invocation, so it
    cannot by itself prove a human wrote *this* turn — a scheduled follow-up
    fired into a chat the user already owns still reads as interactive. The
    user-turn watermark in ``load_bound_proposal`` is what closes that gap at
    the confirm step.
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
    # Positive match, so a legacy ``None`` (a session persisted before
    # ``origin`` existed) is refused too: an unknown origin cannot prove a
    # human is here, and the cost is a chat older than this deploy needing a
    # new one before it can staff. The block resume path takes the opposite
    # side of the same unknown on purpose — see ``blocks/autopilot.py``.
    if session.metadata.origin != "interactive":
        return ErrorResponse(
            message=(
                "This session was started by an automation, or predates the "
                "check that tells them apart, so it cannot hire, raise, or "
                "edit a teammate. Report what the team would need and let the "
                "user make the change in a new Autopilot chat."
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

    # Checked before the DEL below so a premature confirm doesn't burn the
    # proposal the user is about to approve for real.
    if proposal.user_turn_watermark is None:
        return _unwatermarked_preview_error(session.session_id)
    if user_turn_watermark(session) <= proposal.user_turn_watermark:
        return _unapproved_preview_error(session.session_id)

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

    The preview merged the requested fields over the stored soul, so this
    writes the whole soul back — which would silently revert an edit made
    elsewhere in the meantime. Compare-and-set against the snapshot the
    preview read (mirroring ``soul_proposal``) so a soul that moved refuses
    instead.
    """
    preview = proposal.preview
    if proposal.expert_id is None:
        return _discarded_proposal_error(session_id, "expert", "update_expert")
    if proposal.expected_soul is None:
        return _discarded_proposal_error(session_id, "soul snapshot", "update_expert")
    expected = proposal.expected_soul
    try:
        updated = await experts_db().update_soul_if_current(
            user_id,
            proposal.expert_id,
            ExpertSoulUpdate(
                name=preview.name,
                identity=preview.about,
                boundaries=preview.boundaries,
                voice_preferences=preview.voice_preferences,
            ),
            expected_name=expected.name,
            expected_identity=expected.identity,
            expected_voice_preferences=expected.voice_preferences,
            expected_boundaries=expected.boundaries,
        )
    except ExpertNotFoundError:
        return _stale_expert_error(session_id)
    except ExpertWriteNotReadableError:
        return _applied_but_unreadable_error(session_id, preview.name)
    except Exception as e:
        return _unexpected_failure(e, session_id, "update_expert")
    if updated is None:
        return _stale_expert_error(session_id)
    return ExpertChangeAppliedResponse(
        message=f"{updated.name} is updated. Tell the user exactly what changed.",
        session_id=session_id,
        kind="update",
        expert=_summary(updated),
    )


def _stale_expert_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "That expert is gone or was edited somewhere else since this "
            "preview, so nothing was changed. Call update_expert again to "
            "preview the current version."
        ),
        session_id=session_id,
    )


def _applied_but_unreadable_error(session_id: str, name: str) -> ErrorResponse:
    """The edit committed, then the teammate disappeared before the read-back.

    Never route this to :func:`_stale_expert_error`: the change did land, so
    telling the model to re-preview would have it re-apply an edit that is
    already saved, or announce it was dropped when it was not.
    """
    return ErrorResponse(
        message=(
            f"The edit to {name} was saved, but they were removed from the "
            "team before it could be read back. Do not re-preview or retry: "
            "tell the user the change applied and that the teammate is no "
            "longer on the team."
        ),
        session_id=session_id,
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
