"""Server-side storage and atomic application for expert Soul proposals."""

import logging

from pydantic import BaseModel, Field, ValidationError

from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.data.redis_client import AsyncRedisClient

from .models import (
    ErrorResponse,
    ExpertSoulUpdatedResponse,
    SoulFieldChange,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

PROPOSAL_TTL_SECONDS = 15 * 60
_PROPOSAL_KEY_PREFIX = "copilot:soul_edit_proposal:"
_LOG_ID_PREFIX_LENGTH = 12


class SoulEditProposal(BaseModel):
    """The exact pending edit stored between preview and confirm."""

    user_id: str
    session_id: str
    expert_id: str
    changes: list[SoulFieldChange] = Field(min_length=1)


def proposal_key(confirmation_id: str) -> str:
    return f"{_PROPOSAL_KEY_PREFIX}{confirmation_id}"


def _stale_preview_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            "This confirmation_id is unknown, expired, or already used. "
            "Call update_expert_soul again for a fresh preview."
        ),
        session_id=session_id,
    )


async def load_bound_proposal(
    redis: AsyncRedisClient,
    confirmation_id: str,
    user_id: str,
    session: ChatSession,
) -> SoulEditProposal | ErrorResponse:
    key = proposal_key(confirmation_id)
    raw = await redis.get(key)
    if raw is None:
        return _stale_preview_error(session.session_id)

    try:
        proposal = SoulEditProposal.model_validate_json(raw)
    except ValidationError:
        await redis.delete(key)
        logger.warning(
            "Discarding malformed soul-edit proposal for user %s",
            user_id[:_LOG_ID_PREFIX_LENGTH],
        )
        return _stale_preview_error(session.session_id)

    if (
        proposal.user_id != user_id
        or proposal.expert_id != session.expert_id
        or proposal.session_id != session.session_id
    ):
        return ErrorResponse(
            message="This confirmation_id belongs to a different chat or expert.",
            session_id=session.session_id,
        )

    # GET and DEL are intentionally separate: binding must be checked before
    # consumption so a mismatched caller cannot invalidate a legitimate proposal.
    if await redis.delete(key) == 0:
        return _stale_preview_error(session.session_id)
    return proposal


async def apply_proposal(
    user_id: str,
    expert_id: str,
    session_id: str,
    proposal: SoulEditProposal,
) -> ToolResponseBase:
    update_args: dict[str, str] = {}
    for change in proposal.changes:
        update_args[change.field] = change.after
        update_args[f"expected_{change.field}"] = change.before

    try:
        applied = await experts_db().update_soul_fields_if_current(
            user_id,
            expert_id,
            **update_args,
        )
    except Exception:
        logger.warning(
            "Soul edit apply failed for user %s",
            user_id[:_LOG_ID_PREFIX_LENGTH],
            exc_info=True,
        )
        return ErrorResponse(
            message=(
                "Couldn't apply the Soul edit — the proposal has been "
                "discarded. Call update_expert_soul again to re-preview "
                "and retry."
            ),
            session_id=session_id,
        )

    if not applied:
        return ErrorResponse(
            message=(
                "The expert is no longer available or its Soul changed since "
                "this preview, so the proposal was discarded. Call "
                "update_expert_soul again to preview the current Soul."
            ),
            session_id=session_id,
        )
    return ExpertSoulUpdatedResponse(
        message="Soul updated. Tell the user exactly what changed.",
        session_id=session_id,
        applied=True,
        changes=proposal.changes,
    )
