"""Two-step tools for editing the current expert's Soul.

Step 1 (update_expert_soul): compute the before/after diff, store the exact
proposal server-side, and return a one-time confirmation_id — nothing written.
Step 2 (confirm_expert_soul_update): after the user approves the diff, apply
exactly the stored proposal by confirmation_id. New field values are rejected
at this step, the id is single-use, and a Soul that changed since the preview
aborts the apply. Mirrors the memory_forget_search -> memory_forget_confirm
flow in graphiti_forget.py.
"""

import logging
import uuid
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .models import (
    ErrorResponse,
    ExpertSoulUpdatedResponse,
    SoulFieldChange,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

_PLAIN_SESSION_REFUSAL = (
    "I can only edit an expert's Soul inside that expert's chat. Open a thread "
    "with a hired expert to change their identity, voice, or boundaries."
)

_EDITABLE_FIELDS = ("identity", "voice_preferences", "boundaries")

_PROPOSAL_KEY_PREFIX = "copilot:soul_edit_proposal:"
_PROPOSAL_TTL_SECONDS = 15 * 60


class _SoulEditProposal(BaseModel):
    """The exact pending edit stored between preview and confirm."""

    user_id: str
    session_id: str
    expert_id: str
    changes: list[SoulFieldChange]


def _proposal_key(confirmation_id: str) -> str:
    return f"{_PROPOSAL_KEY_PREFIX}{confirmation_id}"


def _soul_snapshot(expert) -> dict[str, str]:
    return {field: getattr(expert, field) for field in _EDITABLE_FIELDS}


class UpdateExpertSoulTool(BaseTool):
    """Preview a Soul edit for the current session's expert (never writes)."""

    @property
    def name(self) -> str:
        return "update_expert_soul"

    @property
    def description(self) -> str:
        return (
            "Propose an edit to this expert's Soul — identity/personality, "
            "voice preferences, or boundaries. Never writes: it returns the "
            "before/after diff plus a one-time confirmation_id. Show the user "
            "the diff and, only after they explicitly approve, call "
            "confirm_expert_soul_update with that confirmation_id."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "identity": {
                    "type": "string",
                    "description": "New identity/personality text; omit to leave unchanged.",
                },
                "voice_preferences": {
                    "type": "string",
                    "description": "New voice/tone preferences; omit to leave unchanged.",
                },
                "boundaries": {
                    "type": "string",
                    "description": "New boundaries; omit to leave unchanged.",
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Please sign in to edit an expert's Soul.",
                session_id=session_id,
            )
        if not session.expert_id:
            return ErrorResponse(message=_PLAIN_SESSION_REFUSAL, session_id=session_id)

        edits = {
            field: kwargs[field]
            for field in _EDITABLE_FIELDS
            if isinstance(kwargs.get(field), str)
        }
        if not edits:
            return ErrorResponse(
                message="Provide at least one of identity, voice_preferences, or boundaries.",
                session_id=session_id,
            )

        expert = await experts_db().get_expert(
            user_id, session.expert_id, include_workflows=False
        )
        if expert is None:
            return ErrorResponse(
                message="This expert no longer exists.",
                session_id=session_id,
            )

        before = _soul_snapshot(expert)
        changes = [
            SoulFieldChange(field=field, before=before[field], after=edits[field])
            for field in _EDITABLE_FIELDS
            if field in edits and edits[field] != before[field]
        ]
        if not changes:
            return ErrorResponse(
                message="Those values match the current Soul — nothing to change.",
                session_id=session_id,
            )

        confirmation_id = str(uuid.uuid4())
        proposal = _SoulEditProposal(
            user_id=user_id,
            session_id=session_id,
            expert_id=session.expert_id,
            changes=changes,
        )
        redis = await get_redis_async()
        await redis.setex(
            _proposal_key(confirmation_id),
            _PROPOSAL_TTL_SECONDS,
            proposal.model_dump_json(),
        )
        return ExpertSoulUpdatedResponse(
            message=(
                "Nothing saved yet. Show the user this before/after diff and ask "
                "them to approve. Only after they explicitly approve, call "
                "confirm_expert_soul_update with this confirmation_id."
            ),
            session_id=session_id,
            applied=False,
            changes=changes,
            confirmation_id=confirmation_id,
        )


class ConfirmExpertSoulUpdateTool(BaseTool):
    """Apply a previewed Soul edit by its one-time confirmation_id."""

    @property
    def name(self) -> str:
        return "confirm_expert_soul_update"

    @property
    def description(self) -> str:
        return (
            "Apply a Soul edit previously proposed by update_expert_soul, after "
            "the user has approved the diff. Takes only the confirmation_id "
            "from that preview and applies the stored proposal exactly as "
            "previewed; the id is single-use."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "confirmation_id": {
                    "type": "string",
                    "description": "The confirmation_id returned by update_expert_soul.",
                },
            },
            "required": ["confirmation_id"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        confirmation_id: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Please sign in to edit an expert's Soul.",
                session_id=session_id,
            )
        if not session.expert_id:
            return ErrorResponse(message=_PLAIN_SESSION_REFUSAL, session_id=session_id)

        if any(field in kwargs for field in _EDITABLE_FIELDS):
            return ErrorResponse(
                message=(
                    "confirm_expert_soul_update applies exactly the previewed "
                    "proposal and does not accept field values. Call "
                    "update_expert_soul to propose a different edit."
                ),
                session_id=session_id,
            )
        if not confirmation_id:
            return ErrorResponse(
                message="A confirmation_id from update_expert_soul is required.",
                session_id=session_id,
            )

        stale_preview_error = ErrorResponse(
            message=(
                "This confirmation_id is unknown, expired, or already used. "
                "Call update_expert_soul again for a fresh preview."
            ),
            session_id=session_id,
        )
        redis = await get_redis_async()
        key = _proposal_key(confirmation_id)
        raw = await redis.get(key)
        # DEL's return value arbitrates concurrent confirms: only the caller
        # that actually removed the key may apply, making the id single-use.
        consumed = await redis.delete(key)
        if raw is None or consumed == 0:
            return stale_preview_error
        try:
            proposal = _SoulEditProposal.model_validate_json(raw)
        except ValidationError:
            logger.warning(
                "Discarding malformed soul-edit proposal for user %s", user_id[:12]
            )
            return stale_preview_error

        if (
            proposal.user_id != user_id
            or proposal.expert_id != session.expert_id
            or proposal.session_id != session_id
        ):
            return ErrorResponse(
                message="This confirmation_id belongs to a different chat or expert.",
                session_id=session_id,
            )

        expert = await experts_db().get_expert(
            user_id, session.expert_id, include_workflows=False
        )
        if expert is None:
            return ErrorResponse(
                message="This expert no longer exists.",
                session_id=session_id,
            )
        current = _soul_snapshot(expert)
        if any(current[change.field] != change.before for change in proposal.changes):
            return ErrorResponse(
                message=(
                    "The expert's Soul changed since this preview, so the "
                    "proposal was discarded. Call update_expert_soul again to "
                    "preview against the current Soul."
                ),
                session_id=session_id,
            )

        await experts_db().update_soul_fields(
            user_id,
            session.expert_id,
            **{change.field: change.after for change in proposal.changes},
        )
        return ExpertSoulUpdatedResponse(
            message="Soul updated. Tell the user exactly what changed.",
            session_id=session_id,
            applied=True,
            changes=proposal.changes,
        )
