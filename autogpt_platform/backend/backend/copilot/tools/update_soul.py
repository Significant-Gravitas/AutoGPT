"""Tool for editing the current expert's Soul (identity / voice / boundaries)."""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db

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


class UpdateExpertSoulTool(BaseTool):
    """Patch identity/voice/boundaries for the current session's expert."""

    @property
    def name(self) -> str:
        return "update_expert_soul"

    @property
    def description(self) -> str:
        return (
            "Edit this expert's Soul — its identity/personality, voice "
            "preferences, or boundaries. Only for the expert whose chat this is, "
            "and changes affect every future session. Requires confirm=true; "
            "without it nothing is written and you must show the user the "
            "proposed before/after and ask them to confirm first."
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
                "confirm": {
                    "type": "boolean",
                    "description": (
                        "Must be true to apply. When false or omitted the tool "
                        "returns the proposed diff without writing anything."
                    ),
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

        before = {
            "identity": expert.identity,
            "voice_preferences": expert.voice_preferences,
            "boundaries": expert.boundaries,
        }
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

        if kwargs.get("confirm") is not True:
            return ExpertSoulUpdatedResponse(
                message=(
                    "Not saved yet. Show the user this before/after and ask them "
                    "to confirm, then call again with confirm=true."
                ),
                session_id=session_id,
                applied=False,
                changes=changes,
            )

        await experts_db().update_soul_fields(
            user_id,
            session.expert_id,
            **{change.field: change.after for change in changes},
        )
        return ExpertSoulUpdatedResponse(
            message="Soul updated. Tell the user exactly what changed.",
            session_id=session_id,
            applied=True,
            changes=changes,
        )
