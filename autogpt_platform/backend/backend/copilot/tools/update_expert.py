"""Preview a soul edit for an expert already on the team (never writes).

``identity`` and ``boundaries`` are injected into that expert's system prompt
on every later turn, and there is no undo for boundaries the user wrote by
hand — so this is the same class of edit ``update_expert_soul`` already gates,
and it goes through the same preview + ``confirm_expert_change`` flow as
hire/raise. The tool merges the requested edits over the current soul, shows
the result, and parks it under a one-time ``confirmation_id``;
``confirm_expert_change`` writes it through ``update_soul_if_current``, the
team UI's soul-editor write plus a compare-and-set against the soul this
preview read.
"""

import uuid
from typing import Any

from pydantic import ValidationError

from backend.api.features.experts.models import ExpertSoulUpdate
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .expert_proposal import (
    PROPOSAL_TTL_MINUTES,
    ExpertChangeProposal,
    ExpertSoulSnapshot,
    autopilot_session_guard,
    store_proposal,
    user_turn_watermark,
)
from .models import (
    ErrorResponse,
    ExpertChangePreview,
    ExpertChangeProposedResponse,
    ToolResponseBase,
)


class UpdateExpertTool(BaseTool):
    """Propose a soul edit to an expert already on the team."""

    @property
    def name(self) -> str:
        return "update_expert"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Propose an edit to an expert already on the team — rename them "
            "or rewrite their soul: what they own (about), where they stop "
            "(boundaries), how they sound (voice). Pass only the fields that "
            "change; the rest stay as they are. Never writes: returns the "
            "merged soul plus a one-time confirmation_id. Show the user "
            "exactly what would change and, only after they approve, call "
            "confirm_expert_change with that id."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": "The team expert to edit.",
                },
                "name": {
                    "type": "string",
                    "description": (
                        "New personal first name; omit to keep the current one."
                    ),
                },
                "about": {
                    "type": "string",
                    "description": (
                        "Replacement charter in second person: what they "
                        "own, how they work, what good looks like. Omit to "
                        "keep."
                    ),
                },
                "boundaries": {
                    "type": "string",
                    "description": (
                        "Replacement boundaries: what they never do, and "
                        "what they escalate instead. Omit to keep."
                    ),
                },
                "voice_preferences": {
                    "type": "string",
                    "description": (
                        "Replacement voice; empty string clears it. Omit to keep."
                    ),
                },
            },
            "required": ["expert_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        expert_id: str = "",
        name: str | None = None,
        about: str | None = None,
        boundaries: str | None = None,
        voice_preferences: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if error := autopilot_session_guard(user_id, session):
            return error
        assert user_id is not None

        if all(value is None for value in (name, about, boundaries, voice_preferences)):
            return ErrorResponse(
                message="Nothing to change — pass at least one field to edit.",
                session_id=session_id,
            )

        expert = await experts_db().get_expert(
            user_id, expert_id, include_workflows=False
        )
        if expert is None:
            return ErrorResponse(
                message=(
                    "No active expert with that id on the team. Check the "
                    "team context for the right expert_id."
                ),
                session_id=session_id,
            )

        try:
            merged = ExpertSoulUpdate(
                # Collapsed, not just stripped — a newline in the name forges
                # extra lines in the ``<team_context>`` roster block.
                name=" ".join(name.split()) if name is not None else expert.name,
                identity=about if about is not None else expert.identity,
                # Blank keeps the stored value: unlike voice_preferences,
                # boundaries has no documented empty-string clearing, and a
                # raise requires them — an update must not silently drop them.
                boundaries=(
                    boundaries
                    if boundaries is not None and boundaries.strip()
                    else expert.boundaries
                ),
                voice_preferences=(
                    voice_preferences
                    if voice_preferences is not None
                    else expert.voice_preferences
                ),
            )
        except ValidationError as e:
            return ErrorResponse(
                message=f"Invalid soul edit — {_validation_detail(e)}",
                session_id=session_id,
            )

        preview = ExpertChangePreview(
            kind="update",
            name=merged.name,
            role=expert.role,
            about=merged.identity,
            boundaries=merged.boundaries,
            voice_preferences=merged.voice_preferences,
            avatar_url=expert.avatar_url,
            color=expert.color,
        )
        confirmation_id = str(uuid.uuid4())
        await store_proposal(
            await get_redis_async(),
            confirmation_id,
            ExpertChangeProposal(
                user_id=user_id,
                session_id=session_id,
                preview=preview,
                expert_id=expert.id,
                expected_soul=ExpertSoulSnapshot(
                    name=expert.name,
                    identity=expert.identity,
                    voice_preferences=expert.voice_preferences,
                    boundaries=expert.boundaries,
                ),
                user_turn_watermark=user_turn_watermark(session),
            ),
        )
        return ExpertChangeProposedResponse(
            message=(
                f"Nothing changed yet. Show the user exactly how {expert.name} "
                "would be rewritten, including anything this replaces. Only "
                "after they explicitly approve, call confirm_expert_change "
                f"with this confirmation_id; it expires in "
                f"{PROPOSAL_TTL_MINUTES} minutes."
            ),
            session_id=session_id,
            preview=preview,
            confirmation_id=confirmation_id,
        )


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )
