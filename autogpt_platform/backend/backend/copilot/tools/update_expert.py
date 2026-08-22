"""Edit an existing expert's soul, applied immediately.

Unlike hire/raise — which create someone new and go through the
preview + ``confirm_expert_change`` gate — an update edits a teammate the
user already owns and can re-edit just as cheaply, so the user's ask IS the
confirmation. The tool merges the requested edits over the current soul and
writes through the same ``update_soul`` path the team UI's soul editor uses.
"""

from typing import Any

from pydantic import ValidationError

from backend.api.features.experts.models import ExpertSoulUpdate
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.util.exceptions import ExpertNotFoundError

from .base import BaseTool
from .expert_proposal import autopilot_session_guard
from .models import (
    ErrorResponse,
    ExpertChangeAppliedResponse,
    ExpertSummary,
    ToolResponseBase,
)


class UpdateExpertTool(BaseTool):
    """Apply a soul edit to an expert already on the team."""

    @property
    def name(self) -> str:
        return "update_expert"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Edit an expert already on the team — rename them or rewrite "
            "their soul: what they own (about), where they stop "
            "(boundaries), how they sound (voice). Pass only the fields "
            "that change; the rest stay as they are. Applies immediately: "
            "when the user asks for a change, call this once with it — do "
            "not ask them to confirm first. Afterwards tell them exactly "
            "what changed."
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
                name=name if name is not None else expert.name,
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

        try:
            updated = await experts_db().update_soul(user_id, expert.id, merged)
        except ExpertNotFoundError:
            return ErrorResponse(
                message=(
                    "That expert vanished mid-edit — they may have just been "
                    "archived. Nothing was changed."
                ),
                session_id=session_id,
            )
        return ExpertChangeAppliedResponse(
            message=f"{updated.name} is updated. Tell the user exactly what changed.",
            session_id=session_id,
            kind="update",
            expert=ExpertSummary(
                id=updated.id,
                name=updated.name,
                role=updated.role,
                avatar_url=updated.avatar_url,
                color=updated.color,
            ),
        )


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )
