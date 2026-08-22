"""Preview hiring an expert off the roster (never writes).

Step 1 of the confirm-gated hire flow: resolve the template, check the team
has room, and park the exact proposal under a one-time ``confirmation_id``.
``confirm_expert_change`` applies it once the user approves.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.api.features.experts.models import EXPERT_NAME_MAX_LENGTH, Expert
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .expert_proposal import (
    ExpertChangeProposal,
    autopilot_session_guard,
    capacity_error,
    store_proposal,
    user_turn_watermark,
)
from .models import (
    ErrorResponse,
    ExpertChangePreview,
    ExpertChangeProposedResponse,
    ToolResponseBase,
)


class _HireParams(BaseModel):
    """The rename field, bounded the same way as raise_expert's ``name``."""

    name: str = Field(default="", max_length=EXPERT_NAME_MAX_LENGTH)


class HireExpertTool(BaseTool):
    """Propose hiring a roster expert for the user's team."""

    @property
    def name(self) -> str:
        return "hire_expert"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Propose hiring a ready-made expert from the roster. Never "
            "writes: returns who would join plus a one-time confirmation_id. "
            "Show the user who they'd hire and, only after they approve, "
            "call confirm_expert_change with that id. Use raise_expert when "
            "no template fits."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": ("Roster template to hire; never invent an id."),
                },
                "name": {
                    "type": "string",
                    "description": "Rename the hire; omit to keep the template name.",
                },
            },
            "required": ["template_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        template_id: str = "",
        name: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if error := autopilot_session_guard(user_id, session):
            return error
        assert user_id is not None

        if not template_id.strip():
            return ErrorResponse(
                message="template_id is required.", session_id=session_id
            )

        try:
            params = _HireParams(name=" ".join(name.split()))
        except ValidationError as e:
            return ErrorResponse(
                message=f"Invalid expert name — {_validation_detail(e)}",
                session_id=session_id,
            )

        template = await _find_template(template_id.strip())
        if template is None:
            return ErrorResponse(
                message=(
                    f"No expert template with id {template_id.strip()} is on "
                    "the roster. List the roster and pick a current one."
                ),
                session_id=session_id,
            )
        if error := await capacity_error(user_id, session_id, "hire"):
            return error

        preview = ExpertChangePreview(
            kind="hire",
            name=params.name or template.name,
            role=template.role,
            about=template.identity,
            boundaries=template.boundaries,
            voice_preferences=template.voice_preferences,
            template_id=template.id,
            avatar_url=template.avatar_url,
            color=template.color,
        )
        confirmation_id = str(uuid.uuid4())
        await store_proposal(
            await get_redis_async(),
            confirmation_id,
            ExpertChangeProposal(
                user_id=user_id,
                session_id=session_id,
                preview=preview,
                user_turn_watermark=user_turn_watermark(session),
            ),
        )
        return ExpertChangeProposedResponse(
            message=(
                "Nothing hired yet. Show the user who would join, what they "
                "own, and that the hire draws on the shared weekly budget. "
                "Only after they explicitly approve, call "
                "confirm_expert_change with this confirmation_id."
            ),
            session_id=session_id,
            preview=preview,
            confirmation_id=confirmation_id,
        )


async def _find_template(template_id: str) -> Expert | None:
    templates = await experts_db().list_templates()
    return next((t for t in templates if t.id == template_id), None)


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )
