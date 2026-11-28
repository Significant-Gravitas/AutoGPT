"""Preview hiring an expert off the roster (never writes).

Step 1 of the confirm-gated hire flow: resolve the template, check the team
has room, and park the exact proposal under a one-time ``confirmation_id``.
``confirm_expert_change`` applies it once the user approves.

A hire proposed for a task the user already described carries that task and
is parked twice, plain hire and "hire and start" (see
``expert_handoff_proposal``).
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.api.features.experts.models import EXPERT_NAME_MAX_LENGTH, Expert
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .expert_handoff_proposal import (
    HANDOFF_PARAMETERS,
    park_proposal,
    parse_handoff,
    proposed_message,
)
from .expert_proposal import (
    ExpertChangeProposal,
    autopilot_session_guard,
    capacity_error,
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
            "The user answers on a card, so do not describe it in text — one "
            "short line at most, then wait. Only after they approve, call "
            "tool:confirm_expert_change with the id they approved. For a task "
            "already in the chat, pass handoff_title + handoff_task. Use "
            "tool:raise_expert when no template fits."
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
                **HANDOFF_PARAMETERS,
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
        handoff_title: str = "",
        handoff_task: str = "",
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

        params = _hire_params(name)
        if isinstance(params, str):
            return ErrorResponse(message=params, session_id=session_id)
        handoff = parse_handoff(handoff_title, handoff_task)
        if isinstance(handoff, str):
            return ErrorResponse(message=handoff, session_id=session_id)

        template = await _find_template(template_id.strip())
        if template is None:
            return ErrorResponse(
                message=(
                    f"No expert template with id {template_id.strip()} is on "
                    'the roster. Find one with find_capability(kind="expert").'
                ),
                session_id=session_id,
            )
        if error := await capacity_error(user_id, session_id, "hire"):
            return error

        preview = _preview(template, params.name)
        confirmation_id, handoff_confirmation_id = await park_proposal(
            ExpertChangeProposal(
                user_id=user_id,
                session_id=session_id,
                preview=preview,
                handoff=handoff,
            )
        )
        return ExpertChangeProposedResponse(
            message=proposed_message("hire", handoff is not None),
            session_id=session_id,
            preview=preview,
            confirmation_id=confirmation_id,
            handoff=handoff,
            handoff_confirmation_id=handoff_confirmation_id,
        )


def _hire_params(name: str) -> _HireParams | str:
    """The validated hire arguments, or why they are refused."""
    try:
        return _HireParams(name=" ".join(name.split()))
    except ValidationError as e:
        return f"Invalid hire — {_validation_detail(e)}"


def _preview(template: Expert, name: str) -> ExpertChangePreview:
    return ExpertChangePreview(
        kind="hire",
        name=name or template.name,
        role=template.role,
        tagline=template.tagline or "",
        about=template.identity,
        boundaries=template.boundaries,
        voice_preferences=template.voice_preferences,
        template_id=template.id,
        avatar_url=template.avatar_url,
        color=template.color,
    )


async def _find_template(template_id: str) -> Expert | None:
    templates = await experts_db().list_templates()
    return next((t for t in templates if t.id == template_id), None)


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )
