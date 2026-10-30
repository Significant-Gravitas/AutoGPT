"""Preview hiring an expert off the roster (never writes).

Step 1 of the confirm-gated hire flow: resolve the template, check the team
has room, and park the exact proposal under a one-time ``confirmation_id``.
``confirm_expert_change`` applies it once the user approves.

A hire proposed for a task the user already described carries that task, and
is parked twice: the plain hire under ``confirmation_id`` and "hire and
start" under ``handoff_confirmation_id``. The card asks one question, Hire
and start or Not now, so the user's one click both hires and starts the work
(see ``hire_handoff``); the plain id stays for a user who asks in chat to
hire without starting the task.
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
    ExpertHandoffTask,
    ToolResponseBase,
)

# Long enough to name a task the way the user would, short enough to sit in
# a button beside the expert's name.
HANDOFF_TITLE_MAX_LENGTH = 80
HANDOFF_TASK_MAX_LENGTH = 4000


class _HireParams(BaseModel):
    """The rename field, bounded the same way as raise_expert's ``name``,
    and the task the hire is for."""

    name: str = Field(default="", max_length=EXPERT_NAME_MAX_LENGTH)
    handoff_title: str = Field(default="", max_length=HANDOFF_TITLE_MAX_LENGTH)
    handoff_task: str = Field(default="", max_length=HANDOFF_TASK_MAX_LENGTH)

    def handoff(self) -> ExpertHandoffTask | None:
        if not self.handoff_task:
            return None
        return ExpertHandoffTask(title=self.handoff_title, brief=self.handoff_task)


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
                "handoff_title": {
                    "type": "string",
                    "description": "That task in a few words, for the card.",
                },
                "handoff_task": {
                    "type": "string",
                    "description": "The task, written for the expert.",
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

        params = _hire_params(name, handoff_title, handoff_task)
        if isinstance(params, str):
            return ErrorResponse(message=params, session_id=session_id)

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
        handoff = params.handoff()
        confirmation_id, handoff_confirmation_id = await _park(
            ExpertChangeProposal(
                user_id=user_id,
                session_id=session_id,
                preview=preview,
                user_turn_watermark=user_turn_watermark(session),
                handoff=handoff,
            )
        )
        return ExpertChangeProposedResponse(
            message=_proposed_message(handoff is not None),
            session_id=session_id,
            preview=preview,
            confirmation_id=confirmation_id,
            handoff=handoff,
            handoff_confirmation_id=handoff_confirmation_id,
        )


def _hire_params(name: str, handoff_title: str, handoff_task: str) -> _HireParams | str:
    """The validated hire arguments, or why they are refused."""
    try:
        params = _HireParams(
            name=" ".join(name.split()),
            handoff_title=" ".join(handoff_title.split()),
            handoff_task=handoff_task.strip(),
        )
    except ValidationError as e:
        return f"Invalid hire — {_validation_detail(e)}"
    if bool(params.handoff_title) != bool(params.handoff_task):
        return "Pass handoff_title and handoff_task together, or neither."
    return params


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


async def _park(proposal: ExpertChangeProposal) -> tuple[str, str | None]:
    """Park *proposal* under a fresh id and, when it carries a task, its
    "hire and start" twin under a second; returns both ids."""
    redis = await get_redis_async()
    confirmation_id = str(uuid.uuid4())
    if proposal.handoff is None:
        await store_proposal(redis, confirmation_id, proposal)
        return confirmation_id, None
    handoff_confirmation_id = str(uuid.uuid4())
    await store_proposal(
        redis,
        confirmation_id,
        proposal.model_copy(
            update={"sibling_confirmation_id": handoff_confirmation_id}
        ),
    )
    await store_proposal(
        redis,
        handoff_confirmation_id,
        proposal.model_copy(
            update={"hand_off": True, "sibling_confirmation_id": confirmation_id}
        ),
    )
    return confirmation_id, handoff_confirmation_id


def _proposed_message(with_handoff: bool) -> str:
    if not with_handoff:
        return (
            "Nothing hired yet. The user is looking at this expert on a "
            "card with Approve and Decline buttons — do not describe "
            "them in text. Reply with one short line at most and wait. "
            "Only after they explicitly approve, call "
            "tool:confirm_expert_change with this confirmation_id."
        )
    return (
        "Nothing hired yet. The user is looking at this expert on a card "
        "with two buttons: hire and start the task, or not now — do not "
        "describe them in text. Reply with one short line at most and wait. "
        "If they approve, call tool:confirm_expert_change with "
        "handoff_confirmation_id. Use confirmation_id only if they ask to "
        "hire without starting the task."
    )


async def _find_template(template_id: str) -> Expert | None:
    templates = await experts_db().list_templates()
    return next((t for t in templates if t.id == template_id), None)


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )
