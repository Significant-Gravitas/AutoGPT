"""Preview raising a brand-new expert from a charter (never writes).

Step 1 of the confirm-gated raise flow. The parameters mirror the raise API
(``name`` / ``role`` / ``color`` / ``about`` / ``voice_preferences`` /
``weekly_budget``) plus ``boundaries``, so the model has to collect a full
charter — what the expert owns, what good looks like, and where they stop —
before the user is ever asked to approve.
"""

import logging
import uuid
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.api.features.experts.models import (
    EXPERT_COLOR_MAX_LENGTH,
    EXPERT_NAME_MAX_LENGTH,
    WEEKLY_BUDGET_MAX_CREDITS,
    Expert,
    ExpertSoulFieldsPatch,
)
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

# The accent palette the raise flow's color step offers, as opaque design
# tokens the client maps to swatches. Kept in the tool schema as an enum so
# the model picks a real swatch instead of inventing CSS.
COLOR_TOKENS = [
    "rose-300",
    "red-300",
    "orange-300",
    "amber-300",
    "yellow-300",
    "lime-300",
    "green-300",
    "emerald-300",
    "teal-300",
    "cyan-300",
    "sky-300",
    "blue-300",
    "indigo-300",
    "violet-300",
    "fuchsia-300",
]


logger = logging.getLogger(__name__)


class _RaiseParams(BaseModel):
    """Non-Soul raise fields, validated with the same bounds as the API."""

    name: str = Field(min_length=1, max_length=EXPERT_NAME_MAX_LENGTH)
    role: str = Field(default="", max_length=EXPERT_NAME_MAX_LENGTH)
    color: str = Field(default="", max_length=EXPERT_COLOR_MAX_LENGTH)
    weekly_budget: int | None = Field(default=None, ge=0, le=WEEKLY_BUDGET_MAX_CREDITS)


class RaiseExpertTool(BaseTool):
    """Propose raising a new expert written from scratch."""

    @property
    def name(self) -> str:
        return "raise_expert"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Propose a brand-new expert when no roster template fits. Give "
            "them a personal first name (the role field carries the job "
            "title) and an accent color, then write their charter: what they "
            "own, what good looks like, and where they stop (always fill "
            "boundaries — an expert without them oversteps). Never writes: "
            "returns the proposed expert plus a one-time confirmation_id. "
            "Read the FULL charter back — name, role, color, what they own, "
            "where they stop, voice, weekly budget — and only after the user "
            "approves, call confirm_expert_change with that id."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "A personal first name the user will call them — "
                        "like a teammate's name, never a job title. The "
                        "role field carries the title."
                    ),
                },
                "role": {
                    "type": "string",
                    "description": "Short title for what they own.",
                },
                "color": {
                    "type": "string",
                    "enum": COLOR_TOKENS,
                    "description": (
                        "Accent color for their avatar and chat theme. "
                        "Pick one that fits their personality."
                    ),
                },
                "about": {
                    "type": "string",
                    "description": (
                        "Their charter in second person: what they own, how "
                        "they work, what good looks like. Becomes identity."
                    ),
                },
                "boundaries": {
                    "type": "string",
                    "description": (
                        "Where they stop: what they never do, and what they "
                        "escalate instead."
                    ),
                },
                "voice_preferences": {
                    "type": "string",
                    "description": "How they should sound; omit if unknown.",
                },
                "weekly_budget": {
                    "type": "integer",
                    "description": "Weekly credit cap (100 = $1); omit for default.",
                },
            },
            "required": ["name", "about", "boundaries"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        name: str = "",
        role: str = "",
        color: str = "",
        about: str = "",
        boundaries: str = "",
        voice_preferences: str = "",
        weekly_budget: int | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if error := autopilot_session_guard(user_id, session):
            return error
        assert user_id is not None

        color = color.strip()
        if color and color not in COLOR_TOKENS:
            return ErrorResponse(
                message=(
                    "Invalid expert charter — color must be one of: "
                    + ", ".join(COLOR_TOKENS)
                ),
                session_id=session_id,
            )
        try:
            params = _RaiseParams(
                # Collapsed, not just stripped: the roster block in
                # ``expert_context`` renders one line per teammate, so an
                # embedded newline in either field forges extra roster
                # entries that ``escape_prompt_xml_tags`` cannot neutralise.
                name=" ".join(name.split()),
                role=" ".join(role.split()),
                color=color,
                weekly_budget=weekly_budget,
            )
            soul = ExpertSoulFieldsPatch(
                identity=about,
                boundaries=boundaries,
                voice_preferences=voice_preferences,
            )
        except ValidationError as e:
            return ErrorResponse(
                message=f"Invalid expert charter — {_validation_detail(e)}",
                session_id=session_id,
            )
        if not soul.boundaries:
            return ErrorResponse(
                message=(
                    "boundaries is required — say where this expert stops and "
                    "what they escalate instead."
                ),
                session_id=session_id,
            )
        try:
            duplicate = await _active_expert_named(user_id, params.name)
        except _RosterUnavailable as e:
            # Nothing downstream enforces name uniqueness, so a roster read we
            # could not perform must not pass for "no duplicate".
            logger.warning(f"raise_expert duplicate-name check failed: {e}")
            return ErrorResponse(
                message="Could not check the team roster right now. Try again.",
                session_id=session_id,
            )
        if duplicate:
            return ErrorResponse(
                message=(
                    f"An active expert named {duplicate.name} already exists "
                    f"(expert_id: {duplicate.id}, role: {duplicate.role}) — "
                    "do not raise them again. Delegate work to them with "
                    "delegate_to_expert, or change their charter with "
                    "update_expert. Only propose a differently-named expert "
                    "if the user truly wants a second, separate one."
                ),
                session_id=session_id,
            )
        if error := await capacity_error(user_id, session_id, "raise"):
            return error

        preview = ExpertChangePreview(
            kind="raise",
            name=params.name,
            role=params.role,
            color=params.color,
            about=soul.identity or "",
            boundaries=soul.boundaries,
            voice_preferences=soul.voice_preferences or "",
            weekly_budget=params.weekly_budget,
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
                "Nothing created yet. Read the full charter back to the "
                "user — name, role, color, what this expert owns, where "
                "they stop, voice and weekly budget — and only after they "
                "explicitly approve, call confirm_expert_change with this "
                "confirmation_id."
            ),
            session_id=session_id,
            preview=preview,
            confirmation_id=confirmation_id,
        )


class _RosterUnavailable(Exception):
    """The roster could not be read — distinct from "nobody has this name"."""


async def _active_expert_named(user_id: str, name: str) -> Expert | None:
    """The active expert already carrying *name*, or None.

    Only the read is treated as recoverable: a bug in the matching below is
    not a roster outage and propagates instead of being retried forever.
    """
    try:
        experts = await experts_db().list_experts(user_id, with_metrics=False)
    except Exception as e:
        raise _RosterUnavailable(str(e)) from e
    wanted = name.strip().casefold()
    return next(
        (
            expert
            for expert in experts
            if not expert.is_archived and expert.name.strip().casefold() == wanted
        ),
        None,
    )


def _validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in error.errors()
    )
