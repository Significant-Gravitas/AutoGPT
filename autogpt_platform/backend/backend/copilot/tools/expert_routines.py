"""Offer, switch on, and switch off an expert's standing work.

An expert arrives with its routines already listed and every one of them off.
Switching one on is a conversation, not a toggle: the template ships a proposal
and the questions it needs answered, and the owner's answers become the prompt
and the cadence that actually run. That is the whole point of the round trip —
what a roster wrote for everybody is never what any one account should run
unattended.
"""

import logging
from typing import Any

from backend.api.features.experts.models import ExpertRoutine
from backend.api.features.experts.routines import (
    RoutineNotFoundError,
    RoutineUnansweredAsksError,
)
from backend.copilot.model import ChatSession
from backend.data.activity_event import ActivityEventDraft
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .expert_scope import resolve_target_expert
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)

_EXPERT_ID_PARAM = {
    "type": "string",
    "description": "Target expert (AutoPilot only; experts act on themselves).",
}


class ExpertRoutinesResponse(ToolResponseBase):
    type: ResponseType = ResponseType.EXPERT_ROUTINES
    expert_id: str
    routines: list[ExpertRoutine]


class ExpertRoutineResponse(ToolResponseBase):
    type: ResponseType = ResponseType.EXPERT_ROUTINE
    expert_id: str
    routine: ExpertRoutine


class ListExpertRoutinesTool(BaseTool):
    @property
    def name(self) -> str:
        return "list_expert_routines"

    @property
    def description(self) -> str:
        return (
            "List this expert's standing work: its recurring routines, which "
            "are switched on, and what each still needs answered. Read before "
            "offering one."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expert_id": _EXPERT_ID_PARAM},
            "required": [],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )
        target = await resolve_target_expert(user_id, session, expert_id)
        if isinstance(target, ErrorResponse):
            return target
        routines = await experts_db().list_routines(user_id, target)
        return ExpertRoutinesResponse(
            message=(
                f"{len(routines)} routine(s); "
                f"{sum(1 for r in routines if r.enabled)} switched on."
            ),
            expert_id=target,
            routines=routines,
            session_id=session_id,
        )


class SetExpertRoutineTool(BaseTool):
    @property
    def name(self) -> str:
        return "set_expert_routine"

    @property
    def description(self) -> str:
        return (
            "Switch one of this expert's routines on or off. ON commits to "
            "what it will do: answer its 'asks' with the user first (never "
            "guess), pass the 'prompt' you rewrote from the proposal and the "
            "'crons' for the time they chose, and show them the wording "
            "before calling."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "routine_id": {
                    "type": "string",
                    "description": "Routine id from list_expert_routines.",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "True switches it on, false switches it off.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The routine as it will run, rewritten from the "
                        "proposal with the user's answers. Required when it "
                        "has unanswered 'asks'."
                    ),
                },
                "crons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "5-field crons in the user's timezone, one per fire "
                        "time. Pass when they name a time; omit for the "
                        "routine's own suggested cadence."
                    ),
                },
                "pin_to_this_chat": {
                    "type": "boolean",
                    "description": (
                        "Run it in THIS chat instead of its own thread. "
                        "Default false, which is almost always right."
                    ),
                },
                "grants_credentials": {
                    "type": "boolean",
                    "description": (
                        "Let it use the expert's connected services. Default "
                        "false: without it the routine researches and drafts "
                        "but touches nothing outside the platform. True only "
                        "when the user agreed to THIS routine using their "
                        "accounts."
                    ),
                },
                "expert_id": _EXPERT_ID_PARAM,
            },
            "required": ["routine_id", "enabled"],
        }

    def activity_event(
        self,
        session: ChatSession,
        result: ToolResponseBase,
        **kwargs,
    ) -> ActivityEventDraft | None:
        if not isinstance(result, ExpertRoutineResponse):
            return None
        on = result.routine.enabled
        return ActivityEventDraft(
            category="SCHEDULE",
            event_type="routine.enabled" if on else "routine.disabled",
            title=(
                f"Switched on '{result.routine.title}'"
                if on
                else f"Switched off '{result.routine.title}'"
            ),
        )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        routine_id: str = "",
        enabled: bool = False,
        expert_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )
        if not routine_id:
            return ErrorResponse(
                message="`routine_id` is required.",
                error="missing_routine_id",
                session_id=session_id,
            )
        target = await resolve_target_expert(user_id, session, expert_id)
        if isinstance(target, ErrorResponse):
            return target
        try:
            routine = await self._apply(
                user_id, target, routine_id, enabled, session, kwargs
            )
        except RoutineUnansweredAsksError as e:
            return ErrorResponse(
                message=str(e),
                error="unanswered_asks",
                session_id=session_id,
            )
        except RoutineNotFoundError:
            return ErrorResponse(
                message=f"Routine '{routine_id}' was not found on this expert.",
                error="routine_not_found",
                session_id=session_id,
            )
        except ValueError as e:
            return ErrorResponse(
                message=f"That cadence is not valid: {e}",
                error="invalid_cron",
                session_id=session_id,
            )
        return ExpertRoutineResponse(
            message=_confirmation(routine),
            expert_id=target,
            routine=routine,
            session_id=session_id,
        )

    async def _apply(
        self,
        user_id: str,
        expert_id: str,
        routine_id: str,
        enabled: bool,
        session: ChatSession,
        kwargs: dict[str, Any],
    ) -> ExpertRoutine:
        if not enabled:
            return await experts_db().disable_routine(user_id, expert_id, routine_id)
        pin_here = bool(kwargs.get("pin_to_this_chat"))
        return await experts_db().enable_routine(
            user_id,
            expert_id,
            routine_id,
            prompt=kwargs.get("prompt"),
            crons=kwargs.get("crons"),
            session_id=session.session_id if pin_here else None,
            grants_credentials=bool(kwargs.get("grants_credentials")),
        )


def _confirmation(routine: ExpertRoutine) -> str:
    """What the model should be able to repeat back without checking again."""
    if not routine.enabled:
        return f"'{routine.title}' is switched off. Nothing is scheduled for it."
    reach = (
        "It can use the expert's connected services."
        if routine.grants_credentials
        else "It reaches nothing outside the platform."
    )
    where = {
        "THREAD": "in its own thread",
        "HERE": "in this chat",
        "FRESH": "in a new chat each time",
    }.get(routine.session_mode, "in its own thread")
    return (
        f"'{routine.title}' is on: {', '.join(routine.crons)} "
        f"in the user's timezone, {where}. {reach}"
    )
