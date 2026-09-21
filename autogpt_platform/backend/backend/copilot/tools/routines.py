"""Offer, switch on, and switch off standing work.

An expert arrives with its routines already listed and every one of them off.
Switching one on is a conversation, not a toggle: the template ships a proposal
and the questions it needs answered, and the owner's answers become the prompt
and the cadence that actually run. That is the whole point of the round trip —
what a roster wrote for everybody is never what any one account should run
unattended.

The account has standing work too. Otto is not a row in ``Expert``, so its
routines hang off the owner directly; everything else about them is the same,
including that nothing a routine's own turn does can create another one.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.api.features.experts.models import ExpertRoutine
from backend.api.features.experts.routines import (
    RoutineNotFoundError,
    RoutineUnansweredAsksError,
)
from backend.copilot.model import ChatSession, get_chat_session
from backend.data.activity_event import ActivityEventDraft
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .expert_scope import RoutineOwner, resolve_routine_owner
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)

# The shortest deferral ``schedule_followup`` accepts, kept identical so the
# two tools cannot disagree about what "soon" means.
_MIN_DELAY_SECONDS = 60

_EXPERT_ID_PARAM = {
    "type": "string",
    "description": (
        "Whose standing work. Experts act on themselves and omit it. From "
        "personal AutoPilot, name an expert, or omit for the account's own."
    ),
}


class RoutinesResponse(ToolResponseBase):
    type: ResponseType = ResponseType.ROUTINES
    expert_id: str | None = None
    routines: list[ExpertRoutine]


class RoutineResponse(ToolResponseBase):
    type: ResponseType = ResponseType.ROUTINE
    expert_id: str | None = None
    routine: ExpertRoutine


class ListRoutinesTool(BaseTool):
    @property
    def name(self) -> str:
        return "list_routines"

    @property
    def description(self) -> str:
        return (
            "List standing work: routines and pending one-offs, which are "
            "switched on, and what each still needs answered. Read before "
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
        owner = await resolve_routine_owner(user_id, session, expert_id)
        if isinstance(owner, ErrorResponse):
            return owner
        routines = await experts_db().list_routines(user_id, owner.expert_id)
        return RoutinesResponse(
            message=(
                f"{len(routines)} routine(s); "
                f"{sum(1 for r in routines if r.enabled)} switched on."
            ),
            expert_id=owner.expert_id,
            routines=routines,
            session_id=session_id,
        )


class ScheduleRoutineTool(BaseTool):
    @property
    def name(self) -> str:
        return "schedule_routine"

    @property
    def description(self) -> str:
        return (
            "Create standing work, or switch a routine on or off. Omit "
            "'routine_id' and pass 'title'/'prompt' plus 'crons' or "
            "'delay_seconds' to set up something new you agreed with the "
            "user. ON commits to what it will do: answer any 'asks' first "
            "(never guess), use the time THEY chose, and show them the "
            "wording before calling."
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
                    "description": (
                        "Routine id from list_routines. Omit to create a new "
                        "one from 'title' and 'prompt'."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Short name, for a new routine. Required when there "
                        "is no routine_id."
                    ),
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
                        "Repeating work: 5-field crons in the user's timezone, "
                        "one per fire time. Minute 'H' means any minute in "
                        "that hour, picked once and kept — use it when they "
                        "gave an hour rather than a time, so routines do not "
                        "pile up. Not with 'delay_seconds'."
                    ),
                },
                "delay_seconds": {
                    "type": "integer",
                    "minimum": _MIN_DELAY_SECONDS,
                    "description": (
                        "One-off work: seconds from now. Convert an absolute "
                        "time ('at six') to a delay. Not with 'crons'."
                    ),
                },
                "session_mode": {
                    "type": "string",
                    "enum": ["THREAD", "PINNED", "FRESH"],
                    "description": (
                        "Where each run lands. THREAD (default): its own "
                        "thread, reused, which is how it remembers what it "
                        "already reported. PINNED: an existing chat — this "
                        "one unless 'session_id' names another. FRESH: a new "
                        "chat each time, remembering nothing."
                    ),
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "With PINNED, the chat each run lands in. Defaults to "
                        "this one; its id is in <session_context>. Same user "
                        "and same expert only."
                    ),
                },
                "grants_credentials": {
                    "type": "boolean",
                    "description": (
                        "Let it use the owner's connected services. Omit to "
                        "leave as-is. False: it researches and drafts but "
                        "touches nothing outside the platform. True only when "
                        "the user agreed to THIS routine using their accounts."
                    ),
                },
                "expert_id": _EXPERT_ID_PARAM,
            },
            "required": ["enabled"],
        }

    def activity_event(
        self,
        session: ChatSession,
        result: ToolResponseBase,
        **kwargs,
    ) -> ActivityEventDraft | None:
        if not isinstance(result, RoutineResponse):
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
        if not routine_id and not kwargs.get("title"):
            return ErrorResponse(
                message=(
                    "Give a `routine_id` to change an existing routine, or a "
                    "`title`, a `prompt` and either `crons` or "
                    "`delay_seconds` to create one."
                ),
                error="missing_routine",
                session_id=session_id,
            )
        owner = await resolve_routine_owner(user_id, session, expert_id)
        if isinstance(owner, ErrorResponse):
            return owner
        pinned = await self._pinned_session(user_id, session, owner, kwargs)
        if isinstance(pinned, ErrorResponse):
            return pinned
        try:
            routine = await self._apply(
                user_id, owner, routine_id, enabled, pinned, kwargs
            )
        except RoutineUnansweredAsksError as e:
            return ErrorResponse(
                message=str(e),
                error="unanswered_asks",
                session_id=session_id,
            )
        except RoutineNotFoundError:
            return ErrorResponse(
                message=f"Routine '{routine_id}' was not found.",
                error="routine_not_found",
                session_id=session_id,
            )
        except ValueError as e:
            return ErrorResponse(
                message=f"That schedule is not valid: {e}",
                error="invalid_schedule",
                session_id=session_id,
            )
        return RoutineResponse(
            message=_confirmation(routine),
            expert_id=owner.expert_id,
            routine=routine,
            session_id=session_id,
        )

    async def _pinned_session(
        self,
        user_id: str,
        session: ChatSession,
        owner: RoutineOwner,
        kwargs: dict[str, Any],
    ) -> str | None | ErrorResponse:
        """The chat a PINNED routine fires into, validated before it is stored.

        Same rule ``schedule_followup`` applies to its ``session_id``: the
        target must be the caller's and in the routine owner's scope, so a
        routine cannot be aimed at another persona's memory. The fire path
        re-checks both on every run, because a chat can be deleted or a scope
        can change long after this.

        Scoped to ``owner``, not to the calling session: personal AutoPilot
        managing an expert's routine is in no expert scope itself, so
        defaulting to "this chat" would pin the owner's Otto chat onto an
        expert's routine — which the fire path then refuses on every run.

        Returns ``None`` when the call did not ask to pin anything. THREAD and
        FRESH have no chat to name, and an omitted ``session_mode`` on an
        existing routine means "leave it as it is" — answering with the
        caller's chat there would silently re-pin a routine that was already
        pinned somewhere else.
        """
        requested: str | None = kwargs.get("session_id")
        current = session.session_id if session else None
        in_owner_scope = session.expert_id == owner.expert_id

        if requested:
            if requested == current and in_owner_scope:
                return requested
            target = await get_chat_session(requested, user_id)
            if target is None or target.expert_id != owner.expert_id:
                return ErrorResponse(
                    message=(
                        f"Session {requested} not found, not owned by the "
                        "calling user, or outside the routine owner's memory "
                        "scope."
                    ),
                    error="session_not_found",
                    session_id=current,
                )
            return requested

        # Nothing named. Only a call that explicitly asks to pin needs a chat;
        # everything else has none to give and must not be refused for it.
        if str(kwargs.get("session_mode") or "").upper() != "PINNED":
            return None
        if in_owner_scope:
            return current
        return ErrorResponse(
            message=(
                "PINNED needs `session_id`: this chat belongs to a different "
                "scope than the routine's owner, so a routine pinned to it "
                "would never run. Name one of that owner's chats, or use "
                "THREAD."
            ),
            error="session_required",
            session_id=current,
        )

    async def _apply(
        self,
        user_id: str,
        owner: RoutineOwner,
        routine_id: str,
        enabled: bool,
        pinned_session_id: str | None,
        kwargs: dict[str, Any],
    ) -> ExpertRoutine:
        run_at = _run_at(kwargs.get("delay_seconds"))
        crons = kwargs.get("crons")
        grant = kwargs.get("grants_credentials")
        if not routine_id:
            created = await experts_db().create_routine(
                user_id,
                owner.expert_id,
                title=kwargs.get("title") or "",
                prompt=kwargs.get("prompt") or "",
                crons=crons,
                run_at=run_at,
                session_mode=kwargs.get("session_mode"),
                session_id=pinned_session_id,
                # A routine dictated in conversation is the owner's own words,
                # so it reaches what they reach unless they said otherwise.
                # ``None`` is "they did not say", which on a new row is the
                # default rather than a change to leave alone.
                grants_credentials=True if grant is None else bool(grant),
            )
            if not enabled:
                return created
            routine_id = created.id
            # Already resolved onto the new row; passing them again would
            # re-validate the same values and, for a one-shot, recompute a
            # delay against a later now.
            crons, run_at = None, None
        if not enabled:
            return await experts_db().disable_routine(
                user_id, owner.expert_id, routine_id
            )
        return await experts_db().enable_routine(
            user_id,
            owner.expert_id,
            routine_id,
            prompt=kwargs.get("prompt"),
            crons=crons,
            run_at=run_at,
            session_mode=kwargs.get("session_mode"),
            # Only read when the mode is PINNED; passed always so the tool
            # never has to know which modes want it.
            pinned_session_id=pinned_session_id,
            # Absent means "leave the grant alone", so rewording a routine
            # never quietly widens what it can touch.
            grants_credentials=grant,
        )


def _run_at(delay_seconds: int | None) -> datetime | None:
    if delay_seconds is None:
        return None
    if delay_seconds < _MIN_DELAY_SECONDS:
        raise ValueError(f"`delay_seconds` must be at least {_MIN_DELAY_SECONDS}.")
    return datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)


def _confirmation(routine: ExpertRoutine) -> str:
    """What the model should be able to repeat back without checking again."""
    if not routine.enabled:
        return f"'{routine.title}' is switched off. Nothing is scheduled for it."
    reach = (
        "It can use the owner's connected services."
        if routine.grants_credentials
        else "It reaches nothing outside the platform."
    )
    where = {
        "THREAD": "in its own thread",
        "PINNED": "in the chat it was pinned to",
        "FRESH": "in a new chat each time",
    }.get(routine.session_mode, "in its own thread")
    if routine.crons:
        when = f"{', '.join(routine.crons)} in the user's timezone"
    elif routine.run_at is not None:
        when = f"once at {routine.run_at:%Y-%m-%d %H:%M} UTC"
    else:
        when = "at no time it can name"
    return f"'{routine.title}' is on: {when}, {where}. {reach}"
