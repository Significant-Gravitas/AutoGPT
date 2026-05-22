"""Tool for scheduling a follow-up copilot turn on a session.

When the model wants to defer work — "check the CI in 20 minutes",
"send the email at 7am" — it calls this tool with the message to send
at the scheduled time. At fire time the scheduler enqueues a new
copilot turn against ``session_id``, so the conversation resumes with
full history intact.

By default the followup targets the current session, but the caller
can pass ``session_id`` explicitly to target a different conversation
they own (e.g. a parent autopilot scheduling a wake-up for a
sub-session, or an admin tool resuming a specific user session).
Ownership is validated against the calling user — you cannot schedule
followups on someone else's sessions.

The tool ends the current turn; the model should send its final
user-facing message *before* calling this. The deferred work happens
in a fresh turn that the user will see arrive at the scheduled time.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.triggers.cron import CronTrigger

from backend.copilot.model import ChatSession, get_chat_session
from backend.copilot.tracking import track_followup_scheduled
from backend.data.db_accessors import user_db
from backend.util.clients import get_scheduler_client
from backend.util.timezone_utils import get_user_timezone_or_utc

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


class ScheduleCreatedResponse(ToolResponseBase):
    """Response confirming a follow-up was scheduled."""

    type: ResponseType = ResponseType.SCHEDULE_CREATED
    schedule_id: str
    next_run_time: str
    is_recurring: bool


class ScheduleFollowupTool(BaseTool):
    """Schedule a follow-up turn on a copilot session.

    Defaults to the current session ("check on this in 20 min"). Pass
    ``session_id`` to target a different conversation owned by the
    same user. Exactly one of ``delay_seconds`` or ``cron`` must be
    provided.
    """

    @property
    def name(self) -> str:
        return "schedule_followup"

    @property
    def description(self) -> str:
        return (
            "Schedule a copilot follow-up turn. The 'message' is sent at "
            "the scheduled time and the model resumes with the target "
            "conversation's full history. Defaults to the current "
            "session; pass 'session_id' to target a different "
            "conversation you own. Use 'delay_seconds' for one-shot "
            "followups ('in 20 minutes', 'at 7am tomorrow' — convert "
            "absolute times to a delay) or 'cron' for recurring "
            "schedules ('every Monday at 9am'). After calling this tool "
            "your turn ends — send your final user-facing message "
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
                "message": {
                    "type": "string",
                    "description": (
                        "Instruction to send at the scheduled time, e.g. "
                        "'Check CI status on PR #13187 and fix any "
                        "failing checks.'"
                    ),
                },
                "delay_seconds": {
                    "type": "integer",
                    "description": (
                        "Fire once after this many seconds from now. "
                        "Mutually exclusive with cron."
                    ),
                    "minimum": 60,
                },
                "cron": {
                    "type": "string",
                    "description": (
                        "5-field cron expression for a recurring "
                        "follow-up, evaluated in the user's timezone "
                        "(e.g. '0 9 * * 1' = Mondays at 9am). Mutually "
                        "exclusive with delay_seconds."
                    ),
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Target session for the follow-up. Defaults to "
                        "the current conversation (see <session_context>). "
                        "Must be a session owned by the calling user — "
                        "others are rejected as 'session_not_found'."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "Optional short label shown in the schedules UI.",
                },
            },
            "required": ["message"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        current_session_id = session.session_id if session else None
        if not user_id or not current_session_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=current_session_id,
            )

        message: str | None = kwargs.get("message")
        if not message or not message.strip():
            return ErrorResponse(
                message="`message` is required.",
                error="missing_message",
                session_id=current_session_id,
            )

        # Target session: explicit override if provided AND owned by the
        # caller, otherwise the current session. `get_chat_session` returns
        # None when the session doesn't exist OR belongs to a different user,
        # so this single check covers both "not found" and "not yours".
        override_session_id: str | None = kwargs.get("session_id")
        if override_session_id and override_session_id != current_session_id:
            target = await get_chat_session(override_session_id, user_id)
            if target is None:
                return ErrorResponse(
                    message=(
                        f"Session {override_session_id} not found or not "
                        f"owned by the calling user."
                    ),
                    error="session_not_found",
                    session_id=current_session_id,
                )
            target_session_id = override_session_id
        else:
            target_session_id = current_session_id

        delay_seconds: int | None = kwargs.get("delay_seconds")
        cron: str | None = kwargs.get("cron")
        name: str | None = kwargs.get("name")

        if (delay_seconds is None) == (cron is None):
            return ErrorResponse(
                message="Provide exactly one of `delay_seconds` or `cron`.",
                error="invalid_trigger",
                session_id=current_session_id,
            )

        run_at: datetime | None = None
        if delay_seconds is not None:
            if delay_seconds < 60:
                return ErrorResponse(
                    message="`delay_seconds` must be at least 60.",
                    error="delay_too_short",
                    session_id=current_session_id,
                )
            run_at = datetime.now(tz=timezone.utc) + timedelta(seconds=delay_seconds)

        user = await user_db().get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)

        # Pre-validate cron locally — CronTrigger.from_crontab raises ValueError
        # on the scheduler service, which crosses the RPC boundary as a generic
        # RemoteError and won't match `except ValueError` below. Validating
        # here gives the model a clean, actionable error message.
        if cron is not None:
            try:
                CronTrigger.from_crontab(cron, timezone=user_timezone)
            except ValueError as e:
                return ErrorResponse(
                    message=f"Invalid cron expression: {e}",
                    error="invalid_cron",
                    session_id=current_session_id,
                )

        try:
            info = await get_scheduler_client().add_copilot_turn_schedule(
                user_id=user_id,
                session_id=target_session_id,
                message=message,
                cron=cron,
                run_at=run_at,
                name=name,
                user_timezone=user_timezone,
            )
        except ValueError as e:
            return ErrorResponse(
                message=f"Invalid schedule: {e}",
                error="invalid_schedule",
                session_id=current_session_id,
            )

        is_recurring = cron is not None
        track_followup_scheduled(
            user_id=user_id,
            session_id=target_session_id,
            schedule_id=info.id,
            is_recurring=is_recurring,
        )
        target_note = (
            ""
            if target_session_id == current_session_id
            else f" on session {target_session_id}"
        )
        when_str = (
            f"every '{cron}' ({user_timezone})"
            if is_recurring
            else f"once at {info.next_run_time}"
        )
        return ScheduleCreatedResponse(
            message=f"Follow-up scheduled {when_str}{target_note}.",
            schedule_id=info.id,
            next_run_time=info.next_run_time,
            is_recurring=is_recurring,
            session_id=current_session_id,
        )
