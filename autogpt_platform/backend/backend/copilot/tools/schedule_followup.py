"""Tool for scheduling a follow-up copilot turn on the same session.

When the model wants to defer work — "check the CI in 20 minutes",
"send the email at 7am" — it calls this tool with the message to send
to itself at the scheduled time. At fire time the scheduler enqueues a
new copilot turn against the same ``session_id``, so the conversation
resumes with full history intact.

The tool ends the current turn; the model should send its final user-
facing message *before* calling this. The deferred work happens in a
fresh turn that the user will see arrive at the scheduled time.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.copilot.model import ChatSession
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
    """Schedule a follow-up turn on this copilot session.

    Use this when the user asks you to do something later — check on
    progress, send a message at a specific time, run a recurring task.
    Exactly one of ``delay_seconds`` or ``cron`` must be provided.
    """

    @property
    def name(self) -> str:
        return "schedule_followup"

    @property
    def description(self) -> str:
        return (
            "Schedule a follow-up turn on this conversation. The 'message' "
            "you provide will be sent to yourself at the scheduled time, "
            "with full conversation history available. Use 'delay_seconds' "
            "for one-shot followups ('in 20 minutes') or 'cron' for "
            "recurring schedules ('every Monday at 9am'). After calling "
            "this tool your turn ends — send your final user-facing "
            "message before calling."
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
                        "The instruction to send to yourself at the "
                        "scheduled time, e.g. 'Check the CI status on "
                        "PR #13187 and fix any failing checks.'"
                    ),
                },
                "delay_seconds": {
                    "type": "integer",
                    "description": (
                        "Fire once after this many seconds from now. Use "
                        "for 'in 20 minutes' or 'at 7am tomorrow' style "
                        "requests (convert the absolute time to a delay). "
                        "Mutually exclusive with cron."
                    ),
                    "minimum": 60,
                },
                "cron": {
                    "type": "string",
                    "description": (
                        "Cron expression (5-field) for a recurring "
                        "follow-up, evaluated in the user's timezone, "
                        "e.g. '0 9 * * 1' for Mondays at 9am. Mutually "
                        "exclusive with delay_seconds."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": ("Optional short label shown in the schedules UI."),
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
        session_id = session.session_id if session else None
        if not user_id or not session_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        message: str | None = kwargs.get("message")
        if not message or not message.strip():
            return ErrorResponse(
                message="`message` is required.",
                error="missing_message",
                session_id=session_id,
            )

        delay_seconds: int | None = kwargs.get("delay_seconds")
        cron: str | None = kwargs.get("cron")
        name: str | None = kwargs.get("name")

        if (delay_seconds is None) == (cron is None):
            return ErrorResponse(
                message=("Provide exactly one of `delay_seconds` or `cron`."),
                error="invalid_trigger",
                session_id=session_id,
            )

        run_at: datetime | None = None
        if delay_seconds is not None:
            if delay_seconds < 60:
                return ErrorResponse(
                    message="`delay_seconds` must be at least 60.",
                    error="delay_too_short",
                    session_id=session_id,
                )
            run_at = datetime.now(tz=timezone.utc) + timedelta(seconds=delay_seconds)

        user = await user_db().get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)

        try:
            info = await get_scheduler_client().add_copilot_turn_schedule(
                user_id=user_id,
                session_id=session_id,
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
                session_id=session_id,
            )

        is_recurring = cron is not None
        when_str = (
            f"every '{cron}' ({user_timezone})"
            if is_recurring
            else f"once at {info.next_run_time}"
        )
        return ScheduleCreatedResponse(
            message=f"Follow-up scheduled {when_str}.",
            schedule_id=info.id,
            next_run_time=info.next_run_time,
            is_recurring=is_recurring,
            session_id=session_id,
        )
