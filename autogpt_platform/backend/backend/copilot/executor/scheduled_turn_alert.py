"""Alert when a scheduled copilot turn fails after it was dispatched.

The scheduler only learns whether it got a turn onto the queue. What happens
next, such as the turn erroring or a tool it called returning an error
(``run_agent`` in SECRT-2799, every Monday for ten weeks), is only visible
here in the copilot executor, and nobody is watching an unattended turn. So a
failed scheduled turn raises one Sentry event and a metric.
"""

import logging

import orjson
import sentry_sdk
from pydantic import BaseModel

from backend.copilot import stream_registry
from backend.copilot.response_model import StreamBaseResponse, StreamToolOutputAvailable
from backend.copilot.tools.models import ResponseType
from backend.monitoring.instrumentation import COPILOT_SCHEDULED_TURN_FAILURES

from .utils import CoPilotExecutionEntry, ScheduledTurnOrigin

logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 500


class ToolFailure(BaseModel):
    tool: str
    error: str


class ScheduledTurnWatch:
    """Collects a scheduled turn's tool errors while it streams, then reports
    once the turn has ended."""

    def __init__(self, entry: CoPilotExecutionEntry, origin: ScheduledTurnOrigin):
        self._entry = entry
        self._origin = origin
        self.tool_failures: list[ToolFailure] = []

    @classmethod
    def for_entry(cls, entry: CoPilotExecutionEntry) -> "ScheduledTurnWatch | None":
        if entry.scheduled is None:
            return None
        return cls(entry, entry.scheduled)

    def observe(self, chunk: StreamBaseResponse) -> None:
        if not isinstance(chunk, StreamToolOutputAvailable):
            return
        error = tool_error(chunk.output)
        if error is not None:
            self.tool_failures.append(
                ToolFailure(tool=chunk.toolName or "unknown", error=error)
            )

    def report(self, turn_error: str | None) -> None:
        """Never raises: it runs in the turn's cleanup."""
        if turn_error == stream_registry.CANCELLED_MESSAGE:
            turn_error = None
        if turn_error is None and not self.tool_failures:
            return
        try:
            self._count(turn_error)
            self._alert(turn_error)
        except Exception:
            logger.warning(
                "Could not report failed scheduled copilot turn %s",
                self._entry.turn_id,
                exc_info=True,
            )

    def _count(self, turn_error: str | None) -> None:
        if turn_error is not None:
            COPILOT_SCHEDULED_TURN_FAILURES.labels(reason="turn_error", tool="").inc()
        for failure in self.tool_failures:
            COPILOT_SCHEDULED_TURN_FAILURES.labels(
                reason="tool_error", tool=failure.tool
            ).inc()

    def _alert(self, turn_error: str | None) -> None:
        if turn_error is not None:
            reason, summary = "turn_error", "the turn errored"
        else:
            reason = "tool_error"
            summary = f"{self.tool_failures[0].tool} returned an error"
        details = {
            "schedule_id": self._origin.schedule_id,
            "routine_id": self._origin.routine_id,
            "cron": self._origin.cron,
            "session_id": self._entry.session_id,
            "turn_id": self._entry.turn_id,
            "turn_error": turn_error,
            "tool_failures": [f.model_dump() for f in self.tool_failures],
        }
        logger.warning(
            f"Scheduled copilot turn failed: {summary} "
            f"(schedule {self._origin.schedule_id}, session "
            f"{self._entry.session_id[:12]}): {turn_error or self.tool_failures}"
        )
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("copilot_scheduled_turn_failure", reason)
            scope.set_user({"id": self._entry.user_id})
            scope.set_context("copilot_scheduled_turn", details)
            sentry_sdk.capture_message(
                f"Scheduled copilot turn failed: {summary}", level="error"
            )


def tool_error(output: str | dict) -> str | None:
    """The error a copilot tool reported, or None when it did not report one.
    Tools report failures as an ``ErrorResponse`` payload rather than by
    raising, so a turn can carry on and still look successful."""
    payload = output
    if isinstance(output, str):
        try:
            payload = orjson.loads(output)
        except orjson.JSONDecodeError:
            return None
    if not isinstance(payload, dict) or payload.get("type") != ResponseType.ERROR:
        return None
    error = payload.get("error") or payload.get("message") or "unknown error"
    return str(error)[:_MAX_ERROR_CHARS]
