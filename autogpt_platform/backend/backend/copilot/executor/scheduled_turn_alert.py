"""Alert when a scheduled copilot turn fails after it was dispatched.

The scheduler only learns whether it got a turn onto the queue. What happens
next, such as the turn erroring or a tool it called returning an error
(``run_agent`` in SECRT-2799, every Monday for ten weeks), is only visible
here in the copilot executor, and nobody is watching an unattended turn. So a
failed scheduled turn raises one Sentry event and a metric, and every
scheduled turn's outcome counts toward its schedule's failure streak, which
pauses a chat that keeps failing the same way and tells its owner
(SECRT-2879).
"""

import asyncio
import logging
import re

import orjson
import sentry_sdk
from pydantic import BaseModel

from backend.copilot import stream_registry
from backend.copilot.constants import COMPACTION_TOOL_NAME
from backend.copilot.permissions import SDK_BUILTIN_TOOL_NAMES
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamToolOutputAvailable,
    StreamTurnPaused,
)
from backend.copilot.scheduled_chats.notices import schedule_title
from backend.copilot.scheduled_chats.outcomes import (
    PAUSE_ERROR_CODES,
    RunOutcome,
    ToolFailureDetail,
    classify_run,
    text_hash,
    tool_failure_detail,
)
from backend.copilot.scheduled_chats.service import ScheduledRun
from backend.copilot.tools.models import ResponseType
from backend.monitoring.instrumentation import COPILOT_SCHEDULED_TURN_FAILURES
from backend.util.clients import get_database_manager_async_client

from .utils import CoPilotExecutionEntry, ScheduledTurnOrigin

logger = logging.getLogger(__name__)

# How long a finished turn may spend recording its outcome. The turn is
# already marked done; this only holds the worker.
_RECORD_TIMEOUT_SECONDS = 20

# A tool's error text can quote the user's data, and neither the log sinks nor
# Sentry are redacted, so a failure is reported by its error code alone.
_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
UNCLASSIFIED = "unclassified"

# Finding a block or an agent and reading its schema is how a run gets ready
# to work, not work done: a run whose one real step failed after them failed
# outright, like a weekly report that finds its agent and then fails to run it
# (SECRT-2799).
_LOOKUP_TOOLS = frozenset(
    {"find_capability", "describe_capability", "find_agent", "find_library_agent"}
)


class ToolFailure(BaseModel):
    tool: str
    error_type: str


class ScheduledTurnWatch:
    """Collects a scheduled turn's error code and tool errors while it streams,
    then reports once the turn has ended."""

    def __init__(self, entry: CoPilotExecutionEntry, origin: ScheduledTurnOrigin):
        self._entry = entry
        self._origin = origin
        self.tool_failures: list[ToolFailure] = []
        self._turn_error_code: str | None = None
        # For the failure streak: what each failure said, and how many
        # platform tool calls worked.
        self._failure_details: list[tuple[str, ToolFailureDetail]] = []
        self._tool_successes = 0
        # Set when the turn landed at a limit, or was refused one at pickup.
        self._pause: dict | None = None

    @classmethod
    def for_entry(cls, entry: CoPilotExecutionEntry) -> "ScheduledTurnWatch | None":
        if entry.scheduled is None:
            return None
        return cls(entry, entry.scheduled)

    def observe(self, chunk: StreamBaseResponse) -> None:
        if isinstance(chunk, StreamError):
            self._turn_error_code = chunk.code
            return
        if isinstance(chunk, StreamTurnPaused):
            self._pause = chunk.pause
            return
        if not isinstance(chunk, StreamToolOutputAvailable):
            return
        # The CLI's own tools stay out, whatever they return: a missed Read or
        # Grep is the model probing, and a file it read can hold anything,
        # an error payload included.
        if not _ours(chunk.toolName):
            return
        error_type = tool_error_type(chunk.output)
        # A platform tool can also fail without an ErrorResponse payload (a
        # login prompt, a held read, a plain-text error).
        if error_type is None and not chunk.success:
            error_type = UNCLASSIFIED
        tool = chunk.toolName or "unknown"
        if error_type is not None:
            self.tool_failures.append(ToolFailure(tool=tool, error_type=error_type))
            detail = tool_failure_detail(chunk.output) or ToolFailureDetail(
                error_type=UNCLASSIFIED, text_hash=text_hash(chunk.output)
            )
            self._failure_details.append((tool, detail))
        elif chunk.toolName not in _LOOKUP_TOOLS:
            self._tool_successes += 1

    def outcome(self, turn_error: str | None) -> RunOutcome:
        landed = self._pause is not None
        return classify_run(
            cancelled=turn_error == stream_registry.CANCELLED_MESSAGE,
            turn_error_code=self._landed_error_code() or self._turn_error_code,
            # A pause decides the run as a turn error does: its tools were
            # refused while it wrote a checkpoint, so they are no failures.
            turn_errored=turn_error is not None or landed,
            tool_failures=[] if landed else self._failure_details,
            tool_successes=self._tool_successes,
        )

    def _landed_error_code(self) -> str | None:
        """The error a pause at the run's own budget or step limit counts as;
        None for the account's windows, and for a run refused at pickup, which
        never ran."""
        if self._pause is None or self._pause.get("started") is False:
            return None
        reason = self._pause.get("reason")
        return PAUSE_ERROR_CODES.get(reason) if isinstance(reason, str) else None

    async def record(self, turn_error: str | None) -> None:
        """Count this run toward its schedule's failure streak.

        Never raises, and bounded: it runs in the turn's cleanup, after the
        turn has been marked done.
        """
        if self._origin.purpose != "followup":
            # A chat's own turn, resumed or sent at its budget reset: the
            # chat is the user's, not a schedule that can be paused.
            return
        outcome = self.outcome(turn_error)
        schedule_key = self._origin.routine_id or self._origin.schedule_id
        if outcome.neutral or schedule_key is None or self._entry.user_id is None:
            return
        run = ScheduledRun(
            user_id=self._entry.user_id,
            schedule_key=schedule_key,
            schedule_id=self._origin.schedule_id,
            routine_id=self._origin.routine_id,
            session_id=self._entry.session_id,
            title=schedule_title(routine_title=None, message=self._entry.message),
            outcome=outcome,
        )
        try:
            await asyncio.wait_for(
                get_database_manager_async_client().record_scheduled_run(run),
                timeout=_RECORD_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.warning(
                "Could not record scheduled copilot turn %s toward its schedule's "
                "failure streak",
                self._entry.turn_id,
                exc_info=True,
            )

    def report(self, turn_error: str | None) -> None:
        """Never raises: it runs in the turn's cleanup."""
        if turn_error == stream_registry.CANCELLED_MESSAGE:
            turn_error = None
        failures = self.tool_failures
        if self._pause is not None:
            # The tools a landing refused are no failures; the run's own
            # limits report as the errors they replaced.
            failures = []
            turn_error = turn_error or self._landed_error_code()
        if turn_error is None and not failures:
            return
        try:
            self._count(turn_error, failures)
            self._alert(turn_error, failures)
        except Exception:
            logger.warning(
                "Could not report failed scheduled copilot turn %s",
                self._entry.turn_id,
                exc_info=True,
            )

    def _count(self, turn_error: str | None, failures: list[ToolFailure]) -> None:
        if turn_error is not None:
            COPILOT_SCHEDULED_TURN_FAILURES.labels(reason="turn_error", tool="").inc()
        for failure in failures:
            COPILOT_SCHEDULED_TURN_FAILURES.labels(
                reason="tool_error", tool=failure.tool
            ).inc()

    def _alert(self, turn_error: str | None, failures: list[ToolFailure]) -> None:
        if turn_error is not None:
            reason, summary = "turn_error", "the turn errored"
        else:
            reason = "tool_error"
            summary = f"{failures[0].tool} returned an error"
        details = {
            "schedule_id": self._origin.schedule_id,
            "routine_id": self._origin.routine_id,
            "cron": self._origin.cron,
            "session_id": self._entry.session_id,
            "turn_id": self._entry.turn_id,
            "turn_error_type": (
                error_type(self._turn_error_code or turn_error) if turn_error else None
            ),
            "tool_failures": [f.model_dump() for f in failures],
        }
        failed_tools = ", ".join(f"{f.tool} ({f.error_type})" for f in failures)
        logger.warning(
            f"Scheduled copilot turn failed: {summary} "
            f"(schedule {self._origin.schedule_id}, session "
            f"{self._entry.session_id[:12]}, turn error: "
            f"{details['turn_error_type'] or 'none'}, "
            f"failed tools: {failed_tools or 'none'})"
        )
        with sentry_sdk.new_scope() as scope:
            scope.set_tag("copilot_scheduled_turn_failure", reason)
            scope.set_user({"id": self._entry.user_id})
            scope.set_context("copilot_scheduled_turn", details)
            sentry_sdk.capture_message(
                f"Scheduled copilot turn failed: {summary}", level="error"
            )


def tool_error_type(output: str | dict) -> str | None:
    """The type of error a copilot tool reported, or None when it did not
    report one. Tools report failures as an ``ErrorResponse`` payload rather
    than by raising, so a turn can carry on and still look successful."""
    payload = output
    if isinstance(output, str):
        try:
            payload = orjson.loads(output)
        except orjson.JSONDecodeError:
            return None
    if not isinstance(payload, dict) or payload.get("type") != ResponseType.ERROR:
        return None
    return error_type(payload.get("error"))


def error_type(error: object) -> str:
    """*error* when it is an error code such as ``library_agent_not_found``.
    Free text, which can carry anything, is reported only as unclassified."""
    if isinstance(error, str) and _ERROR_CODE.fullmatch(error):
        return error
    return UNCLASSIFIED


def _ours(tool_name: str | None) -> bool:
    """A platform tool call. The CLI's own tools stay out (a missed Read or
    Grep is the model probing), and so does the compaction notice, which no
    tool returned: counting it as work done would keep a run that failed
    everywhere else from ever reading as failed."""
    return (
        tool_name is not None
        and tool_name not in SDK_BUILTIN_TOOL_NAMES
        and tool_name != COMPACTION_TOOL_NAME
    )
