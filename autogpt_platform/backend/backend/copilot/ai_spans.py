"""AutoPilot's replies, tool calls and expert subtasks as PostHog LLM traces.

OpenRouter Broadcast sends each model call of a reply as an ``$ai_generation``
whose ``$ai_trace_id`` is the reply's trace (see ``llm_attribution``). The
backend sends the rest of that trace itself (SECRT-2979):

- ``$ai_trace``: one per reply, when its turn ends, with the turn's latency
  and whether it failed.
- ``$ai_span`` per tool call: the tool's name, latency and whether it failed,
  under the turn that ran it (or under the tool that called it).
- ``$ai_span`` per expert subtask: a turn another turn spawned joins the
  trace of the reply that spawned it, under the tool call that started it,
  and its own tool calls nest under it.

No event carries a tool's input or output, or an error's message: those can
hold Gmail or Drive content, which Google's Limited Use terms keep out of
analytics. ``$ai_input_state`` and ``$ai_output_state`` are left out for
that reason, and an error is named by its code or exception type only.

Like the generations and ``$ai_metric``, these are sent server-side for every
user, whatever their cookie consent, on the user's distinct id (the platform
user id) and with ``impersonated`` set when an admin acting as the user ran
the turn. ``posthog_client.capture`` queues each event for the client's
background thread and swallows any failure, so a turn never waits on
PostHog or fails because of it.
"""

import asyncio
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel

from backend.copilot.feedback_db import RATEABLE_ROLES
from backend.copilot.llm_attribution import is_turn_impersonated, turn_trace_id
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

if TYPE_CHECKING:
    from backend.copilot.executor.utils import CoPilotExecutionEntry
    from backend.copilot.model import ChatMessage

logger = logging.getLogger(__name__)

SOURCE = "chat_copilot"
TRACE_NAME = "copilot-turn"
SUBTASK_SPAN_NAME = "copilot-subtask"

# The span a tool call or subtask started now nests under: the running
# tool's, or else the running turn's. The executor sets it for every turn.
_current_span_id: ContextVar[str | None] = ContextVar("_current_span_id", default=None)
# How the running tool call is ending, for the tool's own code to mark.
_current_outcome: ContextVar["SpanOutcome | None"] = ContextVar(
    "_current_outcome", default=None
)


def trace_id_for(entry: "CoPilotExecutionEntry") -> str:
    """The trace a turn's events join.

    A spawned turn (an expert subtask, a sub-session) joins the trace of the
    turn that spawned it, which its envelope records when it is derived
    (``TurnEnvelope.ai_trace_id``). That is not always its tree's root: an
    approval wake is a reply of its own under the tree it was held in. A
    queued child and a woken subtask start under their stored envelope, so
    they keep that trace. An envelope stored before it kept the trace falls
    back to its tree's root. Every other turn is a reply of its own.
    """
    envelope = entry.envelope
    if envelope is not None and envelope.ai_trace_id:
        return envelope.ai_trace_id
    if envelope is not None and envelope.depth > 0 and envelope.tree_id:
        return envelope.tree_id
    return entry.turn_id


def set_turn_span(span_id: str | None) -> None:
    """Make *span_id* (the running turn's id) the parent of its tool calls."""
    _current_span_id.set(span_id or None)


def current_span_id() -> str | None:
    """The span a tool call or subtask started now nests under."""
    return _current_span_id.get()


def mark_tool_failed() -> None:
    """Mark the running tool call failed: it answered with an error rather
    than raising one. Outside a tool span it does nothing."""
    outcome = _current_outcome.get()
    if outcome is not None:
        outcome.is_error = True


def mark_tool_raised(error: BaseException) -> None:
    """Mark the running tool call failed by *error*, which the tool caught
    and answered with instead of letting it reach ``tool_span``. Only the
    exception's type is kept, never its message. Outside a tool span it does
    nothing."""
    outcome = _current_outcome.get()
    if outcome is not None:
        outcome.is_error = True
        outcome.error = type(error).__name__


class SpanOutcome(BaseModel):
    """How a span ended; the code it wraps marks a failure."""

    is_error: bool = False
    error: str | None = None


@contextmanager
def tool_span(
    tool_name: str,
    *,
    user_id: str | None,
    session_id: str,
    tool_call_id: str | None = None,
) -> Iterator[SpanOutcome]:
    """Time the block as one tool call of the running turn, sent as an
    ``$ai_span`` when it ends. Outside a turn there is no trace, so nothing
    is sent. Anything the block runs nests under this span."""
    outcome = SpanOutcome()
    trace_id = turn_trace_id()
    if not trace_id:
        yield outcome
        return
    span_id = str(uuid4())
    enclosing = _current_span_id.get()
    enclosing_outcome = _current_outcome.get()
    parent_id = enclosing or trace_id
    _current_span_id.set(span_id)
    _current_outcome.set(outcome)
    started_at = time.monotonic()
    cancelled = False
    try:
        yield outcome
    except asyncio.CancelledError:
        # The turn was stopped; the tool itself did not fail.
        cancelled = True
        raise
    except BaseException as error:
        outcome.is_error = True
        outcome.error = type(error).__name__
        raise
    finally:
        _current_span_id.set(enclosing)
        _current_outcome.set(enclosing_outcome)
        _send_span(
            user_id=user_id,
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            name=tool_name,
            session_id=session_id,
            latency_seconds=time.monotonic() - started_at,
            outcome=outcome,
            extra={"tool_call_id": tool_call_id, "cancelled": cancelled},
        )


class _Started(BaseModel):
    at: float
    trace_id: str
    parent_id: str


# Built-in tools the CLI calls often and cheaply, sub-agents included: tens
# a turn, each a lookup or a checklist update whose span says little. They
# get no span, which keeps the span volume to the calls that act or spend.
UNTRACED_BUILTIN_TOOLS: frozenset[str] = frozenset(
    {"TodoWrite", "Read", "Glob", "Grep", "LS", "NotebookRead"}
)


class ToolSpanTimer:
    """Times tool calls whose start and end arrive in separate callbacks: the
    Claude Agent SDK's Pre/PostToolUse hooks for the CLI's built-in tools
    (Bash, Task, WebSearch, ...), which never pass through ``BaseTool``.

    One timer serves one turn's hooks and goes with them, so a call whose
    end never arrives is dropped at the end of its turn.

    *enabled* off (``ChatConfig.posthog_builtin_tool_spans``) sends no span
    at all; on, every built-in but ``UNTRACED_BUILTIN_TOOLS`` gets one."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._started: dict[str, _Started] = {}

    def start(self, tool_use_id: str, tool_name: str) -> None:
        if not self._enabled or tool_name in UNTRACED_BUILTIN_TOOLS:
            return
        trace_id = turn_trace_id()
        if not trace_id:
            return
        self._started[tool_use_id] = _Started(
            at=time.monotonic(),
            trace_id=trace_id,
            parent_id=_current_span_id.get() or trace_id,
        )

    def finish(
        self,
        tool_use_id: str,
        tool_name: str,
        *,
        user_id: str | None,
        session_id: str | None,
        is_error: bool,
    ) -> None:
        """Send the call's span. A call that never started (refused, or
        outside a turn) sends nothing."""
        started = self._started.pop(tool_use_id, None)
        if started is None:
            return
        _send_span(
            user_id=user_id,
            trace_id=started.trace_id,
            span_id=str(uuid4()),
            parent_id=started.parent_id,
            name=tool_name,
            session_id=session_id,
            latency_seconds=time.monotonic() - started.at,
            outcome=SpanOutcome(is_error=is_error),
            extra={"tool_call_id": tool_use_id, "builtin": True},
        )


def capture_turn(
    entry: "CoPilotExecutionEntry",
    *,
    latency_seconds: float,
    outcome: str,
    error_code: str | None,
    engine: str,
    origin: str,
    tool_call_count: int,
) -> None:
    """Send the end of a turn: the ``$ai_trace`` of a reply, or the
    ``$ai_span`` of a subtask in the trace of the reply it works for.

    *outcome* is ``completed``, ``cancelled`` or ``failed``; only ``failed``
    is an error, and *error_code* (a code, never a message) says which.

    A redelivered turn ends twice; the turn id is the event's dedup key, so
    PostHog keeps one root (or one subtask span) per turn."""
    try:
        trace_id = trace_id_for(entry)
        failed = outcome == "failed"
        properties: dict[str, Any] = {
            "$ai_trace_id": trace_id,
            "$ai_session_id": entry.session_id,
            "$ai_latency": latency_seconds,
            "$ai_is_error": failed,
            "chat_session_id": entry.session_id,
            "turn_id": entry.turn_id,
            "outcome": outcome,
            "engine": engine,
            "turn_origin": origin,
            "tool_call_count": tool_call_count,
            "impersonated": entry.impersonated,
        }
        if failed and error_code:
            properties["$ai_error"] = error_code
        if trace_id == entry.turn_id:
            properties["$ai_span_name"] = TRACE_NAME
            event = PostHogEvent.AI_TRACE
        else:
            properties["$ai_span_name"] = SUBTASK_SPAN_NAME
            properties["$ai_span_id"] = entry.turn_id
            envelope = entry.envelope
            properties["$ai_parent_id"] = (
                envelope.ai_parent_span_id if envelope else None
            ) or trace_id
            properties["depth"] = envelope.depth if envelope else None
            event = PostHogEvent.AI_SPAN
        posthog_client.capture(
            entry.user_id,
            event,
            properties,
            source=SOURCE,
            dedup_key=entry.turn_id,
        )
    except Exception:
        logger.warning("Failed to send the turn's PostHog trace", exc_info=True)


def stamp_turn_trace_id(messages: Sequence["ChatMessage"], start_index: int) -> None:
    """Record the running turn's trace on the reply rows it wrote, so a
    rating of the reply names the trace (``feedback._send_to_posthog``).

    Bounded to the turn and never overwriting, like the other end-of-turn
    stamps; a row already flushed mid-turn is flagged for back-fill."""
    trace_id = turn_trace_id()
    if not trace_id:
        return
    for message in messages[start_index:]:
        if message.role in RATEABLE_ROLES and message.trace_id is None:
            message.trace_id = trace_id
            if message.sequence is not None:
                message.stamps_pending_save = True


def _send_span(
    *,
    user_id: str | None,
    trace_id: str,
    span_id: str,
    parent_id: str,
    name: str,
    session_id: str | None,
    latency_seconds: float,
    outcome: SpanOutcome,
    extra: dict[str, Any],
) -> None:
    try:
        properties: dict[str, Any] = {
            "$ai_trace_id": trace_id,
            "$ai_span_id": span_id,
            "$ai_parent_id": parent_id,
            "$ai_span_name": name,
            "$ai_latency": latency_seconds,
            "$ai_is_error": outcome.is_error,
            "impersonated": is_turn_impersonated(),
            **{key: value for key, value in extra.items() if value is not None},
        }
        if session_id:
            properties["$ai_session_id"] = session_id
            properties["chat_session_id"] = session_id
        if outcome.is_error and outcome.error:
            properties["$ai_error"] = outcome.error
        posthog_client.capture(user_id, PostHogEvent.AI_SPAN, properties, source=SOURCE)
    except Exception:
        logger.warning(f"Failed to send the PostHog span of {name}", exc_info=True)
