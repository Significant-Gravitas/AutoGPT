"""Sentry transactions for copilot turns and their tool calls.

A turn runs in the copilot executor as one ``copilot.turn`` transaction that
continues the trace of whatever dispatched it: the chat request, a parent
turn's tool, a graph node. Each tool call the turn makes is a ``copilot.tool``
transaction of its own in that trace, so a long autopilot turn stays under the
SDK's span limit however many tools it runs.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sentry_sdk.tracing import Span

from backend.copilot.constants import is_platform_text
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolInputStart,
)
from backend.copilot.service import strip_user_context_tags
from backend.util import tracing

if TYPE_CHECKING:
    from backend.copilot.executor.utils import CoPilotExecutionEntry
    from backend.copilot.model import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

TURN_OP = "copilot.turn"
TOOL_OP = "copilot.tool"

WAITED_AFTER_S = 60.0
"""A turn that starts this long after its chat request waited for it (queued
behind another turn, held at a limit, redelivered), so timing it from the
request would measure the wait rather than the reply."""

MODEL_OUTPUT_CHUNKS = (StreamTextDelta, StreamReasoningDelta, StreamToolInputStart)
"""Chunks that carry model output. The first one is the turn's first token."""


def is_model_output(chunk: StreamBaseResponse) -> bool:
    """Whether the chunk carries model output; a notice the platform streams
    as text, where the model gave no answer, does not."""
    return isinstance(chunk, MODEL_OUTPUT_CHUNKS) and not _is_platform_delta(chunk)


def is_answer_text(chunk: StreamBaseResponse) -> bool:
    """Whether the chunk is text of the model's answer. Whitespace on its own
    shows nothing, so it doesn't start the answer."""
    return (
        isinstance(chunk, StreamTextDelta)
        and bool(chunk.delta.strip())
        and not is_platform_text(chunk.delta)
    )


def _is_platform_delta(chunk: StreamBaseResponse) -> bool:
    return isinstance(chunk, StreamTextDelta) and is_platform_text(chunk.delta)


def _is_latest_user_message(messages: list["ChatMessage"], message: str) -> bool:
    """Whether *message* is already saved at the end of the chat, compared
    without the server blocks a saved copy can carry."""
    for row in reversed(messages):
        if row.role != "user":
            return False
        if strip_user_context_tags(row.content or "") == message:
            return True
    return False


class TurnTrace:
    """One turn's timings and outcome, recorded on its transaction."""

    def __init__(self, span: Span, requested_at: float = 0.0) -> None:
        self.span = span
        self._started_at = time.monotonic()
        self._requested_at = requested_at
        self._first_token_seen = False
        self._first_text_seen = False
        self._shape: str | None = None
        self._chunks = 0
        self._tool_calls = 0

    def describe_chat(self, session: "ChatSession", message: str | None = None) -> None:
        """Tag the kind of chat the turn is in, so its timings split by it: a
        chat's first turn sets up its sandbox and context, and an expert's
        turn loads the expert as well. The first turn answers the chat's first
        user message; a greeting or note the platform wrote before it doesn't
        make it a follow-up.

        ``message`` is the user message the turn answers, its server tags
        stripped. The chat route saves it before dispatch, sometimes behind a
        server block (a voice turn's instructions, a first turn's context); a
        sub-session doesn't, and the engine saves it when the turn starts.
        Either way it counts once. A turn with no user message of its own (a
        kickoff, a scheduled follow-up the scheduler stores as the model's, a
        continuation) is the first only in a chat nobody has written to yet."""
        try:
            users = sum(1 for row in session.messages if row.role == "user")
            if message is None:
                first = users == 0
            else:
                if message and not _is_latest_user_message(session.messages, message):
                    users += 1
                first = users <= 1
            self.span.set_tag("copilot.turn.first", "true" if first else "false")
            self.span.set_tag(
                "copilot.expert", "true" if session.expert_id else "false"
            )
        except Exception:
            logger.debug("Could not tag the turn's kind of chat", exc_info=True)

    def engine_started(self, engine: str) -> None:
        """Setup is done and the turn is handed to ``engine``."""
        self.span.set_tag("copilot.engine", engine)
        tracing.set_timing("copilot.turn.setup_ms", self._elapsed_ms(), span=self.span)

    def observe(self, chunk: StreamBaseResponse) -> None:
        """Account for one chunk the turn streamed."""
        self._chunks += 1
        if isinstance(chunk, StreamToolInputAvailable):
            self._tool_calls += 1
        if not self._first_token_seen and is_model_output(chunk):
            self._first_token_seen = True
            tracing.set_timing(
                "copilot.turn.ttft_ms", self._elapsed_ms(), span=self.span
            )
            self._time_since_request("copilot.turn.request_to_ttft_ms")
        if self._shape is None and isinstance(chunk, StreamToolInputStart):
            self._shape = "tool_first"
        if not self._first_text_seen and is_answer_text(chunk):
            self._first_text_seen = True
            self._shape = self._shape or "text_first"
            tracing.set_timing(
                "copilot.turn.first_text_ms", self._elapsed_ms(), span=self.span
            )
            self._time_since_request("copilot.turn.request_to_first_text_ms")

    def set_outcome(self, error_message: str | None, cancelled_message: str) -> None:
        """Tag how the turn ended, given the error it reports to the user."""
        if not error_message:
            outcome, status = "completed", "ok"
        elif error_message == cancelled_message:
            outcome, status = "cancelled", "cancelled"
        else:
            outcome, status = "failed", "internal_error"
        self.span.set_tag("copilot.turn.outcome", outcome)
        tracing.set_status(status, span=self.span)

    def close(self) -> None:
        # Whether the answer's text or a tool call came first: a turn that
        # runs tools before it answers can't show its answer as early.
        self.span.set_tag("copilot.turn.shape", self._shape or "none")
        tracing.set_timing(
            "copilot.turn.duration_ms", self._elapsed_ms(), span=self.span
        )
        tracing.set_data("copilot.turn.chunk_count", self._chunks, span=self.span)
        tracing.set_data(
            "copilot.turn.tool_call_count", self._tool_calls, span=self.span
        )

    def _elapsed_ms(self) -> float:
        return (time.monotonic() - self._started_at) * 1000

    def _time_since_request(self, name: str) -> None:
        """Time *name* from when the chat route received the message, which a
        turn started without one (a schedule, a parent turn) doesn't have."""
        if self._requested_at:
            tracing.set_timing(
                name, (time.time() - self._requested_at) * 1000, span=self.span
            )


@contextmanager
def turn_transaction(entry: "CoPilotExecutionEntry") -> Iterator[TurnTrace]:
    """Run one turn as a ``copilot.turn`` transaction, continuing the trace
    the entry was dispatched with."""
    with tracing.start_transaction(
        TURN_OP,
        TURN_OP,
        trace_headers=entry.trace_headers,
        tags={
            "copilot.llm_auth_provider": entry.llm_auth_provider,
            "copilot.model_tier": entry.model or "default",
            "copilot.turn.origin": _origin(entry),
        },
        data={
            key: value
            for key, value in {
                "copilot.session_id": entry.session_id,
                "copilot.turn_id": entry.turn_id,
                "copilot.user_id": entry.user_id,
            }.items()
            if value
        },
    ) as span:
        if entry.enqueued_at:
            tracing.set_timing(
                "copilot.turn.queue_wait_ms",
                (time.time() - entry.enqueued_at) * 1000,
                span=span,
            )
        turn = TurnTrace(span, requested_at=_prompt_request_time(entry, span))
        try:
            yield turn
        finally:
            turn.close()


def _prompt_request_time(entry: "CoPilotExecutionEntry", span: Span) -> float:
    """When the turn's chat request arrived, if the turn started promptly
    after it; 0 for a turn that waited (tagged) or had no request."""
    if not entry.request_arrival_at:
        return 0.0
    waited = time.time() - entry.request_arrival_at > WAITED_AFTER_S
    span.set_tag("copilot.turn.waited", "true" if waited else "false")
    return 0.0 if waited else entry.request_arrival_at


def _origin(entry: "CoPilotExecutionEntry") -> str:
    if entry.scheduled is not None or entry.unattended:
        return "scheduled"
    if entry.envelope is not None and entry.envelope.depth > 0:
        return "spawned"
    if not entry.is_user_message:
        return "system"
    return "chat"


@contextmanager
def tool_transaction(
    tool_name: str, tool_call_id: str, session_id: str
) -> Iterator[Span]:
    """Run one tool call as a ``copilot.tool`` transaction named after the
    tool, as a child of the turn that made it."""
    with tracing.start_transaction(
        TOOL_OP,
        tool_name,
        data={"copilot.tool_call_id": tool_call_id, "copilot.session_id": session_id},
    ) as span:
        yield span
