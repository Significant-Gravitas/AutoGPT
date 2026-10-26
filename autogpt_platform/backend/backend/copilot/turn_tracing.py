"""Sentry transactions for copilot turns and their tool calls.

A turn runs in the copilot executor as one ``copilot.turn`` transaction that
continues the trace of whatever dispatched it: the chat request, a parent
turn's tool, a graph node. Each tool call the turn makes is a ``copilot.tool``
transaction of its own in that trace, so a long autopilot turn stays under the
SDK's span limit however many tools it runs.

A turn that reached its engine also ends as an entry in PostHog's LLM
analytics: the ``$ai_trace`` of a reply, or the ``$ai_span`` of a subtask in
its reply's trace (see ``copilot.ai_spans``).
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sentry_sdk.tracing import Span

from backend.copilot import ai_spans
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolInputStart,
)
from backend.util import tracing

if TYPE_CHECKING:
    from backend.copilot.executor.utils import CoPilotExecutionEntry

TURN_OP = "copilot.turn"
TOOL_OP = "copilot.tool"

MODEL_OUTPUT_CHUNKS = (StreamTextDelta, StreamReasoningDelta, StreamToolInputStart)
"""Chunks that carry model output. The first one is the turn's first token."""


class TurnTrace:
    """One turn's timings and outcome, recorded on its transaction."""

    def __init__(self, span: Span) -> None:
        self.span = span
        self._started_at = time.monotonic()
        self._first_token_seen = False
        self._chunks = 0
        self._tool_calls = 0
        self.engine: str | None = None
        self.outcome = "completed"
        self.error_code: str | None = None

    def engine_started(self, engine: str) -> None:
        """Setup is done and the turn is handed to ``engine``."""
        self.engine = engine
        self.span.set_tag("copilot.engine", engine)
        tracing.set_timing("copilot.turn.setup_ms", self._elapsed_ms(), span=self.span)

    def observe(self, chunk: StreamBaseResponse) -> None:
        """Account for one chunk the turn streamed."""
        self._chunks += 1
        if isinstance(chunk, StreamToolInputAvailable):
            self._tool_calls += 1
        if not self._first_token_seen and isinstance(chunk, MODEL_OUTPUT_CHUNKS):
            self._first_token_seen = True
            tracing.set_timing(
                "copilot.turn.ttft_ms", self._elapsed_ms(), span=self.span
            )

    def set_outcome(
        self,
        error_message: str | None,
        cancelled_message: str,
        error_code: str | None = None,
    ) -> None:
        """Tag how the turn ended, given the error it reports to the user
        and the code it reports with it."""
        if not error_message:
            outcome, status = "completed", "ok"
        elif error_message == cancelled_message:
            outcome, status = "cancelled", "cancelled"
        else:
            outcome, status = "failed", "internal_error"
        self.outcome = outcome
        self.error_code = error_code
        self.span.set_tag("copilot.turn.outcome", outcome)
        tracing.set_status(status, span=self.span)

    def close(self) -> None:
        tracing.set_timing(
            "copilot.turn.duration_ms", self._elapsed_ms(), span=self.span
        )
        tracing.set_data("copilot.turn.chunk_count", self._chunks, span=self.span)
        tracing.set_data(
            "copilot.turn.tool_call_count", self._tool_calls, span=self.span
        )

    def report_to_posthog(self, entry: "CoPilotExecutionEntry") -> None:
        """Send the turn's end to PostHog's LLM analytics. A turn that never
        reached its engine made no model or tool calls, so it sends none."""
        if self.engine is None:
            return
        ai_spans.capture_turn(
            entry,
            latency_seconds=self._elapsed_ms() / 1000,
            outcome=self.outcome,
            error_code=self.error_code,
            engine=self.engine,
            origin=_origin(entry),
            tool_call_count=self._tool_calls,
        )

    def _elapsed_ms(self) -> float:
        return (time.monotonic() - self._started_at) * 1000


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
        turn = TurnTrace(span)
        try:
            yield turn
        finally:
            turn.close()
            turn.report_to_posthog(entry)


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
