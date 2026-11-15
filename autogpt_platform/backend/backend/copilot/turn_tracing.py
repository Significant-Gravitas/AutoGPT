"""Sentry transactions for copilot turns and their tool calls.

A turn runs in the copilot executor as one ``copilot.turn`` transaction that
continues the trace of whatever dispatched it: the chat request, a parent
turn's tool, a graph node. Each tool call the turn makes is a ``copilot.tool``
transaction of its own in that trace, so a long autopilot turn stays under the
SDK's span limit however many tools it runs.

A turn failed when it ended on an error its user was shown, with no reply
after it: an error frame, whether the engine streamed it or the executor
published it for an exception, or a notice an engine streams in place of a
reply (``NO_REPLY_NOTICES``). A tool error the model went on from is not a
failure, nor is a pause at a limit; a turn its user stopped is cancelled.

A turn that reached its engine also ends as an entry in PostHog's LLM
analytics: the ``$ai_trace`` of a reply, or the ``$ai_span`` of a subtask in
its reply's trace (see ``copilot.ai_spans``). Every turn the executor ran is
counted when it ends, on ``copilot_executor_turns_total``.
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sentry_sdk.tracing import Span

from backend.copilot import ai_spans
from backend.copilot.constants import (
    AUTOPILOT_BLOCK_TOOL_NAME,
    EMPTY_COMPLETION_NOTICE,
    STREAM_INCOMPLETE_MARKER,
)
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolInputStart,
)
from backend.copilot.turn_metrics import turns_counter
from backend.util import tracing

if TYPE_CHECKING:
    from backend.copilot.executor.utils import CoPilotExecutionEntry
    from backend.copilot.model import ChatSessionOrigin

logger = logging.getLogger(__name__)

TURN_OP = "copilot.turn"
TOOL_OP = "copilot.tool"

MODEL_OUTPUT_CHUNKS = (StreamTextDelta, StreamReasoningDelta, StreamToolInputStart)
"""Chunks that carry model output. The first one that isn't a notice in place
of a reply (``NO_REPLY_NOTICES``) is the turn's first token."""

NO_REPLY_NOTICES: dict[str, str] = {
    STREAM_INCOMPLETE_MARKER: "stream_incomplete",
    EMPTY_COMPLETION_NOTICE: "empty_completion",
}
"""What an engine streams as text in place of a reply it doesn't have, and the
code a turn that ends on it fails with: the SDK engine's notice for a CLI that
ended without a result, and the baseline engine's for a model that finished
without a word (which the SDK engine reports as an error frame)."""


class TurnTrace:
    """One turn's timings and outcome, recorded on its transaction."""

    def __init__(self, span: Span) -> None:
        self.span = span
        self._started_at = time.monotonic()
        self._first_token_seen = False
        self._chunks = 0
        self._tool_calls = 0
        # The code of the notice the turn's output last ended on, if any.
        self._no_reply: str | None = None
        self._ended = False
        self._claimed_elsewhere = False
        self.engine: str | None = None
        self.load_test: str | None = None
        self.origin: str | None = None
        self.outcome = "completed"
        self.error_code: str | None = None

    def load_test_account(self, engine: str) -> None:
        """The turn is a load-test account's, run on *engine*
        (``backend/copilot/load_test.py``), so it can be told apart from
        people's turns even when it fails before its engine starts."""
        self.load_test = engine
        self.span.set_tag("copilot.load_test", engine)

    def engine_started(self, engine: str) -> None:
        """Setup is done and the turn is handed to ``engine``."""
        self.engine = engine
        self.span.set_tag("copilot.engine", engine)
        tracing.set_timing("copilot.turn.setup_ms", self._elapsed_ms(), span=self.span)

    def set_origin(self, origin: str) -> None:
        """What started the turn, once its chat session is known."""
        self.origin = origin
        self.span.set_tag("copilot.turn.origin", origin)

    def observe(self, chunk: StreamBaseResponse) -> None:
        """Account for one chunk the turn streamed."""
        self._chunks += 1
        if isinstance(chunk, StreamToolInputAvailable):
            self._tool_calls += 1
        if not isinstance(chunk, MODEL_OUTPUT_CHUNKS):
            return
        # The latest output decides: a notice in place of a reply ends the
        # turn without one unless the model writes on after it.
        self._no_reply = (
            NO_REPLY_NOTICES.get(chunk.delta)
            if isinstance(chunk, StreamTextDelta)
            else None
        )
        if self._no_reply is None and not self._first_token_seen:
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
        and the code it reports with it. A turn that reports none still
        failed when its output ended on a notice in place of a reply, under
        that notice's code."""
        if error_message and error_message == cancelled_message:
            self._end("cancelled", "cancelled", error_code)
        elif error_message:
            self._end("failed", "internal_error", error_code)
        elif self._no_reply is not None:
            self._end("failed", "internal_error", self._no_reply)
        else:
            self._end("completed", "ok", error_code)

    def claimed_elsewhere(self) -> None:
        """Another worker claimed this background turn and reports its
        outcome, so this worker records none for it, even when closing its
        own pickup raises."""
        self._claimed_elsewhere = True

    def raised(self, error: BaseException) -> None:
        """End a turn that raised before its outcome was set: cancelled by a
        cancellation, failed by anything else."""
        if self._ended or self._claimed_elsewhere:
            return
        if isinstance(error, (asyncio.CancelledError, GeneratorExit)):
            self._end("cancelled", "cancelled", None)
        else:
            self._end("failed", "internal_error", None)

    def _end(self, outcome: str, status: str, error_code: str | None) -> None:
        self._ended = True
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
            origin=self.origin or turn_origin(entry),
            tool_call_count=self._tool_calls,
            load_test=self.load_test,
        )

    def report_to_prometheus(self, entry: "CoPilotExecutionEntry") -> None:
        """Count the turn's end on ``copilot_executor_turns_total``. A turn
        that never reached its engine is counted under ``engine="none"``, so
        it never reads as a model turn. A pickup that ended without an
        outcome ran nothing here (another worker owns its turn), so it is not
        counted."""
        if not self._ended:
            return
        try:
            turns_counter.labels(
                outcome=self.outcome,
                engine=self.engine or "none",
                origin=self.origin or turn_origin(entry),
                load_test=self.load_test or "",
            ).inc()
        except Exception:
            logger.warning("Failed to count the turn's end", exc_info=True)

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
            "copilot.turn.origin": turn_origin(entry),
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
        except BaseException as error:
            turn.raised(error)
            raise
        finally:
            turn.close()
            turn.report_to_posthog(entry)
            turn.report_to_prometheus(entry)


def turn_origin(
    entry: "CoPilotExecutionEntry",
    session_origin: "ChatSessionOrigin | None" = None,
) -> str:
    """What started a turn: ``scheduled`` (a schedule, or a turn nobody is
    watching), ``spawned`` (a subtask), ``automation`` (a workflow's AutoPilot
    block, whose prompts are machine-written), ``system`` (the platform, not a
    message) or ``chat`` (a message someone sent).

    The block is known by what dispatched the turn (``entry.tool_name``), so a
    message someone types into a chat the block opened stays ``chat``. An
    entry queued before ``tool_name`` existed falls back to the chat's
    origin, which the executor passes once the session is loaded."""
    if entry.scheduled is not None or entry.unattended:
        return "scheduled"
    if entry.envelope is not None and entry.envelope.depth > 0:
        return "spawned"
    if entry.tool_name == AUTOPILOT_BLOCK_TOOL_NAME:
        return "automation"
    if entry.tool_name is None and session_origin == "automation":
        return "automation"
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
