"""Cross-process helpers: dispatch + await a copilot session turn.

The sub-AutoPilot tools (``run_sub_session``, ``get_sub_session_result``)
and ``AutoPilotBlock`` all delegate a copilot turn to the
``copilot_executor`` queue and then wait on the shared
``stream_registry`` for the terminal event. This module is the
centralised primitive so every caller agrees on the dispatch shape,
the event aggregation, and the cleanup contract.

Two wait modes:

* :func:`wait_for_session_completion` — cheap "did it finish?" when the
  caller only needs a ``SessionOutcome`` (``running`` / ``completed`` /
  ``failed``). Used by ``get_sub_session_result`` when it only needs to
  decide between returning the final ChatSession state or "still busy".
* :func:`wait_for_session_result` — accumulates stream events into an
  :class:`EventAccumulator` so the caller also gets back
  ``response_text`` / ``tool_calls`` / token usage in memory, without
  an extra DB round-trip. Used by the full-result callers
  (``run_sub_session`` completed path, ``AutoPilotBlock.execute_copilot``).

Plus :func:`run_copilot_turn_via_queue` — the one-shot "create session
meta → enqueue → wait for result" sequence that every caller uses.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from backend.copilot import stream_registry
from backend.copilot.executor.utils import enqueue_copilot_turn
from backend.copilot.pending_message_helpers import (
    is_turn_in_flight,
    queue_user_message,
)
from backend.copilot.response_model import StreamError, StreamFinish

from .stream_accumulator import EventAccumulator, ToolCallEntry, process_event

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions

logger = logging.getLogger(__name__)


SessionOutcome = Literal["completed", "failed", "running", "queued"]


@dataclass
class SessionResult:
    """Aggregated result from a copilot session turn observed via
    ``stream_registry``. Mirrors :class:`collect.CopilotResult` so both
    in-process and cross-process consumers get the same shape.

    When ``queued`` is set, :func:`run_copilot_turn_via_queue` detected an
    in-flight turn on the target session and pushed the message onto the
    pending buffer instead of starting a new turn.  ``response_text`` is
    empty and the aggregate counts are zero in that case; the executor
    running the earlier turn drains the buffer on its next round.
    """

    response_text: str = ""
    tool_calls: list[ToolCallEntry] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error_text: str | None = None
    queued: bool = False
    pending_buffer_length: int = 0


async def wait_for_session_completion(
    *,
    session_id: str,
    user_id: str | None,
    timeout: float,
) -> SessionOutcome:
    """Return the outcome of the latest turn on *session_id* within *timeout*.

    Light-weight variant of :func:`wait_for_session_result` — drops the
    event aggregation so callers that only need to decide between "still
    running" and "terminal" don't pay for building an accumulator.
    """
    outcome, _ = await _drain_until_terminal(
        session_id=session_id,
        user_id=user_id,
        timeout=timeout,
        accumulate=False,
    )
    return outcome


async def wait_for_session_result(
    *,
    session_id: str,
    user_id: str | None,
    timeout: float,
) -> tuple[SessionOutcome, SessionResult]:
    """Drain the session's stream events, aggregate them into a result.

    Returns whatever has been observed at the cap (``running`` + partial
    result) or at the terminal event (``completed`` / ``failed`` + full
    result). Cleans up the subscriber listener on every exit path so
    long-running polls don't leak listeners (sentry r3105348640).
    """
    outcome, acc = await _drain_until_terminal(
        session_id=session_id,
        user_id=user_id,
        timeout=timeout,
        accumulate=True,
    )
    result = SessionResult()
    if acc is not None:
        result.response_text = "".join(acc.response_parts)
        result.tool_calls = list(acc.tool_calls)
        result.prompt_tokens = acc.prompt_tokens
        result.completion_tokens = acc.completion_tokens
        result.total_tokens = acc.total_tokens
    return outcome, result


async def _drain_until_terminal(
    *,
    session_id: str,
    user_id: str | None,
    timeout: float,
    accumulate: bool,
) -> tuple[SessionOutcome, EventAccumulator | None]:
    """Shared drain loop used by both wait helpers."""
    queue = await stream_registry.subscribe_to_session(
        session_id=session_id,
        user_id=user_id,
    )
    if queue is None:
        # Session meta not in Redis yet, or the caller doesn't own it.
        # ``subscribe_to_session`` already retried with backoff before
        # returning None.
        return "running", (EventAccumulator() if accumulate else None)

    acc = EventAccumulator() if accumulate else None
    try:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(timeout, 0)
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return "running", acc
            event = await asyncio.wait_for(queue.get(), timeout=remaining)
            if accumulate and acc is not None:
                process_event(event, acc)
            if isinstance(event, StreamFinish):
                return "completed", acc
            if isinstance(event, StreamError):
                return "failed", acc
    except asyncio.TimeoutError:
        return "running", acc
    finally:
        await stream_registry.unsubscribe_from_session(
            session_id=session_id,
            subscriber_queue=queue,
        )


async def run_copilot_turn_via_queue(
    *,
    session_id: str,
    user_id: str,
    message: str,
    timeout: float,
    permissions: "CopilotPermissions | None" = None,
    tool_call_id: str,
    tool_name: str,
) -> tuple[SessionOutcome, SessionResult]:
    """Dispatch a copilot turn onto the queue and wait for its result.

    The canonical invocation path shared by ``run_sub_session`` (the
    copilot tool), ``AutoPilotBlock`` (the graph block), and any future
    caller that needs to run a copilot turn without occupying its own
    worker with the SDK stream:

    1. Create a ``stream_registry`` session meta record for the turn.
    2. Enqueue a ``CoPilotExecutionEntry`` on the copilot_execution
       exchange. Any idle copilot_executor worker claims it.
    3. Subscribe to the session's Redis stream and drain events until
       ``StreamFinish`` / ``StreamError`` or the cap fires.

    ``tool_call_id`` / ``tool_name`` disambiguate who originated the
    turn in observability / replay (e.g. ``"sub:<parent>"`` for a
    sub-session, ``"autopilot_block"`` for an AutoPilotBlock run).

    Self-defensive queue-fallback: if the target session already has a
    turn running (another ``run_sub_session`` / AutoPilot block / UI
    chat), don't race it on the cluster lock.  Push the message onto the
    pending buffer so the existing turn drains it at its next round
    boundary, then:

    * ``timeout == 0`` — return immediately with
      ``("queued", SessionResult(queued=True, ...))``.  Callers that
      explicitly opted into fire-and-forget (``run_sub_session`` with
      ``wait_for_result=0``) use this to bail without waiting.
    * ``timeout > 0`` — **subscribe to the in-flight turn's stream and
      return its aggregated result** (exactly the same shape as a
      normally-dispatched turn, but with ``result.queued=True`` so
      callers can tell we rode on someone else's turn).  Semantically
      identical to "I asked the session to do something and here is
      what happened next"; no separate deferred-state branch needed in
      ``run_sub_session`` / ``AutoPilotBlock``.
    """
    if await is_turn_in_flight(session_id):
        logger.info(
            "[queue] session=%s has a turn in flight; queueing message "
            "(tool=%s) into pending buffer instead of starting a new turn",
            session_id[:12],
            tool_name,
        )
        state = await queue_user_message(session_id=session_id, message=message)
        if timeout <= 0:
            # Fire-and-forget: caller explicitly asked not to wait.
            return "queued", SessionResult(
                queued=True, pending_buffer_length=state.buffer_length
            )
        # Ride the in-flight turn: subscribe to its stream and return the
        # same aggregated result shape as a fresh dispatch.  The model
        # drains the pending buffer between tool rounds (baseline) or at
        # the next tool boundary via the PostToolUse hook (SDK), so the
        # response we observe will reflect our queued follow-up (or be
        # the terminal result if the in-flight turn finishes before the
        # buffer drains — in that case ``result.queued=True`` is still
        # the correct signal for the caller).
        outcome, observed = await wait_for_session_result(
            session_id=session_id,
            user_id=user_id,
            timeout=timeout,
        )
        observed.queued = True
        observed.pending_buffer_length = state.buffer_length
        return outcome, observed

    turn_id = str(uuid.uuid4())
    await stream_registry.create_session(
        session_id=session_id,
        user_id=user_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        turn_id=turn_id,
    )
    await enqueue_copilot_turn(
        session_id=session_id,
        user_id=user_id,
        message=message,
        turn_id=turn_id,
        permissions=permissions,
    )
    return await wait_for_session_result(
        session_id=session_id,
        user_id=user_id,
        timeout=timeout,
    )
