"""Cross-process helpers: dispatch + await a copilot session turn.

The sub-Otto tools (``run_sub_session``, ``get_sub_session_result``)
and ``AutoPilotBlock`` all delegate a copilot turn to the
``copilot_executor`` queue and then wait on the shared
``stream_registry`` for the terminal event. This module is the
centralised primitive so every caller agrees on the dispatch shape,
the event aggregation, and the cleanup contract.

:func:`wait_for_session_result` accumulates stream events into an
:class:`EventAccumulator` so callers get back ``response_text`` /
``tool_calls`` / token usage in memory without an extra DB round-trip.

:func:`run_copilot_turn_via_queue` is the one-shot "create session meta
→ enqueue → wait for result" sequence every caller uses.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC
from typing import TYPE_CHECKING, Any, Literal

from backend.copilot import stream_registry
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.executor.utils import queue_spawned_turn, schedule_turn
from backend.copilot.llm_attribution import is_turn_impersonated
from backend.copilot.model import (
    CHAT_STATUS_QUEUED,
    _get_session_lock,
    get_chat_session,
)
from backend.copilot.pending_message_helpers import is_turn_in_flight
from backend.copilot.response_model import StreamError, StreamFinish
from backend.copilot.tree import SpawnRequest, TreeRefusal
from backend.copilot.turn_pause import TurnPause
from backend.copilot.turn_queue import SessionNotIdle, queued_turn_refusal
from backend.data.db_accessors import chat_db

from .stream_accumulator import EventAccumulator, ToolCallEntry, process_event

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions

logger = logging.getLogger(__name__)


SessionOutcome = Literal[
    "completed",
    "failed",
    "running",
    # Waiting for one of the user's task slots; it starts on its own.
    "queued_for_slot",
    "rejected_concurrent_turn_cap",
    "refused",
]

# How often a waiter checks whether a queued turn has started.
_QUEUE_POLL_SECONDS = 1.0
_ALREADY_WAITING = (
    "That session already has a task waiting to start, so this task cannot be "
    "handed to it right now. Wait for it, or start a fresh one."
)


@dataclass
class SessionResult:
    """Aggregated result from a copilot session turn observed via
    ``stream_registry``.

    ``queued`` is set while the turn waits for one of the user's task
    slots (``queued_for_slot``); ``response_text`` is empty and the
    aggregate counts are zero in that case.
    """

    response_text: str = ""
    tool_calls: list[ToolCallEntry] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    queued: bool = False
    # Why the turn never started, when the outcome is ``refused``. Written
    # for the model: it names the bound that was hit.
    refusal: str = ""
    # Set when the turn ended as a pause. A turn the executor refused at
    # pickup for want of allowance (``started=False``) comes back as
    # ``refused``, with a ``refusal`` naming the limit: nothing ran.
    pause: TurnPause | None = None


async def wait_for_session_result(
    *,
    session_id: str,
    user_id: str | None,
    timeout: float,
) -> tuple[SessionOutcome, SessionResult]:
    """Drain the session's stream events and aggregate them into a result.

    Returns whatever has been observed at the cap (``running`` + partial
    result) or at the terminal event (``completed`` / ``failed`` + full
    result). Cleans up the subscriber listener on every exit path so
    long-running polls don't leak listeners (sentry r3105348640).
    """
    queue = await stream_registry.subscribe_to_session(
        session_id=session_id,
        user_id=user_id,
    )
    result = SessionResult()
    if queue is None:
        # Session meta not in Redis yet, or the caller doesn't own it.
        # ``subscribe_to_session`` already retried with backoff before
        # returning None.
        return "running", result

    acc = EventAccumulator()
    outcome: SessionOutcome = "running"
    try:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(timeout, 0)
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            _, event = await asyncio.wait_for(queue.get(), timeout=remaining)
            process_event(event, acc)
            if isinstance(event, StreamFinish):
                outcome = "completed"
                break
            if isinstance(event, StreamError):
                outcome = "failed"
                break
    except asyncio.TimeoutError:
        pass
    finally:
        await stream_registry.unsubscribe_from_session(
            session_id=session_id,
            subscriber_queue=queue,
        )

    result.response_text = "".join(acc.response_parts)
    result.tool_calls = list(acc.tool_calls)
    result.prompt_tokens = acc.prompt_tokens
    result.completion_tokens = acc.completion_tokens
    result.total_tokens = acc.total_tokens
    result.pause = pause = acc.pause
    if (
        outcome == "completed"
        and pause is not None
        and not pause.started
        and not result.response_text
        and not result.tool_calls
    ):
        # Only the executor's not-started pause: the turn never ran, and
        # reported as done the caller would take the empty answer for the
        # task's result. A turn that ran and then couldn't continue into
        # queued messages ends with the same pause after its own output,
        # and did complete.
        outcome = "refused"
        result.refusal = _budget_refusal(pause)
    return outcome, result


def _budget_refusal(pause: TurnPause) -> str:
    """What the caller's model is told about a turn refused for budget."""
    if pause.resets_at is None:
        return f"{pause.summary} The AutoPilot budget doesn't reset on its own."
    resets_at = pause.resets_at.astimezone(UTC)
    return (
        f"{pause.summary} The AutoPilot budget resets "
        f"{resets_at:%Y-%m-%d %H:%M} UTC."
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
    spawn: SpawnRequest | None = None,
    message_metadata: dict[str, Any] | None = None,
) -> tuple[SessionOutcome, SessionResult]:
    """Dispatch a copilot turn onto the queue and wait for its result.

    ``spawn`` is what the caller asks for the child's tree envelope; the
    chokepoint clamps it. A target session with a turn in flight, or with a
    task already waiting for a slot, is ``refused``: a prompt never rides a
    turn that is running under a different envelope. Continuing a
    delegation to a teammate who is still working (``delegate_to_expert``
    with ``delegated_session_id``) gets a message telling the caller to wait
    or start fresh, and ``AutoPilotBlock`` raises on ``refused``.

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
    """
    session = await get_chat_session(session_id, user_id)
    if session is None:
        raise RuntimeError("copilot_session_not_found")

    if await chat_db().get_chat_session_status(session_id) == CHAT_STATUS_QUEUED:
        return "refused", SessionResult(refusal=_ALREADY_WAITING)
    if await is_turn_in_flight(session_id):
        return "refused", SessionResult(
            refusal=(
                "That session already has a turn in flight, so this task "
                "cannot be handed to it right now. Wait for it to finish, "
                "or start a fresh one."
            )
        )

    # Same per-user concurrent-turn cap as the HTTP chat route via the
    # shared ``schedule_turn`` helper — without this a graph spawning N
    # AutoPilotBlocks or a tool-loop firing run_sub_session could fan
    # out unbounded copilot turns past the user's cap.
    turn_id = str(uuid.uuid4())
    delegated = session.metadata.delegated_by_session_id is not None
    try:
        await schedule_turn(
            session_id=session_id,
            user_id=user_id,
            turn_id=turn_id,
            message=message,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            permissions=permissions,
            spawn=spawn,
            message_metadata=message_metadata,
            delegated=delegated,
            # A child of an impersonated turn is the admin's work too.
            impersonated=is_turn_impersonated(),
        )
    except TreeRefusal as refused:
        return "refused", SessionResult(refusal=refused.message)
    except ConcurrentTurnLimitError:
        if delegated:
            return await _queue_for_slot(
                session_id=session_id,
                user_id=user_id,
                message=message,
                timeout=timeout,
                permissions=permissions,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                spawn=spawn,
                message_metadata=message_metadata,
                llm_auth_provider=session.metadata.llm_auth_provider,
                llm_credential_id=session.metadata.llm_credential_id,
            )
        # Sub-Otto / run_sub_session caller is at the cap (this is
        # the graph-block / tool path, not the HTTP route). Use a
        # distinct ``rejected_concurrent_turn_cap`` outcome so callers
        # render an actionable "wait for an in-flight turn to finish"
        # error instead of a generic "see transcript" pointer (the
        # session was never created, so the transcript is empty).
        logger.warning(
            "[queue] session=%s user=%s rejected by concurrent-turn cap (tool=%s)",
            session_id[:12],
            user_id[:8] if user_id else "?",
            tool_name,
        )
        return "rejected_concurrent_turn_cap", SessionResult()
    return await wait_for_session_result(
        session_id=session_id,
        user_id=user_id,
        timeout=timeout,
    )


async def wait_for_queued_session(
    *, session_id: str, user_id: str, timeout: float
) -> tuple[SessionOutcome, SessionResult]:
    """Wait for a queued turn to start, then for its result, all within
    ``timeout``. A turn closed at promotion, or cancelled before it started,
    comes back ``refused`` with the note that closed it."""
    deadline = time.monotonic() + timeout
    while await chat_db().get_chat_session_status(session_id) == CHAT_STATUS_QUEUED:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "queued_for_slot", SessionResult(queued=True)
        await asyncio.sleep(min(_QUEUE_POLL_SECONDS, remaining))
    # A cancel or a refusal writes its note under this lock with the flip seen.
    async with _get_session_lock(session_id):
        session = await get_chat_session(session_id, user_id)
    refusal = queued_turn_refusal(session) if session else None
    if refusal is not None:
        return "refused", SessionResult(refusal=refusal)
    return await wait_for_session_result(
        session_id=session_id,
        user_id=user_id,
        timeout=max(deadline - time.monotonic(), 0),
    )


async def _queue_for_slot(
    *,
    session_id: str,
    user_id: str,
    message: str,
    timeout: float,
    permissions: "CopilotPermissions | None",
    tool_call_id: str,
    tool_name: str,
    spawn: SpawnRequest | None,
    message_metadata: dict[str, Any] | None,
    llm_auth_provider: CopilotLlmAuthProvider,
    llm_credential_id: str | None,
) -> tuple[SessionOutcome, SessionResult]:
    """A spawn the reserve refused waits in the queue instead, up to the
    inflight cap; past it the spawn is refused as before."""
    try:
        await queue_spawned_turn(
            session_id=session_id,
            user_id=user_id,
            message=message,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            llm_auth_provider=llm_auth_provider,
            llm_credential_id=llm_credential_id,
            permissions=permissions,
            spawn=spawn,
            message_metadata=message_metadata,
            impersonated=is_turn_impersonated(),
        )
    except TreeRefusal as refused:
        return "refused", SessionResult(refusal=refused.message)
    except SessionNotIdle:
        return "refused", SessionResult(refusal=_ALREADY_WAITING)
    except ConcurrentTurnLimitError:
        logger.warning(
            f"[queue] session={session_id[:12]} user={user_id[:8]} rejected at the "
            f"inflight cap (tool={tool_name})"
        )
        return "rejected_concurrent_turn_cap", SessionResult()
    return await wait_for_queued_session(
        session_id=session_id, user_id=user_id, timeout=timeout
    )
