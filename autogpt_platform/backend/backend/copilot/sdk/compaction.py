"""Compaction tracking for SDK-based chat sessions.

Encapsulates the state machine and event emission for context compaction,
both pre-query (history compressed before SDK query) and SDK-internal
(PreCompact hook fires mid-stream).
"""

import asyncio
import json
import logging

from ..constants import COMPACTION_DONE_MSG
from ..model import ChatMessage, ChatSession
from ..response_model import (
    StreamBaseResponse,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    compaction_end_events,
    compaction_events,
    compaction_start_events,
)

logger = logging.getLogger(__name__)


class CompactionTracker:
    """Tracks compaction state and yields UI events.

    There are two compaction paths:

    1. **Pre-query** — ``_compress_conversation_history`` truncates messages
       before the SDK query starts.  Call :meth:`emit_pre_query` to yield a
       self-contained compaction tool call.

    2. **SDK-internal** — the ``PreCompact`` hook fires mid-stream (the SDK
       compacts its own context).  The hook sets :attr:`compact_start` via
       ``on_compact``.  Call :meth:`emit_start_if_ready` during heartbeat
       ticks, and :meth:`emit_end_if_ready` when a message arrives, to yield
       start/end events separately (for a live spinner in the UI).
    """

    def __init__(self) -> None:
        # asyncio.Event set by the PreCompact hook callback
        self.compact_start = asyncio.Event()
        # True after "Summarizing…" start events emitted
        self._start_emitted = False
        # True after the full compaction cycle (start+end) emitted
        self._done = False
        # Tool-call ID for matching start → end events
        self._tool_call_id = ""

    @property
    def on_compact(self):  # noqa: ANN201 — returns bound method
        """Callback for the PreCompact hook — ``on_compact=tracker.on_compact``."""
        return self.compact_start.set

    # ------------------------------------------------------------------
    # Pre-query compaction
    # ------------------------------------------------------------------

    def emit_pre_query(self, session: ChatSession) -> list[StreamBaseResponse]:
        """Return a self-contained compaction tool call for pre-query compression.

        Also persists the events to ``session.messages`` so they survive a page
        refresh.
        """
        evts = compaction_events(COMPACTION_DONE_MSG)
        _persist_compaction(session, evts)
        return evts

    # ------------------------------------------------------------------
    # SDK-internal compaction (two-phase: start during heartbeat, end on msg)
    # ------------------------------------------------------------------

    def reset_for_query(self) -> None:
        """Reset per-query state so a pre-query compaction doesn't suppress
        SDK-internal notifications."""
        self._done = False
        self._tool_call_id = ""

    def emit_start_if_ready(self) -> list[StreamBaseResponse]:
        """If the PreCompact hook fired, emit start events (spinning tool).

        Call this during heartbeat ticks.  Returns an empty list when not ready.
        """
        if self.compact_start.is_set() and not self._start_emitted and not self._done:
            self.compact_start.clear()
            self._start_emitted = True
            self._tool_call_id, start_evts = compaction_start_events()
            return start_evts
        return []

    async def emit_end_if_ready(self, session: ChatSession) -> list[StreamBaseResponse]:
        """If compaction is in progress, emit end events.

        Call this after receiving an SDK message.  Yields to the event loop
        first (``await asyncio.sleep(0)``) so any pending hook tasks scheduled
        via ``start_soon`` can set ``compact_start`` before we inspect it.

        Persists the full event sequence to ``session.messages``.
        """
        # Give the hook task a chance to execute
        await asyncio.sleep(0)

        if self._done:
            return []
        if not self._start_emitted and not self.compact_start.is_set():
            return []

        if self._start_emitted:
            # Close the open tool call
            done_events = compaction_end_events(self._tool_call_id, COMPACTION_DONE_MSG)
            # For persistence, build the full event list
            # (start events were already yielded)
            all_events = compaction_events(COMPACTION_DONE_MSG)
        else:
            # PreCompact fired but we never emitted start — emit a
            # self-contained compaction tool call.
            done_events = compaction_events(COMPACTION_DONE_MSG)
            all_events = done_events

        self.compact_start.clear()
        self._start_emitted = False
        self._done = True
        _persist_compaction(session, all_events)
        return done_events


def _persist_compaction(session: ChatSession, events: list[StreamBaseResponse]) -> None:
    """Record compaction tool-call + result into the session so it survives refresh.

    Compaction events are synthetic (not real SDK tool calls), so they bypass the
    normal message accumulation in the adapter.  This explicitly appends them as
    assistant + tool messages to ``session.messages``.
    """
    tool_call_id = ""
    tool_name = ""
    output = ""
    for ev in events:
        if isinstance(ev, StreamToolInputAvailable):
            tool_call_id = ev.toolCallId
            tool_name = ev.toolName
        elif isinstance(ev, StreamToolOutputAvailable):
            output = ev.output if isinstance(ev.output, str) else json.dumps(ev.output)
    if not tool_call_id:
        return
    session.messages.append(
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        )
    )
    session.messages.append(
        ChatMessage(role="tool", content=output, tool_call_id=tool_call_id)
    )
