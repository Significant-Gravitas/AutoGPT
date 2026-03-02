"""Compaction tracking for SDK-based chat sessions.

Encapsulates the state machine and event emission for context compaction,
both pre-query (history compressed before SDK query) and SDK-internal
(PreCompact hook fires mid-stream).

All compaction-related helpers live here: event builders, message filtering,
persistence, and the ``CompactionTracker`` state machine.
"""

import asyncio
import json
import logging
import uuid

from ..constants import COMPACTION_DONE_MSG
from ..model import ChatMessage, ChatSession
from ..response_model import (
    StreamBaseResponse,
    StreamFinishStep,
    StreamStartStep,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPACTION_TOOL_NAME = "context_compaction"

# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def compaction_start_events() -> tuple[str, list[StreamBaseResponse]]:
    """Emit tool-call-start events for compaction — renders as a spinning tool.

    Returns ``(tool_call_id, events)`` so the caller can close with the same id.
    """
    tool_call_id = f"compaction-{uuid.uuid4().hex[:12]}"
    return tool_call_id, [
        StreamStartStep(),
        StreamToolInputStart(toolCallId=tool_call_id, toolName=COMPACTION_TOOL_NAME),
        StreamToolInputAvailable(
            toolCallId=tool_call_id, toolName=COMPACTION_TOOL_NAME, input={}
        ),
    ]


def compaction_end_events(tool_call_id: str, message: str) -> list[StreamBaseResponse]:
    """Close a compaction tool call with the result message."""
    return [
        StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName=COMPACTION_TOOL_NAME,
            output=message,
        ),
        StreamFinishStep(),
    ]


def compaction_events(
    message: str, tool_call_id: str | None = None
) -> list[StreamBaseResponse]:
    """Emit a self-contained compaction tool call (already completed).

    When *tool_call_id* is provided it is reused (e.g. for persistence that
    must match an already-streamed start event).  Otherwise a new random ID
    is generated.
    """
    if tool_call_id is None:
        tool_call_id, start = compaction_start_events()
    else:
        start = [
            StreamStartStep(),
            StreamToolInputStart(
                toolCallId=tool_call_id, toolName=COMPACTION_TOOL_NAME
            ),
            StreamToolInputAvailable(
                toolCallId=tool_call_id, toolName=COMPACTION_TOOL_NAME, input={}
            ),
        ]
    return start + compaction_end_events(tool_call_id, message)


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------


def is_compaction_tool_call(tool_call: dict) -> bool:
    """Check if a tool call dict is a synthetic compaction call."""
    return tool_call.get("function", {}).get("name") == COMPACTION_TOOL_NAME


def filter_compaction_messages(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """Remove synthetic compaction tool-call messages (UI-only artifacts).

    Strips assistant messages whose only tool calls are compaction calls,
    and their corresponding tool-result messages.
    """
    compaction_ids: set[str] = set()
    filtered: list[ChatMessage] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if is_compaction_tool_call(tc):
                    compaction_ids.add(tc.get("id", ""))
            non_compaction = [
                tc for tc in msg.tool_calls if not is_compaction_tool_call(tc)
            ]
            if not non_compaction and not msg.content:
                continue
        if msg.role == "tool" and msg.tool_call_id in compaction_ids:
            continue
        filtered.append(msg)
    return filtered


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_compaction(session: ChatSession, events: list[StreamBaseResponse]) -> None:
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


# ---------------------------------------------------------------------------
# CompactionTracker — state machine for streaming sessions
# ---------------------------------------------------------------------------


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
        self.compact_start = asyncio.Event()
        self._start_emitted = False
        self._done = False
        self._tool_call_id = ""

    @property
    def on_compact(self):  # noqa: ANN201
        """Callback for the PreCompact hook."""
        return self.compact_start.set

    # ------------------------------------------------------------------
    # Pre-query compaction
    # ------------------------------------------------------------------

    def emit_pre_query(self, session: ChatSession) -> list[StreamBaseResponse]:
        """Return a self-contained compaction tool call for pre-query compression.

        Persists the events to ``session.messages`` and sets ``_done`` so the
        SDK-internal path doesn't double-notify for the same turn.
        """
        evts = compaction_events(COMPACTION_DONE_MSG)
        persist_compaction(session, evts)
        self._done = True
        return evts

    # ------------------------------------------------------------------
    # SDK-internal compaction (two-phase: start during heartbeat, end on msg)
    # ------------------------------------------------------------------

    def reset_for_query(self) -> None:
        """Reset per-query state for a new SDK query."""
        self._done = False
        self._start_emitted = False
        self._tool_call_id = ""

    def emit_start_if_ready(self) -> list[StreamBaseResponse]:
        """If the PreCompact hook fired, emit start events (spinning tool).

        Call this during heartbeat ticks.
        """
        if self.compact_start.is_set() and not self._start_emitted and not self._done:
            self.compact_start.clear()
            self._start_emitted = True
            self._tool_call_id, start_evts = compaction_start_events()
            return start_evts
        return []

    async def emit_end_if_ready(self, session: ChatSession) -> list[StreamBaseResponse]:
        """If compaction is in progress, emit end events.

        Yields to the event loop first so any pending hook tasks can set
        ``compact_start`` before we inspect it.  Persists the full event
        sequence to ``session.messages``.
        """
        await asyncio.sleep(0)

        if self._done:
            return []
        if not self._start_emitted and not self.compact_start.is_set():
            return []

        if self._start_emitted:
            done_events = compaction_end_events(self._tool_call_id, COMPACTION_DONE_MSG)
            all_events = compaction_events(
                COMPACTION_DONE_MSG, tool_call_id=self._tool_call_id
            )
        else:
            done_events = compaction_events(COMPACTION_DONE_MSG)
            all_events = done_events

        self.compact_start.clear()
        self._start_emitted = False
        self._done = True
        persist_compaction(session, all_events)
        return done_events
