"""Compaction tracking for SDK-based chat sessions.

Encapsulates the state machine and event emission for context compaction,
both pre-query (history compressed before SDK query) and SDK-internal
(PreCompact hook fires mid-stream).

All compaction-related helpers live here: event builders, message filtering,
persistence, and the ``CompactionTracker`` state machine.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..constants import COMPACTION_DONE_MSG, COMPACTION_TOOL_NAME
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


@dataclass
class CompactionResult:
    """Result of emit_end_if_ready — bundles events with compaction metadata.

    Eliminates the need for separate ``compaction_just_ended`` checks,
    preventing TOCTOU races between the emit call and the flag read.
    """

    events: list[StreamBaseResponse] = field(default_factory=list)
    just_ended: bool = False
    transcript_path: str = ""


# ---------------------------------------------------------------------------
# Event builders (private — use CompactionTracker or compaction_events)
# ---------------------------------------------------------------------------


def _start_events(tool_call_id: str) -> list[StreamBaseResponse]:
    """Build the opening events for a compaction tool call."""
    return [
        StreamStartStep(),
        StreamToolInputStart(toolCallId=tool_call_id, toolName=COMPACTION_TOOL_NAME),
        StreamToolInputAvailable(
            toolCallId=tool_call_id, toolName=COMPACTION_TOOL_NAME, input={}
        ),
    ]


def _end_events(tool_call_id: str, message: str) -> list[StreamBaseResponse]:
    """Build the closing events for a compaction tool call."""
    return [
        StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName=COMPACTION_TOOL_NAME,
            output=message,
        ),
        StreamFinishStep(),
    ]


def _new_tool_call_id() -> str:
    return f"compaction-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Public event builder
# ---------------------------------------------------------------------------


def emit_compaction(session: ChatSession) -> list[StreamBaseResponse]:
    """Create, persist, and return a self-contained compaction tool call.

    Convenience for callers that don't use ``CompactionTracker`` (e.g. the
    legacy non-SDK streaming path in ``service.py``).
    """
    tc_id = _new_tool_call_id()
    evts = compaction_events(COMPACTION_DONE_MSG, tool_call_id=tc_id)
    _persist(session, tc_id, COMPACTION_DONE_MSG)
    return evts


def compaction_events(
    message: str, tool_call_id: str | None = None
) -> list[StreamBaseResponse]:
    """Emit a self-contained compaction tool call (already completed).

    When *tool_call_id* is provided it is reused (e.g. for persistence that
    must match an already-streamed start event).  Otherwise a new ID is
    generated.
    """
    tc_id = tool_call_id or _new_tool_call_id()
    return _start_events(tc_id) + _end_events(tc_id, message)


# ---------------------------------------------------------------------------
# Message filtering
# ---------------------------------------------------------------------------


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
            real_calls: list[dict[str, Any]] = []
            for tc in msg.tool_calls:
                if tc.get("function", {}).get("name") == COMPACTION_TOOL_NAME:
                    compaction_ids.add(tc.get("id", ""))
                else:
                    real_calls.append(tc)
            if not real_calls and not msg.content:
                continue
        if msg.role == "tool" and msg.tool_call_id in compaction_ids:
            continue
        filtered.append(msg)
    return filtered


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist(session: ChatSession, tool_call_id: str, message: str) -> None:
    """Append compaction tool-call + result to session messages.

    Compaction events are synthetic so they bypass the normal adapter
    accumulation.  This explicitly records them so they survive a page refresh.
    """
    session.messages.append(
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": COMPACTION_TOOL_NAME,
                        "arguments": "{}",
                    },
                }
            ],
        )
    )
    session.messages.append(
        ChatMessage(role="tool", content=message, tool_call_id=tool_call_id)
    )


# ---------------------------------------------------------------------------
# CompactionTracker — state machine for streaming sessions
# ---------------------------------------------------------------------------


class CompactionTracker:
    """Tracks compaction state and yields UI events.

    Two compaction paths:

    1. **Pre-query** — history compressed before the SDK query starts.
       Call :meth:`emit_pre_query` to yield a self-contained tool call.

    2. **SDK-internal** — ``PreCompact`` hook fires mid-stream.
       Call :meth:`emit_start_if_ready` on heartbeat ticks and
       :meth:`emit_end_if_ready` when a message arrives.
    """

    def __init__(self) -> None:
        self._compact_start = asyncio.Event()
        self._start_emitted = False
        self._done = False
        self._tool_call_id = ""
        self._transcript_path: str = ""

    def on_compact(self, transcript_path: str = "") -> None:
        """Callback for the PreCompact hook. Stores transcript_path."""
        if (
            self._transcript_path
            and transcript_path
            and self._transcript_path != transcript_path
        ):
            logger.warning(
                "[Compaction] Overwriting transcript_path %s -> %s",
                self._transcript_path,
                transcript_path,
            )
        self._transcript_path = transcript_path
        self._compact_start.set()

    # ------------------------------------------------------------------
    # Pre-query compaction
    # ------------------------------------------------------------------

    def emit_pre_query(self, session: ChatSession) -> list[StreamBaseResponse]:
        """Emit + persist a self-contained compaction tool call."""
        self._done = True
        return emit_compaction(session)

    # ------------------------------------------------------------------
    # SDK-internal compaction
    # ------------------------------------------------------------------

    def reset_for_query(self) -> None:
        """Reset per-query state before a new SDK query."""
        self._compact_start.clear()
        self._done = False
        self._start_emitted = False
        self._tool_call_id = ""
        self._transcript_path = ""

    def emit_start_if_ready(self) -> list[StreamBaseResponse]:
        """If the PreCompact hook fired, emit start events (spinning tool)."""
        if self._compact_start.is_set() and not self._start_emitted and not self._done:
            self._compact_start.clear()
            self._start_emitted = True
            self._tool_call_id = _new_tool_call_id()
            return _start_events(self._tool_call_id)
        return []

    async def emit_end_if_ready(self, session: ChatSession) -> CompactionResult:
        """If compaction is in progress, emit end events and persist.

        Returns a ``CompactionResult`` with ``just_ended=True`` and the
        captured ``transcript_path`` when a compaction cycle completes.
        This avoids a separate flag check (TOCTOU-safe).
        """
        # Yield so pending hook tasks can set compact_start
        await asyncio.sleep(0)

        if self._done:
            return CompactionResult()
        if not self._start_emitted and not self._compact_start.is_set():
            return CompactionResult()

        if self._start_emitted:
            # Close the open spinner
            done_events = _end_events(self._tool_call_id, COMPACTION_DONE_MSG)
            persist_id = self._tool_call_id
        else:
            # PreCompact fired but start never emitted — self-contained
            persist_id = _new_tool_call_id()
            done_events = compaction_events(
                COMPACTION_DONE_MSG, tool_call_id=persist_id
            )

        transcript_path = self._transcript_path
        self._compact_start.clear()
        self._start_emitted = False
        self._done = True
        self._transcript_path = ""
        _persist(session, persist_id, COMPACTION_DONE_MSG)
        return CompactionResult(
            events=done_events, just_ended=True, transcript_path=transcript_path
        )
