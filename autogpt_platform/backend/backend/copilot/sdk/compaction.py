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
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.util.prompt import estimate_token_count_str

from ..constants import (
    COMPACTION_DONE_MSG,
    COMPACTION_DROPPED_MSG,
    COMPACTION_TOOL_NAME,
)
from ..model import ChatMessage, ChatSession
from ..response_model import (
    CompactionPhase,
    StreamBaseResponse,
    StreamCompactionProgress,
    StreamFinishStep,
    StreamStartStep,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)


@dataclass
class CompactionResult:
    """Result of emit_end_if_ready — bundles events with compaction metadata.

    Eliminates the need for separate ``compaction_just_ended`` checks,
    preventing TOCTOU races between the emit call and the flag read.
    """

    events: list[StreamBaseResponse] = field(default_factory=list)
    just_ended: bool = False
    transcript_path: str = ""


class CompactionStats(BaseModel):
    """Token and message counts for one compaction cycle.

    Every field is optional: the SDK-internal path learns the counts from
    the CLI transcript (sometimes not at all), while the pre-query path
    reads them straight off ``CompressResult``.

    The camelCase serialization aliases are the wire names, declared once here
    so the tool row's JSON output and the ``data-compaction`` progress event
    cannot drift apart from each other or from the fields.
    """

    model_config = ConfigDict(frozen=True)

    tokens_before: int | None = Field(default=None, serialization_alias="tokensBefore")
    tokens_after: int | None = Field(default=None, serialization_alias="tokensAfter")
    messages_before: int | None = Field(
        default=None, serialization_alias="messagesBefore"
    )
    messages_after: int | None = Field(
        default=None, serialization_alias="messagesAfter"
    )
    # Compression failed outright and the history was dropped, not condensed.
    # A payload fact rather than a wire stat: the settled row reports the
    # reset, while the progress events (``to_wire``) never carry it.
    dropped: bool = Field(default=False, exclude=True)

    def to_wire(self) -> dict[str, Any]:
        """Known counts under their client-facing names; unknowns omitted."""
        return self.model_dump(by_alias=True, exclude_none=True)


def build_compaction_output(stats: "CompactionStats | None") -> str:
    """Encode the tool row's output as JSON, always carrying ``summary``.

    ``summary`` repeats ``COMPACTION_DONE_MSG`` verbatim so a client that
    cannot parse the JSON — or a session persisted before this change —
    still has a human-readable sentence to fall back on.  A dropped
    history gets ``COMPACTION_DROPPED_MSG`` and ``dropped: true`` instead,
    so neither a parsing client nor a legacy one can read a reset as a
    summary.
    """
    if stats is not None and stats.dropped:
        payload: dict[str, Any] = {"summary": COMPACTION_DROPPED_MSG, "dropped": True}
    else:
        payload = {"summary": COMPACTION_DONE_MSG}
    if stats is not None:
        payload.update(stats.to_wire())
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Sizing the SDK-internal path
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _is_message_entry(entry: dict) -> bool:
    return entry.get("type") in ("user", "assistant") or bool(
        entry.get("isCompactSummary")
    )


def _entry_text(entry: dict) -> str:
    """The text the model reads from one CLI transcript entry."""
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "thinking":
            parts.append(str(block.get("thinking") or ""))
        elif kind == "tool_use":
            parts.append(json.dumps(block.get("input") or {}, default=str))
        elif kind == "tool_result":
            result = block.get("content")
            parts.append(
                result if isinstance(result, str) else json.dumps(result, default=str)
            )
    return "\n".join(parts)


def _measure_transcript(
    entries: Iterable[dict], *, model: str
) -> tuple[int | None, int | None]:
    """``(tokens, turns)`` for CLI transcript entries; ``(None, None)`` if empty.

    Turns are user and assistant entries (tool results are user entries in
    the CLI format) plus the compaction summary.  Tokens are estimated over
    their text with the tokenizer ``compress_context`` measures the
    pre-query path with, so the two paths report comparable numbers.  The
    estimate is cosmetic: a tokenizer failure yields no count rather than
    an error.
    """
    rows = [e for e in entries if _is_message_entry(e)]
    if not rows:
        return None, None
    try:
        tokens = estimate_token_count_str(
            "\n".join(_entry_text(e) for e in rows), model=model
        )
    except Exception:
        logger.warning(
            "[SDK] Could not size %d transcript entries for the compaction row",
            len(rows),
            exc_info=True,
        )
        tokens = 0
    return (tokens or None), len(rows)


def transcript_stats(entries: Iterable[dict], *, model: str) -> CompactionStats:
    """Size the context the CLI is about to condense: the before-counts."""
    tokens, turns = _measure_transcript(entries, model=model)
    return CompactionStats(tokens_before=tokens, messages_before=turns)


def sdk_compaction_stats(
    before: Iterable[dict],
    compacted: list[dict] | None,
    *,
    model: str,
    start: "CompactionStats | None" = None,
) -> CompactionStats:
    """Before/after counts for one CLI-side compaction cycle.

    *before* is the transcript builder's mirror of the CLI context prior to
    compaction; *compacted* is what ``read_compacted_entries`` found in the
    session file afterwards — ``None`` when it could not be read, in which
    case only the before-counts are reported and the card falls back to
    its generic copy.  *start* reuses the counts measured when the row
    opened so a cycle is not tokenized twice.
    """
    stats = start if start is not None else transcript_stats(before, model=model)
    if compacted is None:
        return stats
    tokens, turns = _measure_transcript(compacted, model=model)
    return stats.model_copy(update={"tokens_after": tokens, "messages_after": turns})


def _progress(
    phase: CompactionPhase, stats: "CompactionStats | None" = None
) -> StreamCompactionProgress:
    stats = stats or CompactionStats()
    return StreamCompactionProgress(phase=phase, **stats.to_wire())


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


def _summarize_sources(sources: list[str]) -> str:
    counts = Counter(sources)
    parts: list[str] = []
    for source, count in counts.items():
        parts.append(f"{source}:{count}" if count > 1 else source)
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Public event builder
# ---------------------------------------------------------------------------


def emit_compaction(session: ChatSession) -> list[StreamBaseResponse]:
    """Create, persist, and return a self-contained compaction tool call.

    Convenience for callers that don't use ``CompactionTracker`` (e.g. the
    legacy non-SDK streaming path in ``service.py``).
    """
    tc_id = _new_tool_call_id()
    output = build_compaction_output(None)
    evts = compaction_events(output, tool_call_id=tc_id)
    _persist(session, tc_id, output)
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
       Call :meth:`emit_pre_query_start` to open the row before the work
       runs, then :meth:`emit_pre_query_end` (or :meth:`abort_pre_query`
       if the prediction was wrong) to close it.

    2. **SDK-internal** — ``PreCompact`` hook fires mid-stream.
       Call :meth:`emit_start_if_ready` on heartbeat ticks and
       :meth:`emit_end_if_ready` when a message arrives.  The hook also
       sets :attr:`hook_fired`; the SDK message loop waits on it so the
       row opens as soon as the hook lands instead of at the next
       heartbeat — otherwise a compaction that finishes inside one
       heartbeat interval collapses into a self-contained row and the
       live bar never shows.
    """

    def __init__(self) -> None:
        self.hook_fired = asyncio.Event()
        self._start_emitted = False
        self._start_stats: CompactionStats | None = None
        self._tool_call_id = ""
        self._active_transcript_path: str = ""
        self._pending_transcript_paths: deque[str] = deque()
        self._attempted_sources: list[str] = []
        self._completed_sources: list[str] = []
        self._pre_query_tool_call_id: str = ""

    @property
    def attempt_count(self) -> int:
        return len(self._attempted_sources)

    @property
    def attempt_sources(self) -> tuple[str, ...]:
        return tuple(self._attempted_sources)

    @property
    def completed_count(self) -> int:
        return len(self._completed_sources)

    @property
    def completed_sources(self) -> tuple[str, ...]:
        return tuple(self._completed_sources)

    def get_observability_metadata(self) -> dict[str, Any]:
        if not self._attempted_sources and not self._completed_sources:
            return {}

        metadata: dict[str, Any] = {
            "compaction_attempt_count": self.attempt_count,
            "compaction_attempt_sources": _summarize_sources(self._attempted_sources),
        }
        if self._completed_sources:
            metadata["compaction_count"] = self.completed_count
            metadata["compaction_sources"] = _summarize_sources(self._completed_sources)
        return metadata

    def get_log_summary(self) -> dict[str, Any]:
        return {
            "attempt_count": self.attempt_count,
            "attempt_sources": _summarize_sources(self._attempted_sources),
            "completed_count": self.completed_count,
            "completed_sources": _summarize_sources(self._completed_sources),
        }

    def on_compact(self, transcript_path: str = "") -> None:
        """Callback for the PreCompact hook. Queues an SDK compaction attempt."""
        self._attempted_sources.append("sdk_internal")
        self._pending_transcript_paths.append(transcript_path)
        self.hook_fired.set()

    # ------------------------------------------------------------------
    # Pre-query compaction
    # ------------------------------------------------------------------

    def emit_pre_query_start(
        self, tokens_before: int | None = None
    ) -> list[StreamBaseResponse]:
        """Open a compaction row BEFORE the compression work runs.

        The row stays open — no output event — until
        :meth:`emit_pre_query_end` or :meth:`abort_pre_query` closes it,
        so the progress bar spans the real work instead of appearing
        after it.

        *tokens_before* is the pre-check's token estimate for the slice about
        to be compressed.  It rides the ``summarizing`` phase so the client can
        pace its progress curve against the real size of the work — without it
        every compaction, 20K or 500K, animates on the same floor.  Optional
        because the SDK-internal path has no estimate to offer.

        Deliberately does NOT record an attempt: the caller opens this row
        on a *prediction* (``_will_compact``), and a prediction that turns
        out wrong is retired by :meth:`abort_pre_query`.  Counting it here
        would inflate ``compaction_attempt_count`` with compactions that
        never ran.  :meth:`emit_pre_query_end` records the attempt.
        """
        self._pre_query_tool_call_id = _new_tool_call_id()
        return [
            *_start_events(self._pre_query_tool_call_id),
            _progress("summarizing", CompactionStats(tokens_before=tokens_before)),
        ]

    def emit_pre_query_end(
        self, session: ChatSession, stats: "CompactionStats | None"
    ) -> list[StreamBaseResponse]:
        """Close the pre-query row and hand off to the rebuild phase."""
        output = build_compaction_output(stats)
        tc_id = self._pre_query_tool_call_id
        if tc_id:
            events: list[StreamBaseResponse] = list(_end_events(tc_id, output))
        else:
            # No open row (the pre-check missed) — emit a self-contained one.
            tc_id = _new_tool_call_id()
            events = list(_start_events(tc_id) + _end_events(tc_id, output))
        self._pre_query_tool_call_id = ""
        self._attempted_sources.append("pre_query")
        # A drop closes the row honestly but is not a compaction that
        # happened — it must not inflate ``compaction_count``.
        if stats is None or not stats.dropped:
            self._completed_sources.append("pre_query")
        _persist(session, tc_id, output)
        events.append(_progress("rebuilding", stats))
        return events

    def abort_pre_query(self) -> list[StreamBaseResponse]:
        """Close an optimistically-opened row when no compaction happened.

        The pre-check in ``stream_chat_completion_sdk`` predicts compaction
        from a token estimate; when the estimate is wrong the row must be
        retired without persisting anything, so a refresh doesn't replay a
        compaction that never occurred.

        The empty ``output`` is the sentinel that tells the client this row
        is retired rather than completed, and it is the only signal needed:
        the client treats phases left behind by a retired row as stale and
        stops animating.  Emitting a trailing phase here would instead claim
        a stage that never ran.
        """
        tc_id = self._pre_query_tool_call_id
        if not tc_id:
            return []
        self._pre_query_tool_call_id = ""
        return list(_end_events(tc_id, ""))

    # ------------------------------------------------------------------
    # SDK-internal compaction
    # ------------------------------------------------------------------

    def reset_for_query(self) -> None:
        """Reset per-query state before a new SDK query.

        The pre-query row cycle always closes before the query starts, so a
        pre-query id still set here is stale by definition.  Dropping it keeps
        a broken call order from attaching this query's compaction to a row
        the client no longer has open.
        """
        self._start_emitted = False
        self._start_stats = None
        self._tool_call_id = ""
        self._active_transcript_path = ""
        self._pending_transcript_paths.clear()
        self.hook_fired.clear()
        self._pre_query_tool_call_id = ""

    @property
    def has_pending_start(self) -> bool:
        """A PreCompact hook fired and its row has not been opened yet."""
        return bool(self._pending_transcript_paths) and not self._start_emitted

    @property
    def pending_transcript_path(self) -> str | None:
        """Session file of the compaction the next SDK message will close.

        ``None`` when no cycle is in flight.  The path itself may be empty
        when the hook carried none — the caller still closes the row, it
        just cannot read the compacted entries.
        """
        if self._start_emitted:
            return self._active_transcript_path
        if self._pending_transcript_paths:
            return self._pending_transcript_paths[0]
        return None

    @property
    def start_stats(self) -> "CompactionStats | None":
        """Counts measured when the open row was emitted, if any."""
        return self._start_stats

    def emit_start_if_ready(
        self, stats: "CompactionStats | None" = None
    ) -> list[StreamBaseResponse]:
        """If the PreCompact hook fired, emit start events (spinning tool).

        *stats* carries the before-counts measured off the transcript
        builder; ``tokens_before`` paces the client's curve and the
        counts are kept for the settled row (:attr:`start_stats`).
        """
        if self._pending_transcript_paths and not self._start_emitted:
            self._start_emitted = True
            self._start_stats = stats
            self._tool_call_id = _new_tool_call_id()
            self._active_transcript_path = self._pending_transcript_paths.popleft()
            return [*_start_events(self._tool_call_id), _progress("summarizing", stats)]
        return []

    async def emit_end_if_ready(
        self, session: ChatSession, stats: "CompactionStats | None" = None
    ) -> CompactionResult:
        """If compaction is in progress, emit end events and persist.

        Returns a ``CompactionResult`` with ``just_ended=True`` and the
        captured ``transcript_path`` when a compaction cycle completes.
        This avoids a separate flag check (TOCTOU-safe).

        *stats* is the measured before/after (see
        :func:`sdk_compaction_stats`); it lands in the persisted output and
        rides the ``rebuilding`` phase.
        """
        # Yield so pending hook tasks can set compact_start
        await asyncio.sleep(0)

        if not self._start_emitted and not self._pending_transcript_paths:
            return CompactionResult()

        output = build_compaction_output(stats)

        if self._start_emitted:
            # Close the open spinner
            done_events = _end_events(self._tool_call_id, output)
            persist_id = self._tool_call_id
            transcript_path = self._active_transcript_path
        else:
            # PreCompact fired but start never emitted — self-contained
            persist_id = _new_tool_call_id()
            done_events = compaction_events(output, tool_call_id=persist_id)
            transcript_path = (
                self._pending_transcript_paths.popleft()
                if self._pending_transcript_paths
                else ""
            )

        self._start_emitted = False
        self._start_stats = None
        self._tool_call_id = ""
        self._active_transcript_path = ""
        self._completed_sources.append("sdk_internal")
        _persist(session, persist_id, output)
        done_events.append(_progress("rebuilding", stats))
        return CompactionResult(
            events=done_events, just_ended=True, transcript_path=transcript_path
        )
