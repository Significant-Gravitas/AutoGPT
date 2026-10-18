"""One structured summary line per SDK-path turn.

Every turn that enters the SDK path's main ``try`` logs exactly one line,
``[turn-summary] {...}``, from its ``finally`` (successful, failed and
cancelled turns alike) and attaches the same dict to the turn's Langfuse trace
as a ``copilot-turn-summary`` event. It records names, sizes, counts and hashes
only, never prompt or message content, so a measure can be read before and
after a change without regexing prompt text out of traces.

The collector is filled in as the turn runs. None of its methods raise: a
broken measurement must never break the turn it measures.
"""

import functools
import logging
from typing import Callable, ParamSpec

import sentry_sdk
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ToolUseBlock
from langfuse import get_client

from backend.copilot.model import ChatSession
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.sdk.compaction import CompactionTracker
from backend.copilot.sdk.query_blocks import scan_context_blocks
from backend.copilot.sdk.turn_summary_model import (
    TurnSummary,
    TurnUsage,
    outcome,
    query_path,
    record_options,
    record_usage,
)
from backend.copilot.session_kind import SessionKind
from backend.copilot.tree import TurnEnvelope

logger = logging.getLogger(__name__)

LANGFUSE_EVENT_NAME = "copilot-turn-summary"

_P = ParamSpec("_P")


def _never_raises(method: Callable[_P, None]) -> Callable[_P, None]:
    """Log and drop any error from a collector method instead of raising."""

    @functools.wraps(method)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        try:
            method(*args, **kwargs)
        except Exception:
            logger.warning("turn summary: %s failed", method.__name__, exc_info=True)

    return wrapper


class TurnSummaryCollector:
    """Accumulates one turn's summary and emits it once."""

    def __init__(
        self,
        session_id: str = "",
        *,
        voice: bool = False,
        is_user_message: bool | None = None,
    ) -> None:
        self.summary = TurnSummary(
            session_id=session_id or None,
            user_message=is_user_message,
            voice=voice,
        )
        self._call_ids: set[str] = set()
        self._launch_calls = 0
        self._tool_ids: set[str] = set()
        self._pending_history: tuple[int, bool] | None = None
        self._emitted = False

    @_never_raises
    def record_turn(
        self,
        *,
        turn: int,
        message_id: str,
        session: ChatSession,
        permissions: CopilotPermissions | None,
        envelope: TurnEnvelope | None,
        kind: SessionKind,
    ) -> None:
        """Identity and session-kind facts known when the turn starts."""
        meta = session.metadata
        s = self.summary
        s.turn = turn
        s.message_id = message_id
        s.first_turn = len(session.messages) <= 1
        s.origin = kind.origin
        s.session_kind = meta.kind
        s.builder_session = kind.builder
        s.expert_session = session.expert_id is not None
        s.expert_kickoff = kind.expert_kickoff
        s.sub_session = envelope is not None and envelope.depth > 0
        s.delegated = kind.delegated
        s.unattended = kind.unattended
        s.permission_filtered = permissions is not None
        s.dry_run = meta.dry_run

    @_never_raises
    def note_download(self, size: int | None) -> None:
        """Bytes of the stored CLI session downloaded for this turn."""
        self.summary.transcript_download_bytes = size

    @_never_raises
    def note_window_source(self, source: str | None) -> None:
        """Where the pinned window came from (``context_window.window_source``)."""
        self.summary.window_source = source

    @_never_raises
    def note_codex_usage(
        self,
        *,
        input_tokens: int | None,
        cached_input_tokens: int | None,
        boundary_peak_tokens: int | None,
    ) -> None:
        """The Codex gateway's own counts for the turn."""
        s = self.summary
        s.codex_input_tokens = input_tokens
        s.codex_cached_input_tokens = cached_input_tokens
        s.codex_boundary_peak_tokens = boundary_peak_tokens

    @_never_raises
    def note_after_compaction_resend(self) -> None:
        """One round of the first-turn context re-sent after a CLI compaction."""
        resends = (self.summary.after_compaction_resends or 0) + 1
        self.summary.after_compaction_resends = resends

    @_never_raises
    def note_history_rows(self, rows: int, *, compressed: bool) -> None:
        """Rows ``_build_query_message`` turned into a history block.

        Held until the next :meth:`record_query`, which reads the built query.
        """
        self._pending_history = (rows, compressed)

    @_never_raises
    def record_query(
        self,
        query: str,
        *,
        use_resume: bool,
        covered_rows: int,
        session_rows: int,
        rebuilt: bool = False,
    ) -> None:
        """The query as built for the next CLI launch: blocks, sizes and path."""
        s = self.summary
        blocks, prefix_chars = scan_context_blocks(query)
        names = [name for name, _ in blocks]
        largest = max(blocks, key=lambda block: block[1], default=None)
        s.context_blocks = names
        s.context_prefix_chars = prefix_chars
        s.context_largest_block = largest[0] if largest else None
        s.context_largest_block_chars = largest[1] if largest else 0
        s.query_chars = len(query)
        history = self._pending_history if "conversation_history" in names else None
        self._pending_history = None
        s.history_rows = history[0] if history else 0
        s.history_compressed = history[1] if history else None
        path = query_path(
            use_resume=use_resume,
            first_turn=bool(s.first_turn),
            history_sent=history is not None,
        )
        s.query_path = f"rebuilt_{path}" if rebuilt else path
        s.transcript_covered_rows = covered_rows
        s.session_rows = session_rows
        if rebuilt:
            s.context_rebuilds = (s.context_rebuilds or 0) + 1

    @_never_raises
    def note_cli_launch(self) -> None:
        """One CLI launch: the first attempt, a retry or a building-mode restart."""
        launches = (self.summary.cli_launches or 0) + 1
        self.summary.cli_launches = launches
        self.summary.retries = launches - 1
        self._launch_calls = 0

    @_never_raises
    def observe_assistant(self, message: AssistantMessage) -> None:
        """Count model calls, tool calls and main-thread calls that missed the cache.

        The CLI emits one ``AssistantMessage`` per content block, all carrying
        the same ``message_id`` and usage, so a call is counted once per id.
        A call after the first of its CLI launch that read nothing from the
        prompt cache counts towards ``calls_without_cache_read``.
        """
        s = self.summary
        self._tool_ids.update(
            block.id for block in message.content if isinstance(block, ToolUseBlock)
        )
        s.tool_calls = len(self._tool_ids)
        call_id = message.message_id or f"anonymous-{len(self._call_ids)}"
        if call_id in self._call_ids:
            return
        self._call_ids.add(call_id)
        if message.parent_tool_use_id is not None:
            s.subagent_model_calls = (s.subagent_model_calls or 0) + 1
            return
        s.model_calls = (s.model_calls or 0) + 1
        self._launch_calls += 1
        if self._launch_calls == 1 or message.usage is None:
            return
        if not message.usage.get("cache_read_input_tokens"):
            s.calls_without_cache_read = (s.calls_without_cache_read or 0) + 1

    @_never_raises
    def note_error(self, code: str) -> None:
        """A failure the turn reported to the client by code."""
        self.summary.error_class = code

    @_never_raises
    def note_upload(self, content: bytes | None) -> None:
        """The CLI session handed to ``upload_transcript``, or none on disk."""
        if content:
            self.summary.transcript_upload_bytes = len(content)
        else:
            self.summary.transcript_upload_skipped = "no_session_file"

    @_never_raises
    def note_upload_failed(self) -> None:
        self.summary.transcript_upload_skipped = "upload_error"

    @_never_raises
    def finish(
        self,
        *,
        error: BaseException | None,
        session: ChatSession,
        options: ClaudeAgentOptions | None,
        compaction: CompactionTracker | None,
        transport: str,
        model_requested: str | None,
        model_answered: str | None,
        e2b: bool,
        usage: TurnUsage,
        codex: bool,
        cost_reconcile: bool,
        upload_gate: str | None,
        ended_with_stream_error: bool,
    ) -> None:
        """Facts only known at the end of the turn.

        *upload_gate* is why the turn did not try to upload its CLI session
        (see :func:`~backend.copilot.sdk.turn_summary_model.upload_gate_reason`),
        or ``None`` when it did.
        """
        s = self.summary
        s.transport = transport
        s.model_requested = model_requested
        s.model_answered = model_answered
        s.release = sentry_sdk.get_client().options.get("release")
        s.e2b = e2b
        s.building_guide_in_prompt = session.guide_in_system_prompt
        record_usage(
            s,
            usage,
            model=model_answered or model_requested,
            codex=codex,
            cost_reconcile=cost_reconcile,
        )
        if options is not None:
            record_options(s, options)
        if compaction is not None:
            sources = compaction.attempt_sources
            s.cli_compactions = sources.count("sdk_internal")
            s.pre_query_compactions = sources.count("pre_query")
            # Every closed CLI cycle sends one ``copilot-compaction`` event; a
            # failed one is a cycle that left no readable summary.
            failed = compaction.failed_cycle_count
            closed = compaction.completed_sources.count("sdk_internal")
            s.cli_compactions_failed = failed
            s.cli_compactions_landed = closed - failed
            s.cli_compaction_failures = list(compaction.failed_cycle_sources) or None
        if upload_gate is not None:
            s.transcript_upload_skipped = upload_gate
        s.outcome, s.error_class = outcome(
            error, s.error_class, ended_with_stream_error
        )

    def emit(self, langfuse_trace_id: str | None = None) -> None:
        """Log the line and attach it to the trace, once. Never raises."""
        if self._emitted:
            return
        self._emitted = True
        try:
            logger.info(self.summary.to_line())
        except Exception:
            logger.warning("turn summary: emit failed", exc_info=True)
            return
        if not langfuse_trace_id:
            return
        try:
            get_client().create_event(
                trace_context={"trace_id": langfuse_trace_id},
                name=LANGFUSE_EVENT_NAME,
                metadata=self.summary.model_dump(exclude_none=True),
            )
        except Exception:
            logger.debug("turn summary: Langfuse event emit failed", exc_info=True)
