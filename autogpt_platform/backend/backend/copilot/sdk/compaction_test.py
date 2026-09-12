"""Tests for sdk/compaction.py — event builders, filtering, persistence, and
CompactionTracker state machine."""

import json as stdlib_json

import pytest

from backend.copilot.constants import (
    COMPACTION_DONE_MSG,
    COMPACTION_DROPPED_MSG,
    COMPACTION_TOOL_NAME,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import (
    StreamCompactionProgress,
    StreamFinishStep,
    StreamStartStep,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.sdk.compaction import (
    CompactionStats,
    CompactionTracker,
    build_compaction_output,
    compaction_events,
    emit_compaction,
    filter_compaction_messages,
    sdk_compaction_stats,
    transcript_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> ChatSession:
    return ChatSession.new(user_id="test-user", dry_run=False)


# ---------------------------------------------------------------------------
# compaction_events
# ---------------------------------------------------------------------------


class TestCompactionEvents:
    def test_returns_start_and_end_events(self):
        evts = compaction_events("done")
        assert len(evts) == 5
        assert isinstance(evts[0], StreamStartStep)
        assert isinstance(evts[1], StreamToolInputStart)
        assert isinstance(evts[2], StreamToolInputAvailable)
        assert isinstance(evts[3], StreamToolOutputAvailable)
        assert isinstance(evts[4], StreamFinishStep)

    def test_uses_provided_tool_call_id(self):
        evts = compaction_events("msg", tool_call_id="my-id")
        tool_start = evts[1]
        assert isinstance(tool_start, StreamToolInputStart)
        assert tool_start.toolCallId == "my-id"

    def test_generates_id_when_not_provided(self):
        evts = compaction_events("msg")
        tool_start = evts[1]
        assert isinstance(tool_start, StreamToolInputStart)
        assert tool_start.toolCallId.startswith("compaction-")

    def test_tool_name_is_context_compaction(self):
        evts = compaction_events("msg")
        tool_start = evts[1]
        assert isinstance(tool_start, StreamToolInputStart)
        assert tool_start.toolName == COMPACTION_TOOL_NAME


# ---------------------------------------------------------------------------
# emit_compaction
# ---------------------------------------------------------------------------


class TestEmitCompaction:
    def test_persists_to_session(self):
        session = _make_session()
        assert len(session.messages) == 0
        evts = emit_compaction(session)
        assert len(evts) == 5
        # Should have appended 2 messages (assistant tool call + tool result)
        assert len(session.messages) == 2
        assert session.messages[0].role == "assistant"
        assert session.messages[0].tool_calls is not None
        assert (
            session.messages[0].tool_calls[0]["function"]["name"]
            == COMPACTION_TOOL_NAME
        )
        assert session.messages[1].role == "tool"
        assert (
            stdlib_json.loads(session.messages[1].content or "")["summary"]
            == COMPACTION_DONE_MSG
        )


# ---------------------------------------------------------------------------
# filter_compaction_messages
# ---------------------------------------------------------------------------


class TestFilterCompactionMessages:
    def test_removes_compaction_tool_calls(self):
        msgs = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "comp-1",
                        "type": "function",
                        "function": {"name": COMPACTION_TOOL_NAME, "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(
                role="tool", content=COMPACTION_DONE_MSG, tool_call_id="comp-1"
            ),
            ChatMessage(role="assistant", content="world"),
        ]
        filtered = filter_compaction_messages(msgs)
        assert len(filtered) == 2
        assert filtered[0].content == "hello"
        assert filtered[1].content == "world"

    def test_keeps_non_compaction_tool_calls(self):
        msgs = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "real-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(role="tool", content="result", tool_call_id="real-1"),
        ]
        filtered = filter_compaction_messages(msgs)
        assert len(filtered) == 2

    def test_keeps_assistant_with_content_and_compaction_call(self):
        """If assistant message has both content and a compaction tool call,
        the message is kept (has real content)."""
        msgs = [
            ChatMessage(
                role="assistant",
                content="I have content",
                tool_calls=[
                    {
                        "id": "comp-1",
                        "type": "function",
                        "function": {"name": COMPACTION_TOOL_NAME, "arguments": "{}"},
                    }
                ],
            ),
        ]
        filtered = filter_compaction_messages(msgs)
        assert len(filtered) == 1

    def test_empty_list(self):
        assert filter_compaction_messages([]) == []


# ---------------------------------------------------------------------------
# CompactionTracker
# ---------------------------------------------------------------------------


class TestCompactionTracker:
    def test_on_compact_registers_pending_attempt(self):
        tracker = CompactionTracker()
        tracker.on_compact()
        assert tracker.attempt_count == 1
        assert list(tracker._pending_transcript_paths) == [""]

    def test_on_compact_sets_hook_fired(self):
        tracker = CompactionTracker()
        assert not tracker.hook_fired.is_set()
        tracker.on_compact()
        assert tracker.hook_fired.is_set()

    def test_pending_transcript_path_follows_the_cycle(self):
        tracker = CompactionTracker()
        assert tracker.pending_transcript_path is None
        assert tracker.has_pending_start is False
        tracker.on_compact("/tmp/session.jsonl")
        assert tracker.has_pending_start is True
        assert tracker.pending_transcript_path == "/tmp/session.jsonl"
        tracker.emit_start_if_ready()
        assert tracker.has_pending_start is False
        assert tracker.pending_transcript_path == "/tmp/session.jsonl"

    @pytest.mark.asyncio
    async def test_start_stats_pace_the_bar_and_end_stats_settle_the_row(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact("/tmp/session.jsonl")
        start = tracker.emit_start_if_ready(
            CompactionStats(tokens_before=128_000, messages_before=412)
        )
        assert isinstance(start[3], StreamCompactionProgress)
        assert start[3].tokensBefore == 128_000
        assert tracker.start_stats is not None
        assert tracker.start_stats.messages_before == 412

        result = await tracker.emit_end_if_ready(
            session,
            CompactionStats(
                tokens_before=128_000,
                tokens_after=31_000,
                messages_before=412,
                messages_after=38,
            ),
        )
        assert tracker.pending_transcript_path is None
        assert tracker.start_stats is None
        parsed = stdlib_json.loads(session.messages[1].content or "")
        assert parsed["tokensAfter"] == 31_000
        assert parsed["messagesBefore"] == 412
        rebuilding = result.events[-1]
        assert isinstance(rebuilding, StreamCompactionProgress)
        assert rebuilding.phase == "rebuilding"
        assert rebuilding.tokensAfter == 31_000

    def test_emit_start_if_ready_no_event(self):
        tracker = CompactionTracker()
        assert tracker.emit_start_if_ready() == []

    def test_emit_start_if_ready_with_event(self):
        tracker = CompactionTracker()
        tracker.on_compact()
        evts = tracker.emit_start_if_ready()
        assert len(evts) == 4
        assert isinstance(evts[0], StreamStartStep)
        assert isinstance(evts[1], StreamToolInputStart)
        assert isinstance(evts[2], StreamToolInputAvailable)
        assert isinstance(evts[3], StreamCompactionProgress)
        assert evts[3].phase == "summarizing"

    def test_emit_start_only_once(self):
        tracker = CompactionTracker()
        tracker.on_compact()
        evts1 = tracker.emit_start_if_ready()
        assert len(evts1) == 4
        # Second call should return empty
        evts2 = tracker.emit_start_if_ready()
        assert evts2 == []

    @pytest.mark.asyncio
    async def test_emit_end_after_start(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact()
        tracker.emit_start_if_ready()
        result = await tracker.emit_end_if_ready(session)
        assert result.just_ended is True
        assert len(result.events) == 3
        assert isinstance(result.events[0], StreamToolOutputAvailable)
        assert isinstance(result.events[1], StreamFinishStep)
        assert isinstance(result.events[2], StreamCompactionProgress)
        assert result.events[2].phase == "rebuilding"
        # Should persist
        assert len(session.messages) == 2

    @pytest.mark.asyncio
    async def test_emit_end_without_start_self_contained(self):
        """If PreCompact fired but start was never emitted, emit_end
        produces a self-contained compaction event."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact()
        # Don't call emit_start_if_ready
        result = await tracker.emit_end_if_ready(session)
        assert result.just_ended is True
        assert len(result.events) == 6  # Full self-contained event + rebuilding phase
        assert isinstance(result.events[0], StreamStartStep)
        assert isinstance(result.events[-1], StreamCompactionProgress)
        assert len(session.messages) == 2

    @pytest.mark.asyncio
    async def test_emit_end_no_op_when_no_new_compaction(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact()
        tracker.emit_start_if_ready()
        result1 = await tracker.emit_end_if_ready(session)
        assert result1.just_ended is True
        # Second call should be no-op (no new on_compact)
        result2 = await tracker.emit_end_if_ready(session)
        assert result2.just_ended is False
        assert result2.events == []

    @pytest.mark.asyncio
    async def test_emit_end_no_op_when_nothing_happened(self):
        tracker = CompactionTracker()
        session = _make_session()
        result = await tracker.emit_end_if_ready(session)
        assert result.just_ended is False
        assert result.events == []

    def test_emit_pre_query_start_then_end(self):
        tracker = CompactionTracker()
        session = _make_session()
        start_evts = tracker.emit_pre_query_start()
        end_evts = tracker.emit_pre_query_end(session, None)
        assert len(start_evts) == 4
        assert len(end_evts) == 3
        assert len(session.messages) == 2
        assert tracker.attempt_count == 1
        assert tracker.completed_count == 1
        assert tracker.get_observability_metadata() == {
            "compaction_attempt_count": 1,
            "compaction_attempt_sources": "pre_query",
            "compaction_count": 1,
            "compaction_sources": "pre_query",
        }

    def test_reset_for_query(self):
        tracker = CompactionTracker()
        tracker.on_compact("/some/path")
        tracker._start_emitted = True
        tracker._tool_call_id = "old"
        tracker._active_transcript_path = "/active/path"
        tracker.reset_for_query()
        assert tracker._start_emitted is False
        assert tracker._tool_call_id == ""
        assert tracker._active_transcript_path == ""
        assert list(tracker._pending_transcript_paths) == []

    def test_reset_for_query_forgets_pre_query_row(self):
        """A pre-query row id never crosses the query boundary."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        tracker.reset_for_query()
        assert tracker._pre_query_tool_call_id == ""
        assert tracker.abort_pre_query() == []
        end_evts = tracker.emit_pre_query_end(session, None)
        assert any(isinstance(e, StreamToolInputStart) for e in end_evts)

    @pytest.mark.asyncio
    async def test_pre_query_does_not_block_sdk_compaction_within_query(self):
        """SDK auto-compaction can still fire after a pre-query compaction."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        tracker.emit_pre_query_end(session, None)
        tracker.on_compact()
        evts = tracker.emit_start_if_ready()
        assert len(evts) == 4
        result = await tracker.emit_end_if_ready(session)
        assert result.just_ended is True
        assert tracker.completed_count == 2

    @pytest.mark.asyncio
    async def test_reset_allows_new_compaction(self):
        """After reset_for_query, compaction can fire again."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        tracker.emit_pre_query_end(session, None)
        tracker.reset_for_query()
        tracker.on_compact()
        evts = tracker.emit_start_if_ready()
        assert len(evts) == 4  # Start events emitted

    @pytest.mark.asyncio
    async def test_tool_call_id_consistency(self):
        """Start and end events use the same tool_call_id."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact()
        start_evts = tracker.emit_start_if_ready()
        result = await tracker.emit_end_if_ready(session)
        start_evt = start_evts[1]
        end_evt = result.events[0]
        assert isinstance(start_evt, StreamToolInputStart)
        assert isinstance(end_evt, StreamToolOutputAvailable)
        assert start_evt.toolCallId == end_evt.toolCallId
        # Persisted ID should also match
        tool_calls = session.messages[0].tool_calls
        assert tool_calls is not None
        assert tool_calls[0]["id"] == start_evt.toolCallId

    @pytest.mark.asyncio
    async def test_multiple_compactions_within_query(self):
        """Two mid-stream compactions within a single query both trigger."""
        tracker = CompactionTracker()
        session = _make_session()

        # First compaction cycle
        tracker.on_compact("/path/1")
        tracker.emit_start_if_ready()
        result1 = await tracker.emit_end_if_ready(session)
        assert result1.just_ended is True
        assert len(result1.events) == 3
        assert result1.transcript_path == "/path/1"

        # Second compaction cycle in the same query
        tracker.on_compact("/path/2")
        start_evts = tracker.emit_start_if_ready()
        assert len(start_evts) == 4
        result2 = await tracker.emit_end_if_ready(session)
        assert result2.just_ended is True
        assert result2.transcript_path == "/path/2"
        assert tracker.completed_count == 2

    @pytest.mark.asyncio
    async def test_multiple_compactions_with_intervening_message(self):
        """Multiple compactions remain supported across query boundaries."""
        tracker = CompactionTracker()
        session = _make_session()

        # First compaction
        tracker.on_compact("/path/1")
        tracker.emit_start_if_ready()
        result1 = await tracker.emit_end_if_ready(session)
        assert result1.just_ended is True
        assert result1.transcript_path == "/path/1"

        # Simulate reset between queries
        tracker.reset_for_query()

        # Second compaction in new query
        tracker.on_compact("/path/2")
        start_evts = tracker.emit_start_if_ready()
        assert len(start_evts) == 4
        result2 = await tracker.emit_end_if_ready(session)
        assert result2.just_ended is True
        assert result2.transcript_path == "/path/2"

    def test_on_compact_queues_transcript_path(self):
        tracker = CompactionTracker()
        tracker.on_compact("/some/path.jsonl")
        assert list(tracker._pending_transcript_paths) == ["/some/path.jsonl"]

    @pytest.mark.asyncio
    async def test_emit_end_returns_transcript_path(self):
        """CompactionResult includes the transcript_path from on_compact."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact("/my/session.jsonl")
        tracker.emit_start_if_ready()
        result = await tracker.emit_end_if_ready(session)
        assert result.just_ended is True
        assert result.transcript_path == "/my/session.jsonl"
        assert tracker._active_transcript_path == ""

    @pytest.mark.asyncio
    async def test_emit_end_clears_active_transcript_path(self):
        """After emit_end, the active transcript path is reset."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact("/first/path.jsonl")
        tracker.emit_start_if_ready()
        await tracker.emit_end_if_ready(session)
        assert tracker._active_transcript_path == ""

    @pytest.mark.asyncio
    async def test_multiple_pending_hooks_are_counted_even_before_completion(self):
        tracker = CompactionTracker()
        session = _make_session()

        tracker.on_compact("/path/1")
        tracker.emit_start_if_ready()
        tracker.on_compact("/path/2")
        tracker.on_compact("/path/3")

        result1 = await tracker.emit_end_if_ready(session)
        assert result1.just_ended is True
        assert result1.transcript_path == "/path/1"
        assert tracker.attempt_count == 3
        assert tracker.completed_count == 1

        tracker.emit_start_if_ready()
        result2 = await tracker.emit_end_if_ready(session)
        assert result2.just_ended is True
        assert result2.transcript_path == "/path/2"

        tracker.emit_start_if_ready()
        result3 = await tracker.emit_end_if_ready(session)
        assert result3.just_ended is True
        assert result3.transcript_path == "/path/3"
        assert tracker.completed_count == 3

    def test_get_observability_metadata_includes_attempts_and_completions(self):
        tracker = CompactionTracker()
        session = _make_session()

        tracker.emit_pre_query_start()
        tracker.emit_pre_query_end(session, None)
        tracker.on_compact("/path/1")
        tracker.on_compact("/path/2")

        assert tracker.get_observability_metadata() == {
            "compaction_attempt_count": 3,
            "compaction_attempt_sources": "pre_query,sdk_internal:2",
            "compaction_count": 1,
            "compaction_sources": "pre_query",
        }

    def test_get_log_summary_includes_attempts_and_completions(self):
        tracker = CompactionTracker()
        session = _make_session()

        tracker.emit_pre_query_start()
        tracker.emit_pre_query_end(session, None)
        tracker.on_compact("/path/1")
        tracker.on_compact("/path/2")

        assert tracker.get_log_summary() == {
            "attempt_count": 3,
            "attempt_sources": "pre_query,sdk_internal:2",
            "completed_count": 1,
            "completed_sources": "pre_query",
        }


# ---------------------------------------------------------------------------
# build_compaction_output
# ---------------------------------------------------------------------------


class TestBuildCompactionOutput:
    def test_encodes_stats_as_json_with_summary(self):
        out = build_compaction_output(
            CompactionStats(
                tokens_before=128_000,
                tokens_after=31_000,
                messages_before=412,
                messages_after=38,
            )
        )
        parsed = stdlib_json.loads(out)
        assert parsed["summary"] == COMPACTION_DONE_MSG
        assert parsed["tokensBefore"] == 128_000
        assert parsed["tokensAfter"] == 31_000
        assert parsed["messagesBefore"] == 412
        assert parsed["messagesAfter"] == 38

    def test_omits_unknown_fields(self):
        parsed = stdlib_json.loads(build_compaction_output(CompactionStats()))
        assert parsed == {"summary": COMPACTION_DONE_MSG}

    def test_none_stats_still_carries_summary(self):
        parsed = stdlib_json.loads(build_compaction_output(None))
        assert parsed == {"summary": COMPACTION_DONE_MSG}

    def test_dropped_history_is_reported_as_dropped(self):
        parsed = stdlib_json.loads(
            build_compaction_output(CompactionStats(dropped=True, messages_before=9))
        )
        assert parsed == {
            "summary": COMPACTION_DROPPED_MSG,
            "dropped": True,
            "messagesBefore": 9,
        }
        # A payload fact, not a wire stat: progress events never carry it.
        assert "dropped" not in CompactionStats(dropped=True).to_wire()


# ---------------------------------------------------------------------------
# Pre-query split emitters
# ---------------------------------------------------------------------------


class TestPreQueryEmitters:
    def test_start_emits_open_row_and_summarizing_phase(self):
        tracker = CompactionTracker()
        evts = tracker.emit_pre_query_start()
        assert isinstance(evts[0], StreamStartStep)
        assert isinstance(evts[1], StreamToolInputStart)
        assert isinstance(evts[2], StreamToolInputAvailable)
        assert isinstance(evts[3], StreamCompactionProgress)
        assert evts[3].phase == "summarizing"
        assert not any(isinstance(e, StreamToolOutputAvailable) for e in evts)

    def test_end_closes_the_same_tool_call_id(self):
        tracker = CompactionTracker()
        session = _make_session()
        start = tracker.emit_pre_query_start()
        tool_start = start[1]
        assert isinstance(tool_start, StreamToolInputStart)

        end = tracker.emit_pre_query_end(session, CompactionStats(tokens_before=9))
        output_evt = end[0]
        assert isinstance(output_evt, StreamToolOutputAvailable)
        assert output_evt.toolCallId == tool_start.toolCallId
        assert isinstance(end[1], StreamFinishStep)
        assert isinstance(end[2], StreamCompactionProgress)
        assert end[2].phase == "rebuilding"

    def test_end_persists_json_output_to_session(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        tracker.emit_pre_query_end(
            session, CompactionStats(tokens_before=100, tokens_after=10)
        )
        assert len(session.messages) == 2
        assert session.messages[1].role == "tool"
        parsed = stdlib_json.loads(session.messages[1].content or "")
        assert parsed["tokensBefore"] == 100
        assert parsed["tokensAfter"] == 10

    def test_dropped_end_closes_honestly_without_counting_a_compaction(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        evts = tracker.emit_pre_query_end(
            session, CompactionStats(dropped=True, messages_before=9)
        )
        assert isinstance(evts[0], StreamToolOutputAvailable)
        parsed = stdlib_json.loads(session.messages[1].content or "")
        assert parsed["dropped"] is True
        assert parsed["summary"] == COMPACTION_DROPPED_MSG
        assert tracker.attempt_sources == ("pre_query",)
        assert tracker.completed_sources == ()

    def test_end_without_start_is_self_contained(self):
        tracker = CompactionTracker()
        session = _make_session()
        evts = tracker.emit_pre_query_end(session, None)
        assert isinstance(evts[0], StreamStartStep)
        assert isinstance(evts[3], StreamToolOutputAvailable)
        assert isinstance(evts[4], StreamFinishStep)
        assert isinstance(evts[5], StreamCompactionProgress)
        assert len(session.messages) == 2

    def test_abort_closes_row_without_persisting(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        evts = tracker.abort_pre_query()
        assert isinstance(evts[0], StreamToolOutputAvailable)
        # The empty output IS the retirement sentinel — a client that sees a
        # summary here renders a compaction that never ran.
        assert evts[0].output == ""
        assert isinstance(evts[1], StreamFinishStep)
        assert session.messages == []

    def test_abort_claims_no_phase_of_its_own(self):
        """A retired prediction reports the retirement, not a stage.

        The empty-output sentinel is the whole signal: the client treats any
        phase left behind by a retired row as stale and stops animating.  A
        trailing phase event here would name a stage that never ran — and
        there is no honest value to send, since the row is being withdrawn
        rather than advanced.
        """
        tracker = CompactionTracker()
        tracker.emit_pre_query_start()
        evts = tracker.abort_pre_query()
        assert not [e for e in evts if isinstance(e, StreamCompactionProgress)]

    def test_abort_is_a_noop_when_no_row_is_open(self):
        tracker = CompactionTracker()
        assert tracker.abort_pre_query() == []

    def test_pre_query_counts_as_an_attempt_and_a_completion(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        # The counters record real work, not predictions — a row that is
        # opened optimistically and then aborted must leave no trace.
        assert tracker.attempt_sources == ()
        tracker.emit_pre_query_end(session, None)
        assert tracker.attempt_sources == ("pre_query",)
        assert tracker.completed_sources == ("pre_query",)

    def test_abort_records_neither_an_attempt_nor_a_completion(self):
        tracker = CompactionTracker()
        tracker.emit_pre_query_start()
        tracker.abort_pre_query()
        assert tracker.attempt_sources == ()
        assert tracker.completed_sources == ()


# ---------------------------------------------------------------------------
# SDK-internal rebuilding phase
# ---------------------------------------------------------------------------


class TestSdkInternalRebuildingPhase:
    @pytest.mark.asyncio
    async def test_end_if_ready_appends_rebuilding_phase(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.on_compact("/tmp/session.jsonl")
        tracker.emit_start_if_ready()
        result = await tracker.emit_end_if_ready(session)
        assert result.just_ended is True
        assert isinstance(result.events[-1], StreamCompactionProgress)
        assert result.events[-1].phase == "rebuilding"


# ---------------------------------------------------------------------------
# Pre-query ordering (row opens before the work, not after)
# ---------------------------------------------------------------------------


class TestPreQueryOrdering:
    def test_start_carries_no_output_event(self):
        """The opening events must leave the row OPEN.

        ``emit_pre_query_start`` runs before the compression work, so any
        output or finish event it carries would close the row while the work
        is still running — the user sees a completed row and then sits
        through the expensive part in silence.  Closing is
        ``emit_pre_query_end``'s job, or ``abort_pre_query``'s.
        """
        tracker = CompactionTracker()
        evts = tracker.emit_pre_query_start()
        assert not any(isinstance(e, StreamToolOutputAvailable) for e in evts)
        assert not any(isinstance(e, StreamFinishStep) for e in evts)

    def test_start_then_end_produces_exactly_one_persisted_row(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        tracker.emit_pre_query_end(session, CompactionStats(tokens_before=10))
        assert len(session.messages) == 2

    def test_aborted_prediction_persists_nothing(self):
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        tracker.abort_pre_query()
        assert session.messages == []

    def test_error_between_start_and_end_can_still_be_closed_by_abort(self):
        """Regression: an exception raised between start and end (e.g. the
        compression work itself failing) must not leave a permanently-open
        row on the client.

        Scope: this covers the tracker primitive only — that ``abort_pre_query``
        closes an open row and that a second call is a no-op.  That
        ``stream_chat_completion_sdk``'s except branch actually reaches for it
        is a separate claim, covered end to end by
        ``retry_scenarios_test.TestPreQueryCompactionRowTiming``.
        """
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query_start()
        try:
            raise RuntimeError("compression blew up")
        except RuntimeError:
            close_evts = tracker.abort_pre_query()
        assert len(close_evts) > 0
        assert session.messages == []
        # A second close (e.g. a defensive call on a later error path) is a
        # no-op rather than emitting a duplicate close.
        assert tracker.abort_pre_query() == []


# ---------------------------------------------------------------------------
# Sizing the SDK-internal path
# ---------------------------------------------------------------------------


class TestSdkCompactionStats:
    BEFORE = [
        {"type": "user", "message": {"role": "user", "content": "hello there"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool_use", "name": "find_block", "input": {"q": "email"}},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "3 blocks"}
                ],
            },
        },
        {"type": "progress", "message": {}},
    ]
    COMPACTED = [
        {
            "type": "summary",
            "isCompactSummary": True,
            "message": {"role": "user", "content": "Summary of the chat"},
        }
    ]

    def test_counts_turns_and_tokens_before(self):
        stats = transcript_stats(self.BEFORE, model="gpt-4o")
        assert stats.messages_before == 3
        assert stats.tokens_before is not None and stats.tokens_before > 0
        assert stats.tokens_after is None
        assert stats.messages_after is None

    def test_after_counts_come_from_the_compacted_file(self):
        stats = sdk_compaction_stats(self.BEFORE, self.COMPACTED, model="gpt-4o")
        assert stats.messages_before == 3
        assert stats.messages_after == 1
        assert stats.tokens_after is not None
        assert stats.tokens_after < (stats.tokens_before or 0)

    def test_unreadable_compacted_file_keeps_only_the_before_counts(self):
        stats = sdk_compaction_stats(self.BEFORE, None, model="gpt-4o")
        assert stats.messages_before == 3
        assert stats.messages_after is None

    def test_reuses_the_counts_measured_at_start(self):
        start = CompactionStats(tokens_before=999, messages_before=7)
        stats = sdk_compaction_stats([], self.COMPACTED, model="gpt-4o", start=start)
        assert stats.tokens_before == 999
        assert stats.messages_before == 7
        assert stats.messages_after == 1

    def test_empty_transcript_has_no_counts(self):
        assert transcript_stats([], model="gpt-4o") == CompactionStats()
