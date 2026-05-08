"""Tests for sdk/compaction.py — event builders, filtering, persistence, and
CompactionTracker state machine."""

import pytest

from backend.copilot.constants import COMPACTION_DONE_MSG, COMPACTION_TOOL_NAME
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import (
    StreamFinishStep,
    StreamStartStep,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)
from backend.copilot.sdk.compaction import (
    CompactionTracker,
    compaction_events,
    emit_compaction,
    filter_compaction_messages,
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
        assert session.messages[1].content == COMPACTION_DONE_MSG


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

    def test_emit_start_if_ready_no_event(self):
        tracker = CompactionTracker()
        assert tracker.emit_start_if_ready() == []

    def test_emit_start_if_ready_with_event(self):
        tracker = CompactionTracker()
        tracker.on_compact()
        evts = tracker.emit_start_if_ready()
        assert len(evts) == 3
        assert isinstance(evts[0], StreamStartStep)
        assert isinstance(evts[1], StreamToolInputStart)
        assert isinstance(evts[2], StreamToolInputAvailable)

    def test_emit_start_only_once(self):
        tracker = CompactionTracker()
        tracker.on_compact()
        evts1 = tracker.emit_start_if_ready()
        assert len(evts1) == 3
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
        assert len(result.events) == 2
        assert isinstance(result.events[0], StreamToolOutputAvailable)
        assert isinstance(result.events[1], StreamFinishStep)
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
        assert len(result.events) == 5  # Full self-contained event
        assert isinstance(result.events[0], StreamStartStep)
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

    def test_emit_pre_query(self):
        tracker = CompactionTracker()
        session = _make_session()
        evts = tracker.emit_pre_query(session)
        assert len(evts) == 5
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

    @pytest.mark.asyncio
    async def test_pre_query_does_not_block_sdk_compaction_within_query(self):
        """SDK auto-compaction can still fire after a pre-query compaction."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query(session)
        tracker.on_compact()
        evts = tracker.emit_start_if_ready()
        assert len(evts) == 3
        result = await tracker.emit_end_if_ready(session)
        assert result.just_ended is True
        assert tracker.completed_count == 2

    @pytest.mark.asyncio
    async def test_reset_allows_new_compaction(self):
        """After reset_for_query, compaction can fire again."""
        tracker = CompactionTracker()
        session = _make_session()
        tracker.emit_pre_query(session)
        tracker.reset_for_query()
        tracker.on_compact()
        evts = tracker.emit_start_if_ready()
        assert len(evts) == 3  # Start events emitted

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
        assert len(result1.events) == 2
        assert result1.transcript_path == "/path/1"

        # Second compaction cycle in the same query
        tracker.on_compact("/path/2")
        start_evts = tracker.emit_start_if_ready()
        assert len(start_evts) == 3
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
        assert len(start_evts) == 3
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

        tracker.emit_pre_query(session)
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

        tracker.emit_pre_query(session)
        tracker.on_compact("/path/1")
        tracker.on_compact("/path/2")

        assert tracker.get_log_summary() == {
            "attempt_count": 3,
            "attempt_sources": "pre_query,sdk_internal:2",
            "completed_count": 1,
            "completed_sources": "pre_query",
        }
