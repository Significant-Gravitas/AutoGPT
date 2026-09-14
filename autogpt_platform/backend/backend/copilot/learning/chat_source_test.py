"""Chat adapter: live eligibility and immutable, bounded evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.db import PaginatedMessages
from backend.copilot.model import ChatMessage, ChatSession, ChatSessionInfo

from . import chat_source
from ._fake_store import FakeLearningStore
from .chat_source import (
    CHAT_SOURCE_KIND,
    ChatSessionSourceAdapter,
    record_chat_turn,
    revision_sequence,
    scope_for,
    source_revision_from_record,
)
from .contract import EligibilityState

USER = "user-1"
EXPERT = "expert-1"
SESSION = "session-1"


def _session_info(
    *, user_id: str = USER, expert_id: str | None = EXPERT
) -> ChatSessionInfo:
    now = datetime.now(timezone.utc)
    return ChatSessionInfo(
        session_id=SESSION,
        user_id=user_id,
        usage=[],
        started_at=now,
        updated_at=now,
        expert_id=expert_id,
        title="CSV import",
    )


def _session(expert_id: str | None = EXPERT) -> ChatSession:
    return ChatSession(
        session_id=SESSION,
        user_id=USER,
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        expert_id=expert_id,
        messages=[],
    )


def _messages(upto: int) -> list[ChatMessage]:
    rows = [
        ChatMessage(role="user", content="import the csv", sequence=0),
        ChatMessage(role="assistant", content="Trying utf-8", sequence=1),
        ChatMessage(
            role="tool", content='{"type":"block_output","success":true}', sequence=2
        ),
        ChatMessage(role="assistant", content="Imported 42 rows", sequence=3),
        ChatMessage(role="user", content="Now delete everything", sequence=4),
        ChatMessage(
            role="tool", content='{"type":"bash_exec","exit_code":0}', sequence=5
        ),
    ]
    return [m for m in rows if m.sequence is not None and m.sequence <= upto]


@pytest.fixture
def chat_db(monkeypatch):
    db = MagicMock()
    db.get_chat_session_metadata = AsyncMock(return_value=_session_info())

    async def paginated(
        *, session_id, limit, user_id, after_sequence=None, before_sequence=None
    ):
        assert before_sequence is not None, "evidence reads must be bounded by revision"
        rows = [
            m
            for m in _messages(5)
            if m.sequence is not None and m.sequence < before_sequence
        ]
        return PaginatedMessages(
            messages=rows, has_more=False, oldest_sequence=0, session=_session_info()
        )

    db.get_chat_messages_paginated = AsyncMock(side_effect=paginated)
    monkeypatch.setattr(chat_source, "chat_db", lambda: db)
    experts = MagicMock()
    expert = MagicMock()
    expert.learning_paused_at = None
    experts.get_expert = AsyncMock(return_value=expert)
    monkeypatch.setattr(chat_source, "experts_db", lambda: experts)
    return db, experts, expert


@pytest.mark.asyncio
async def test_capture_records_a_padded_revision_and_signals(
    fake_store: FakeLearningStore,
):
    record = await record_chat_turn(USER, _session(), _messages(3), "import the csv")
    assert record.source_kind == CHAT_SOURCE_KIND and record.source_id == SESSION
    assert record.revision == "000000000003" and revision_sequence(record.revision) == 3
    assert [s["kind"] for s in record.outcome_signals] == ["tool_result"]
    assert record.origin == "ordinary"
    requested = await record_chat_turn(
        USER, _session(), _messages(3), "Great, save this as a skill."
    )
    assert requested.origin == "requested"


@pytest.mark.asyncio
async def test_evidence_is_bounded_to_the_reviewed_revision(fake_store, chat_db):
    """A new turn (sequences 4-5) landed after the revision under review;
    it must never appear in the bundle for revision 3."""
    record = await record_chat_turn(USER, _session(), _messages(3), "import the csv")
    source = source_revision_from_record(record)
    bundle = await ChatSessionSourceAdapter().load_evidence(source, max_chars=10_000)
    assert [span.ref for span in bundle.spans] == ["msg:0", "msg:1", "msg:2", "msg:3"]
    assert bundle.verification_complete
    assert bundle.omitted_refs == []
    tool_span = next(s for s in bundle.spans if s.ref == "msg:2")
    assert tool_span.outcome == "tool_result"


@pytest.mark.asyncio
async def test_clipped_verification_span_is_reported_not_claimed(fake_store, chat_db):
    record = await record_chat_turn(USER, _session(), _messages(3), "import the csv")
    source = source_revision_from_record(record)
    bundle = await ChatSessionSourceAdapter().load_evidence(source, max_chars=40)
    assert not bundle.verification_complete
    assert "msg:2" in bundle.omitted_refs


@pytest.mark.asyncio
async def test_revalidate_reports_each_ineligibility_reason(fake_store, chat_db):
    db, experts, expert = chat_db
    record = await record_chat_turn(USER, _session(), _messages(3), "import the csv")
    adapter = ChatSessionSourceAdapter()
    scope = scope_for(USER, EXPERT)
    ok = await adapter.revalidate(
        source_id=record.id, revision="3", scope=scope, approval_event_id=None
    )
    assert ok.state == EligibilityState.ELIGIBLE and ok.epoch == 0

    stale = await adapter.revalidate(
        source_id=record.id, revision="2", scope=scope, approval_event_id=None
    )
    assert stale.state == EligibilityState.STALE

    foreign = await adapter.revalidate(
        source_id=record.id,
        revision="3",
        scope=scope_for(USER, "expert-2"),
        approval_event_id=None,
    )
    assert foreign.state == EligibilityState.INACCESSIBLE

    expert.learning_paused_at = datetime.now(timezone.utc)
    paused = await adapter.revalidate(
        source_id=record.id, revision="3", scope=scope, approval_event_id=None
    )
    assert paused.state == EligibilityState.PAUSED
    expert.learning_paused_at = None

    experts.get_expert.return_value = None
    gone = await adapter.revalidate(
        source_id=record.id, revision="3", scope=scope, approval_event_id=None
    )
    assert gone.state == EligibilityState.INACCESSIBLE
    experts.get_expert.return_value = expert

    db.get_chat_session_metadata.return_value = _session_info(user_id="someone-else")
    lost = await adapter.revalidate(
        source_id=record.id, revision="3", scope=scope, approval_event_id=None
    )
    assert lost.state == EligibilityState.INACCESSIBLE

    db.get_chat_session_metadata.return_value = _session_info()
    await fake_store.set_source_eligibility(
        USER, record.id, "excluded", excluded_by_user_id=USER
    )
    excluded = await adapter.revalidate(
        source_id=record.id, revision="3", scope=scope, approval_event_id=None
    )
    assert excluded.state == EligibilityState.EXCLUDED and excluded.epoch == 1


@pytest.mark.asyncio
async def test_record_conversion_carries_approval_checkpoint(fake_store):
    record = await record_chat_turn(USER, _session(), _messages(3), "import the csv")
    assert source_revision_from_record(record).approval is None
    approved = await fake_store.set_source_approval(
        USER,
        record.id,
        approval_event_id="ev-9",
        approval_actor_id="reviewer",
        approved_revision="3",
        eligibility="eligible",
    )
    assert approved is not None
    converted = source_revision_from_record(approved)
    assert converted.approval is not None
    assert converted.approval.event_id == "ev-9"
    assert converted.approval.approved_revision == "000000000003"
    assert converted.epoch == 1
