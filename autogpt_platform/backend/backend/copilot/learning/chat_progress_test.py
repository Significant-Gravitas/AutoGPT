import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.db import PaginatedMessages
from backend.copilot.model import ChatMessage

from . import chat_source
from .chat_source import (
    ChatSessionSourceAdapter,
    record_chat_turn,
    source_revision_from_record,
)
from .chat_source_test import USER, _messages, _session, _session_info


def _recent_messages():
    return [
        ChatMessage(role="user", content="Check the new CSV", sequence=140),
        ChatMessage(
            role="tool", content='{"type":"block_output","success":true}', sequence=141
        ),
        ChatMessage(role="assistant", content="The new CSV passed", sequence=142),
    ]


async def _processed_chat(fake_store, recent, *, large_old_result=False):
    old = _messages(3)
    if large_old_result:
        old[2].content = json.dumps(
            {"type": "block_output", "success": True, "detail": "x" * 30_000}
        )
    record = await record_chat_turn(USER, _session(), old, "import the csv")
    await fake_store.advance_source_cursor(USER, record.id, record.revision)
    current = await record_chat_turn(USER, _session(), recent, recent[0].content or "")
    return old, current


def _page(monkeypatch, rows):
    db = MagicMock()
    db.get_chat_session_metadata = AsyncMock(return_value=_session_info())
    db.get_chat_messages_paginated = AsyncMock(
        return_value=PaginatedMessages(
            messages=rows,
            has_more=False,
            oldest_sequence=rows[0].sequence,
            session=_session_info(),
        )
    )
    monkeypatch.setattr(chat_source, "chat_db", lambda: db)
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize("omission", ["page", "span", "budget"])
async def test_processed_signals_do_not_block_fresh_verification(
    fake_store, monkeypatch, omission
):
    recent = _recent_messages()
    old, current = await _processed_chat(
        fake_store, recent, large_old_result=omission == "span"
    )
    _page(monkeypatch, recent if omission == "page" else old + recent)
    budget = (
        sum(len(row.content or "") for row in recent) + 1
        if omission == "budget"
        else 24_000
    )
    source = source_revision_from_record(current)
    bundle = await ChatSessionSourceAdapter().load_evidence(source, max_chars=budget)
    assert bundle.verification_complete
    assert source.has_verifying_signal
    assert [signal.ref for signal in source.outcome_signals] == ["msg:141"]
    assert "msg:2" in bundle.omitted_refs


@pytest.mark.asyncio
async def test_processed_checks_do_not_qualify_a_turn_without_new_verification(
    fake_store,
):
    _, current = await _processed_chat(
        fake_store,
        [
            ChatMessage(role="user", content="What time is it?", sequence=4),
            ChatMessage(role="assistant", content="No execution yet", sequence=5),
        ],
    )
    assert not source_revision_from_record(current).has_verifying_signal


@pytest.mark.asyncio
async def test_new_clipped_verification_still_blocks_learning(fake_store, monkeypatch):
    recent = _recent_messages()
    recent[1].content = json.dumps(
        {"type": "block_output", "success": True, "detail": "x" * 2500}
    )
    _, current = await _processed_chat(fake_store, recent)
    _page(monkeypatch, recent)
    bundle = await ChatSessionSourceAdapter().load_evidence(
        source_revision_from_record(current), max_chars=2_000
    )
    assert "msg:141" in bundle.clipped_refs
    assert not bundle.verification_complete


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"source_kind": "application_task"},
        {"processed_revision": None},
        {"processed_revision": "non-message-revision"},
    ],
)
async def test_other_adapters_and_unknown_cursors_keep_their_signals(
    fake_store, updates
):
    _, current = await _processed_chat(fake_store, _recent_messages())
    source = source_revision_from_record(current.model_copy(update=updates))
    assert {signal.ref for signal in source.outcome_signals} == {"msg:2", "msg:141"}
