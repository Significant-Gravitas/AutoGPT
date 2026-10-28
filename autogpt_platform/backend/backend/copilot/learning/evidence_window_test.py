import json
from unittest.mock import AsyncMock

import pytest

from backend.copilot.db import PaginatedMessages
from backend.copilot.model import ChatMessage

from . import chat_source_test, nightly_test
from .chat_source import (
    ChatSessionSourceAdapter,
    record_chat_turn,
    source_revision_from_record,
)
from .chat_source_test import EXPERT, USER, _messages, _session, _session_info
from .contract import register_source_adapter
from .nightly_test import _completion, _proposal, _run

chat_db = chat_source_test.chat_db
boundaries = nightly_test.boundaries


def _serve(db, rows, *, unavailable=False, wrong_scope=False):
    async def page(
        *,
        session_id,
        limit,
        user_id,
        after_sequence,
        before_sequence,
        message_sequences=None,
    ):
        assert session_id == "session-1" and user_id == USER
        eligible = [m for m in rows if after_sequence <= m.sequence < before_sequence]
        if message_sequences is not None:
            assert len(message_sequences) <= 120
            if unavailable:
                return None
            eligible = [m for m in eligible if m.sequence in message_sequences]
        selected = eligible[-limit:]
        info = _session_info(
            expert_id="another-expert" if wrong_scope and message_sequences else EXPERT
        )
        return PaginatedMessages(
            messages=selected,
            has_more=len(eligible) > limit,
            oldest_sequence=selected[0].sequence if selected else None,
            session=info,
        )

    db.get_chat_messages_paginated = AsyncMock(side_effect=page)
    return db.get_chat_messages_paginated


async def _long_chat(fake_store, *, proof_size=0):
    old = _messages(3)
    old[1].content = None
    old[1].tool_calls = [
        {
            "id": "check",
            "type": "function",
            "function": {
                "name": "bash_exec",
                "arguments": '{"command":"python verify_csv.py"}',
            },
        }
    ]
    if proof_size:
        old[2].content = json.dumps(
            {"type": "block_output", "success": True, "result": "x" * proof_size}
        )
    await record_chat_turn(USER, _session(), old, "Validate the CSV")
    recent = [
        ChatMessage(role="user", content="Discuss the plan", sequence=i)
        for i in range(4, 145)
    ]
    source = await record_chat_turn(USER, _session(), recent, "Discuss the plan")
    return old + recent, source


@pytest.mark.asyncio
async def test_old_verification_and_its_tool_input_reach_the_reviewer(
    fake_store, chat_db, boundaries
):
    register_source_adapter(ChatSessionSourceAdapter())
    rows, source = await _long_chat(fake_store)
    for message in rows[4:]:
        message.content = "Discussion " + "x" * 2000
    calls = _serve(chat_db[0], rows)
    boundaries["review"].return_value = _completion(_proposal(supported_by=["msg:2"]))
    result = await _run()
    assert result.applied == 1 and result.model_calls == 1
    bundle = boundaries["review"].await_args.args[1]
    assert bundle.verification_complete
    assert "python verify_csv.py" in "\n".join(s.text for s in bundle.spans)
    assert len({s.ref for s in bundle.spans}) == len(bundle.spans)
    assert [int(s.ref.split(":")[1]) for s in bundle.spans] == sorted(
        int(s.ref.split(":")[1]) for s in bundle.spans
    )
    assert sum(len(s.text) for s in bundle.spans) <= 24_000
    assert calls.await_count == 2
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision


@pytest.mark.asyncio
async def test_all_forty_scattered_signals_are_fetched_in_one_bounded_query(
    fake_store, chat_db
):
    old = [
        ChatMessage(
            role="tool",
            content='{"type":"bash_exec","exit_code":0}',
            sequence=i * 10 + 2,
        )
        for i in range(40)
    ]
    await record_chat_turn(USER, _session(), old, "Check forty files")
    recent = [
        ChatMessage(role="user", content="Context", sequence=i)
        for i in range(1000, 1120)
    ]
    source = await record_chat_turn(USER, _session(), recent, "Context")
    future = ChatMessage(role="tool", content="future-result", sequence=1120)
    calls = _serve(chat_db[0], old + recent + [future])
    bundle = await ChatSessionSourceAdapter().load_evidence(
        source_revision_from_record(source), max_chars=24_000
    )
    assert bundle.verification_complete
    assert {f"msg:{i * 10 + 2}" for i in range(40)} <= {s.ref for s in bundle.spans}
    assert "msg:1120" not in {s.ref for s in bundle.spans}
    assert calls.await_count == 2
    targeted = calls.await_args_list[1].kwargs
    assert targeted["limit"] <= 120
    assert len(targeted["message_sequences"]) <= 120
    assert targeted["before_sequence"] == 1120


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "scope", "over-budget"])
async def test_targeted_evidence_still_requires_access_and_complete_proof(
    fake_store, chat_db, boundaries, failure
):
    register_source_adapter(ChatSessionSourceAdapter())
    rows, source = await _long_chat(
        fake_store, proof_size=25_000 if failure == "over-budget" else 0
    )
    _serve(
        chat_db[0],
        rows,
        unavailable=failure == "missing",
        wrong_scope=failure == "scope",
    )
    result = await _run()
    assert result.dispositions == {"inaccessible_evidence": 1}
    boundaries["review"].assert_not_awaited()
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision
