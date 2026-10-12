import json

import pytest

from . import chat_source_test, nightly_test
from .chat_progress_test import _page
from .chat_source import (
    ChatSessionSourceAdapter,
    record_chat_turn,
    source_revision_from_record,
)
from .chat_source_test import USER, _messages, _session
from .contract import register_source_adapter
from .nightly_test import _completion, _proposal, _run

chat_db = chat_source_test.chat_db
boundaries = nightly_test.boundaries


def _large_result(size: int):
    rows = _messages(3)
    rows[2].content = json.dumps(
        {"type": "block_output", "success": True, "output": "x" * size}
    )
    rows[3].content = "Long follow-up context. " * 200
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [2100, 9000, 23000])
async def test_verification_uses_full_text_within_the_total_budget(
    fake_store, monkeypatch, size
):
    rows = _large_result(size)
    record = await record_chat_turn(USER, _session(), rows, "import the csv")
    _page(monkeypatch, rows)
    bundle = await ChatSessionSourceAdapter().load_evidence(
        source_revision_from_record(record), max_chars=24_000
    )
    assert bundle.verification_complete
    proof = next(span for span in bundle.spans if span.ref == "msg:2")
    assert proof.text == rows[2].content
    assert "msg:2" not in bundle.clipped_refs
    assert sum(len(span.text) for span in bundle.spans) <= 24_000
    assert [span.ref for span in bundle.spans] == sorted(
        span.ref for span in bundle.spans
    )


@pytest.mark.asyncio
async def test_nightly_reviews_a_large_complete_verification(
    fake_store, monkeypatch, chat_db, boundaries
):
    register_source_adapter(ChatSessionSourceAdapter())
    rows = _large_result(9000)
    source = await record_chat_turn(USER, _session(), rows, "import the csv")
    _page(monkeypatch, rows)
    boundaries["review"].return_value = _completion(_proposal(supported_by=["msg:2"]))
    result = await _run()
    assert result.applied == 1
    boundaries["review"].assert_awaited_once()
    bundle = boundaries["review"].await_args.args[1]
    assert (
        next(span.text for span in bundle.spans if span.ref == "msg:2")
        == rows[2].content
    )
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision


@pytest.mark.asyncio
async def test_verification_larger_than_the_total_budget_is_still_incomplete(
    fake_store, monkeypatch
):
    rows = _large_result(25000)
    source = await record_chat_turn(USER, _session(), rows, "import the csv")
    _page(monkeypatch, rows)
    bundle = await ChatSessionSourceAdapter().load_evidence(
        source_revision_from_record(source), max_chars=24_000
    )
    assert not bundle.verification_complete
    assert "msg:2" in bundle.clipped_refs
    assert sum(len(span.text) for span in bundle.spans) <= 24_000
