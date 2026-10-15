import json

import pytest

from backend.copilot.model import ChatMessage

from . import chat_source_test, nightly_test
from .chat_evidence import MAX_CONTEXT_SPAN_CHARS
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


def _fixture_chat(request: str, code: str, summary: str) -> list[ChatMessage]:
    """The shape of a real coding turn: a long request with the rules and
    fixtures, the code that ran, a passing check, and a closing summary."""
    return [
        ChatMessage(role="user", content=request, sequence=0),
        ChatMessage(role="assistant", content="Writing the solver.", sequence=1),
        ChatMessage(role="assistant", content=code, sequence=2),
        ChatMessage(role="assistant", content="Running all fixtures.", sequence=3),
        ChatMessage(
            role="tool",
            content=json.dumps(
                {"type": "bash_exec", "exit_code": 0, "stdout": "12/12 passed"}
            ),
            sequence=4,
        ),
        ChatMessage(role="assistant", content=summary, sequence=5),
    ]


@pytest.mark.asyncio
async def test_unused_budget_keeps_the_request_and_the_code_whole(
    fake_store, monkeypatch
):
    rows = _fixture_chat(
        "Validate the supplier file against these fixtures. " * 120,
        "assert solve(case) == expected\n" * 180,
        "All fixtures pass. " * 200,
    )
    record = await record_chat_turn(USER, _session(), rows, rows[0].content or "")
    _page(monkeypatch, rows)
    bundle = await ChatSessionSourceAdapter().load_evidence(
        source_revision_from_record(record), max_chars=24_000
    )
    texts = {span.ref: span.text for span in bundle.spans}
    assert all(
        texts[f"msg:{row.sequence}"] == (row.content or "").strip() for row in rows
    )
    assert bundle.clipped_refs == []
    assert bundle.verification_complete


@pytest.mark.asyncio
async def test_spare_budget_restores_the_request_before_later_narration(
    fake_store, monkeypatch
):
    request = ("Rules and fixtures for the import. " * 150).strip()
    rows = _fixture_chat(request, "code line\n" * 500, "Summary of the run. " * 250)
    first_pass = sum(
        min(len((row.content or "").strip()), MAX_CONTEXT_SPAN_CHARS) for row in rows
    )
    budget = first_pass + len(request) - MAX_CONTEXT_SPAN_CHARS
    record = await record_chat_turn(USER, _session(), rows, request)
    _page(monkeypatch, rows)
    bundle = await ChatSessionSourceAdapter().load_evidence(
        source_revision_from_record(record), max_chars=budget
    )
    texts = {span.ref: span.text for span in bundle.spans}
    assert texts["msg:0"] == request
    assert "msg:0" not in bundle.clipped_refs
    assert {"msg:2", "msg:5"} <= set(bundle.clipped_refs)
    assert sum(len(span.text) for span in bundle.spans) <= budget
    assert bundle.verification_complete
