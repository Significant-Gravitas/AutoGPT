"""The file references in one tool call's arguments share one budget."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.sdk import file_ref
from backend.copilot.sdk.file_ref import FileRefExpansionError, expand_file_refs_in_args

_READ = "backend.copilot.sdk.file_ref.read_file_bytes"


@pytest.mark.asyncio
@pytest.mark.parametrize("embedded", [False, True], ids=["bare", "embedded"])
async def test_a_call_reads_no_more_references_than_the_limit(monkeypatch, embedded):
    monkeypatch.setattr(file_ref, "_MAX_REFS_PER_CALL", 3, raising=False)
    refs = [f"@@agptfile:/tmp/f{i}.txt" for i in range(4)]
    args = {"text": " ".join(refs)} if embedded else {"files": refs}
    read = AsyncMock(return_value=b"x")

    with (
        patch(_READ, read),
        pytest.raises(FileRefExpansionError, match="Too many file references"),
    ):
        await expand_file_refs_in_args(args, user_id="u1", session=_session())

    assert read.await_count == 3


@pytest.mark.asyncio
async def test_text_arguments_share_one_expansion_budget(monkeypatch):
    monkeypatch.setattr(file_ref, "_MAX_TOTAL_EXPAND_CHARS", 10)
    read = AsyncMock(return_value=b"0123456789")

    with patch(_READ, read):
        result = await expand_file_refs_in_args(
            {"a": "a: @@agptfile:/tmp/a.txt", "b": "b: @@agptfile:/tmp/b.txt"},
            user_id="u1",
            session=_session(),
        )

    assert result["a"] == "a: 0123456789"
    assert result["b"] == (
        "b: [file-ref budget exhausted: total expansion limit reached]"
    )
    read.assert_awaited_once()


@pytest.mark.asyncio
async def test_bare_references_share_one_size_budget(monkeypatch):
    monkeypatch.setattr(file_ref, "_MAX_TOTAL_BARE_REF_SIZE", 15, raising=False)
    schema = {"properties": {"files": {"type": "array", "items": {"type": "string"}}}}

    with (
        patch(_READ, AsyncMock(return_value=b"0123456789")),
        pytest.raises(FileRefExpansionError, match="too large in total"),
    ):
        await expand_file_refs_in_args(
            {"files": ["@@agptfile:/tmp/a.txt", "@@agptfile:/tmp/b.txt"]},
            user_id="u1",
            session=_session(),
            input_schema=schema,
        )


def _session() -> MagicMock:
    session = MagicMock()
    session.session_id = "sess-1"
    return session
