"""FILE_LIST's limits: the entry cap, the reply's byte budget and the scan budget."""

from __future__ import annotations

from pathlib import Path

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import FileHandler
from autogpt_local_executor.protocol import (
    FileListMessage,
    FileListPayload,
    FileListResponseMessage,
    dump_message,
    new_id,
    now_ts,
)


def _list(tmp_path: Path, **payload) -> FileListMessage:
    return FileListMessage(
        id=new_id(), ts=now_ts(), payload=FileListPayload(path=str(tmp_path), **payload)
    )


async def _listing(tmp_path: Path, **payload) -> FileListResponseMessage:
    handler = FileHandler(ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"))
    response = await handler.handle_list(_list(tmp_path, **payload))
    assert isinstance(response, FileListResponseMessage)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(("files", "truncated"), [(5, False), (6, True)])
async def test_truncated_means_a_match_was_left_out(
    tmp_path: Path, files: int, truncated: bool
) -> None:
    """Exactly max_entries matches used to come back marked truncated."""
    for index in range(files):
        (tmp_path / f"f{index}").write_text("")

    response = await _listing(tmp_path, max_entries=5)

    assert len(response.payload.entries) == 5
    assert response.payload.truncated is truncated


@pytest.mark.asyncio
async def test_a_listing_stops_at_its_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """10,000 entries with long paths could pass the message limit, and then
    the whole listing was replaced with FILE_TOO_LARGE."""
    for index in range(50):
        (tmp_path / f"{'long-name-' * 8}{index:02d}").write_text("")
    monkeypatch.setattr(handlers_module, "FILE_LIST_REPLY_BUDGET_BYTES", 4000)

    response = await _listing(tmp_path)

    assert 0 < len(response.payload.entries) < 50
    assert response.payload.truncated is True
    entry_bytes = sum(len(e.model_dump_json().encode()) + 1 for e in response.payload.entries)
    assert entry_bytes <= 4000
    assert len(dump_message(response).encode()) < 4000 + 1000


@pytest.mark.asyncio
async def test_a_walk_that_matches_nothing_stops_at_its_scan_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a glob that matches nothing, a recursive walk visited the whole
    tree, holding a request slot and a thread until it was done."""
    for folder in range(5):
        for index in range(10):
            (tmp_path / f"d{folder}").mkdir(exist_ok=True)
            (tmp_path / f"d{folder}" / f"f{index}").write_text("")
    monkeypatch.setattr(handlers_module, "MAX_FILE_LIST_SCANNED", 12)

    response = await _listing(tmp_path, recursive=True, glob="*.nothing")

    assert response.payload.entries == []
    assert response.payload.truncated is True
