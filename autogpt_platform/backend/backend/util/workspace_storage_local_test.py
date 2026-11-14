import asyncio
from pathlib import Path

import aiofiles.threadpool.binary
import pytest

from backend.util import workspace_storage_local
from backend.util.workspace_storage import LocalWorkspaceStorage


@pytest.mark.asyncio
async def test_failed_open_preserves_existing_transcript(tmp_path, mocker):
    storage = LocalWorkspaceStorage(str(tmp_path))
    path = await storage.store("sessions", "owner", "session.jsonl", b"previous turn")
    mocker.patch.object(
        workspace_storage_local.aiofiles, "open", side_effect=OSError(28, "disk full")
    )

    with pytest.raises(OSError, match="disk full"):
        await storage.store("sessions", "owner", "session.jsonl", b"next turn")

    assert storage._parse_storage_path(path).read_bytes() == b"previous turn"
    assert list(tmp_path.rglob("*.*")) == [storage._parse_storage_path(path)]


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_interrupted_write_preserves_completed_files(
    tmp_path, mocker, existing, cancelled
):
    storage = LocalWorkspaceStorage(str(tmp_path))
    if existing:
        await storage.store("sessions", "owner", "session.jsonl", b"previous turn")
    original_write = aiofiles.threadpool.binary.AsyncBufferedIOBase.write
    error = asyncio.CancelledError() if cancelled else OSError(28, "disk full")

    async def partial_write(handle, content):
        await original_write(handle, content[:3])
        raise error

    mocker.patch.object(
        aiofiles.threadpool.binary.AsyncBufferedIOBase, "write", partial_write
    )

    with pytest.raises(type(error)):
        await storage.store("sessions", "owner", "session.jsonl", b"next turn")

    path = tmp_path / "sessions" / "owner" / "session.jsonl"
    if existing:
        assert path.read_bytes() == b"previous turn"
        assert list(path.parent.iterdir()) == [path]
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_failed_store_stops_cleanup_at_relative_root(
    tmp_path, monkeypatch, mocker
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "guard.txt").write_text("keep test parent")
    storage = LocalWorkspaceStorage("outer/storage")
    mocker.patch.object(
        workspace_storage_local.aiofiles, "open", side_effect=OSError("failed")
    )

    with pytest.raises(OSError, match="failed"):
        await storage.store("ws", "file", "data.txt", b"content")

    assert (tmp_path / "outer" / "storage").is_dir()
    assert list((tmp_path / "outer" / "storage").iterdir()) == []


@pytest.mark.asyncio
async def test_successful_overwrite_replaces_complete_file(tmp_path):
    storage = LocalWorkspaceStorage(str(tmp_path))
    original_path = await storage.store("ws", "file", "data.txt", b"old")

    new_path = await storage.store("ws", "file", "data.txt", b"complete new content")

    assert new_path == original_path
    assert await storage.retrieve(new_path) == b"complete new content"
    assert list((tmp_path / "ws" / "file").iterdir()) == [
        storage._parse_storage_path(new_path)
    ]


@pytest.mark.asyncio
async def test_failed_replace_preserves_original_and_removes_temporary_file(
    tmp_path, mocker
):
    storage = LocalWorkspaceStorage(str(tmp_path))
    original_path = await storage.store("ws", "file", "data.txt", b"old")
    mocker.patch.object(Path, "replace", side_effect=OSError("replace failed"))

    with pytest.raises(OSError, match="replace failed"):
        await storage.store("ws", "file", "data.txt", b"new")

    assert await storage.retrieve(original_path) == b"old"
    assert list((tmp_path / "ws" / "file").iterdir()) == [
        storage._parse_storage_path(original_path)
    ]
