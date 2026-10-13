"""Filing a file into a folder holds the folder-hierarchy lock.

``delete_workspace_folder`` takes the same lock, so a folder cannot be deleted
between the check that it is live and the write that files a file under it.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import workspace as ws


def _record(calls: list[str], name: str, result: object) -> AsyncMock:
    async def call(*args, **kwargs):
        calls.append(name)
        return result

    return AsyncMock(side_effect=call)


@pytest.fixture
def calls(mocker) -> list[str]:
    calls: list[str] = []
    tx = MagicMock()
    tx.execute_raw = _record(calls, "lock", None)

    @asynccontextmanager
    async def fake_transaction():
        yield tx

    folders = MagicMock()
    folders.find_first = _record(calls, "check_folder", MagicMock())
    files = MagicMock()
    files.update_many = _record(calls, "write", 1)
    files.find_first = AsyncMock(return_value=None)
    files.create = _record(calls, "write", MagicMock())
    mocker.patch.object(ws, "transaction", fake_transaction)
    mocker.patch.object(ws.UserWorkspaceFolder, "prisma", return_value=folders)
    mocker.patch.object(ws.UserWorkspaceFile, "prisma", return_value=files)
    mocker.patch.object(ws.WorkspaceFile, "from_db", return_value=MagicMock())
    return calls


@pytest.mark.asyncio
async def test_a_move_into_a_folder_checks_it_under_the_lock(calls):
    await ws.update_workspace_file_location(
        "file-1", "ws-1", "a.txt", "/a.txt", folder_id="fld-1"
    )

    assert calls == ["lock", "check_folder", "write"]


@pytest.mark.asyncio
async def test_a_create_into_a_folder_checks_it_under_the_lock(calls):
    await ws.create_workspace_file(
        workspace_id="ws-1",
        file_id="file-1",
        name="a.txt",
        path="/a.txt",
        storage_path="local://ws-1/file-1/a.txt",
        mime_type="text/plain",
        size_bytes=1,
        folder_id="fld-1",
    )

    assert calls == ["lock", "check_folder", "write"]


@pytest.mark.asyncio
async def test_a_move_to_the_root_takes_no_lock(calls):
    await ws.update_workspace_file_location("file-1", "ws-1", "a.txt", "/a.txt")

    assert calls == ["write"]
