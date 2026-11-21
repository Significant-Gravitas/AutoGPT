"""``WorkspaceManager.write_file`` reads the file at the path beside the quota,
with the behaviour of reading it after the quota: the same refusals in the
same order, and a failed read raised in that order too."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.workspace import WorkspaceFile
from backend.util.testing import Rendezvous
from backend.util.workspace import WorkspaceManager

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
_EXISTING = WorkspaceFile(
    id="existing-file-id",
    workspace_id="ws-123",
    name="test.txt",
    path="/test.txt",
    storage_path="ws-123/existing-uuid/test.txt",
    mime_type="text/plain",
    size_bytes=5,
    checksum="abc123",
    metadata={},
    created_at=_NOW,
    updated_at=_NOW,
)


@pytest.fixture
def manager() -> WorkspaceManager:
    return WorkspaceManager(user_id="user-123", workspace_id="ws-123")


@pytest.fixture
def db() -> MagicMock:
    db = MagicMock()
    db.create_workspace_file = AsyncMock(return_value=_EXISTING)
    db.get_workspace_file_by_path = AsyncMock(return_value=None)
    db.get_workspace_total_size = AsyncMock(return_value=0)
    return db


@contextmanager
def _workspace(db: MagicMock, limit: AsyncMock) -> Iterator[SimpleNamespace]:
    storage = AsyncMock()
    storage.store.return_value = "ws-123/new-uuid/test.txt"
    with (
        patch("backend.util.workspace.get_workspace_storage", return_value=storage),
        patch("backend.util.workspace.workspace_db", return_value=db),
        patch(
            "backend.util.workspace.scan_content_safe", new_callable=AsyncMock
        ) as scan,
        patch("backend.util.workspace.get_workspace_storage_limit_bytes", new=limit),
    ):
        yield SimpleNamespace(storage=storage, scan=scan)


@pytest.mark.parametrize("overwrite", [False, True], ids=["new", "overwrite"])
async def test_the_file_at_the_path_is_read_beside_the_quota(
    manager: WorkspaceManager, db: MagicMock, overwrite: bool
):
    meet = Rendezvous(3)
    db.get_workspace_total_size = AsyncMock(side_effect=meet.returning(0))
    db.get_workspace_file_by_path = AsyncMock(side_effect=meet.returning(None))
    with _workspace(db, AsyncMock(side_effect=meet.returning(1_000))):
        await manager.write_file(
            filename="test.txt", content=b"hello", overwrite=overwrite
        )

    db.get_workspace_file_by_path.assert_awaited_once_with("ws-123", "/test.txt")


async def test_a_renamed_write_looks_the_path_up_only_to_find_a_free_name(
    manager: WorkspaceManager, db: MagicMock
):
    with _workspace(db, AsyncMock(return_value=1_000)):
        await manager.write_file(
            filename="test.txt", content=b"hello", rename_on_conflict=True
        )

    db.get_workspace_file_by_path.assert_awaited_once_with("ws-123", "/test.txt")


async def test_an_existing_path_is_refused_before_the_quota_and_the_scan(
    manager: WorkspaceManager, db: MagicMock
):
    """The workspace is full too, but the duplicate path is checked first."""
    db.get_workspace_file_by_path.return_value = _EXISTING
    db.get_workspace_total_size.return_value = 1_000
    with (
        _workspace(db, AsyncMock(return_value=1_000)) as workspace,
        pytest.raises(ValueError, match="File already exists"),
    ):
        await manager.write_file(filename="test.txt", content=b"hello")

    workspace.scan.assert_not_called()
    workspace.storage.store.assert_not_called()
    db.create_workspace_file.assert_not_called()


async def test_a_failed_quota_read_is_raised_before_a_failed_path_read(
    manager: WorkspaceManager, db: MagicMock
):
    """The path read fails first, but it ran after the quota reads, so the
    quota read's error is the one the caller sees."""

    async def limit_fails_later(_user_id):
        for _ in range(3):
            await asyncio.sleep(0)
        raise RuntimeError("tier read failed")

    db.get_workspace_file_by_path = AsyncMock(
        side_effect=LookupError("path read failed")
    )
    with (
        _workspace(db, AsyncMock(side_effect=limit_fails_later)),
        pytest.raises(RuntimeError, match="tier read failed"),
    ):
        await manager.write_file(filename="test.txt", content=b"hello")
