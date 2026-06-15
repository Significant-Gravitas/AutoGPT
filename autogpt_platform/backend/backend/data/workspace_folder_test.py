from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.data import workspace_folder as wf
from backend.util.exceptions import NotFoundError


def _folder_record(**overrides):
    """A lightweight stand-in for a Prisma UserWorkspaceFolder record."""

    class _Rec:
        pass

    rec = _Rec()
    rec.id = overrides.get("id", "fld-1")
    rec.workspaceId = overrides.get("workspaceId", "ws-001")
    rec.name = overrides.get("name", "Reports")
    rec.icon = overrides.get("icon", None)
    rec.createdAt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec.updatedAt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec.Files = overrides.get("Files", [])
    return rec


@pytest.mark.asyncio
async def test_delete_folder_reparents_files_then_soft_deletes(mocker):
    """Files must be moved to root BEFORE the folder is soft-deleted, in one tx."""
    mocker.patch.object(
        wf,
        "_get_folder_record",
        mocker.AsyncMock(return_value=_folder_record()),
    )

    calls: list[str] = []

    file_prisma = mocker.MagicMock()
    file_prisma.update_many = mocker.AsyncMock(
        side_effect=lambda *a, **k: calls.append("reparent_files")
    )
    folder_prisma = mocker.MagicMock()
    folder_prisma.update_many = mocker.AsyncMock(
        side_effect=lambda *a, **k: calls.append("soft_delete_folder")
    )

    mocker.patch.object(
        wf.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=file_prisma)
    )
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )

    @asynccontextmanager
    async def _fake_tx(*args, **kwargs):
        yield None

    mocker.patch.object(wf, "transaction", _fake_tx)

    await wf.delete_folder("fld-1", "ws-001")

    assert calls == ["reparent_files", "soft_delete_folder"]
    # Files reparented to root (folderId=None) scoped to the workspace.
    _, kwargs = file_prisma.update_many.call_args
    assert kwargs["where"] == {"folderId": "fld-1", "workspaceId": "ws-001"}
    assert kwargs["data"] == {"folderId": None}
    # Folder soft-deleted (not hard-deleted) with a TOCTOU-safe isDeleted guard.
    _, kwargs = folder_prisma.update_many.call_args
    assert kwargs["where"] == {"id": "fld-1", "isDeleted": False}
    assert kwargs["data"] == {"isDeleted": True}


@pytest.mark.asyncio
async def test_create_folder_rejects_duplicate_root_name(mocker):
    """Duplicate root-level names are rejected in-app (NULL parentId defeats
    the DB unique constraint)."""
    folder_prisma = mocker.MagicMock()
    folder_prisma.find_first = mocker.AsyncMock(return_value=_folder_record())
    folder_prisma.create = mocker.AsyncMock()
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )

    with pytest.raises(FolderAlreadyExistsError):
        await wf.create_folder("ws-001", "Reports")

    folder_prisma.create.assert_not_awaited()
    _, kwargs = folder_prisma.find_first.call_args
    assert kwargs["where"] == {
        "workspaceId": "ws-001",
        "name": "Reports",
        "parentId": None,
        "isDeleted": False,
    }


@pytest.mark.asyncio
async def test_update_folder_rejects_duplicate_root_name(mocker):
    """Renaming onto an existing root-level name is rejected, excluding self."""
    mocker.patch.object(
        wf, "_get_folder_record", mocker.AsyncMock(return_value=_folder_record())
    )
    folder_prisma = mocker.MagicMock()
    folder_prisma.find_first = mocker.AsyncMock(return_value=_folder_record(id="fld-2"))
    folder_prisma.update = mocker.AsyncMock()
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )

    with pytest.raises(FolderAlreadyExistsError):
        await wf.update_folder("fld-1", "ws-001", name="Reports")

    folder_prisma.update.assert_not_awaited()
    _, kwargs = folder_prisma.find_first.call_args
    assert kwargs["where"]["id"] == {"not": "fld-1"}


@pytest.mark.asyncio
async def test_update_folder_raises_when_concurrently_deleted(mocker):
    """A folder soft-deleted between the ownership check and the write must
    surface NotFoundError, not a false 200 on a logically-deleted row."""
    mocker.patch.object(
        wf, "_get_folder_record", mocker.AsyncMock(return_value=_folder_record())
    )
    folder_prisma = mocker.MagicMock()
    folder_prisma.find_first = mocker.AsyncMock(return_value=None)  # no name dup
    folder_prisma.update_many = mocker.AsyncMock(return_value=0)  # nothing matched
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )

    with pytest.raises(NotFoundError):
        await wf.update_folder("fld-1", "ws-001", name="Renamed")

    _, kwargs = folder_prisma.update_many.call_args
    assert kwargs["where"] == {"id": "fld-1", "isDeleted": False}


@pytest.mark.asyncio
async def test_bulk_move_validates_target_folder_ownership(mocker):
    """A non-null target folder must be verified against the workspace."""
    guard = mocker.patch.object(
        wf, "_get_folder_record", mocker.AsyncMock(return_value=_folder_record())
    )
    file_prisma = mocker.MagicMock()
    file_prisma.update_many = mocker.AsyncMock()
    file_prisma.find_many = mocker.AsyncMock(return_value=[])
    mocker.patch.object(
        wf.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=file_prisma)
    )

    await wf.bulk_move_files_to_folder("ws-001", ["f1"], "fld-1")

    guard.assert_awaited_once_with("fld-1", "ws-001")
    _, kwargs = file_prisma.update_many.call_args
    assert kwargs["data"] == {"folderId": "fld-1"}
    assert kwargs["where"]["workspaceId"] == "ws-001"


@pytest.mark.asyncio
async def test_bulk_move_to_root_skips_folder_lookup(mocker):
    """Moving to root (folder_id=None) must not require a folder lookup."""
    guard = mocker.patch.object(wf, "_get_folder_record", mocker.AsyncMock())
    file_prisma = mocker.MagicMock()
    file_prisma.update_many = mocker.AsyncMock()
    file_prisma.find_many = mocker.AsyncMock(return_value=[])
    mocker.patch.object(
        wf.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=file_prisma)
    )

    await wf.bulk_move_files_to_folder("ws-001", ["f1"], None)

    guard.assert_not_awaited()
    _, kwargs = file_prisma.update_many.call_args
    assert kwargs["data"] == {"folderId": None}


@pytest.mark.asyncio
async def test_bulk_move_empty_file_ids_noop(mocker):
    """Empty file list returns immediately without touching the DB."""
    file_prisma = mocker.MagicMock()
    file_prisma.update_many = mocker.AsyncMock()
    mocker.patch.object(
        wf.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=file_prisma)
    )

    result = await wf.bulk_move_files_to_folder("ws-001", [], None)

    assert result == []
    file_prisma.update_many.assert_not_awaited()
