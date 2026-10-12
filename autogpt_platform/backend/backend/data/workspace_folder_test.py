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
    rec.parentId = overrides.get("parentId", None)
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
    # The subtree walk reads the workspace's folders; this one has no children.
    folder_prisma.find_many = mocker.AsyncMock(return_value=[_folder_record()])

    mocker.patch.object(
        wf.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=file_prisma)
    )
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )

    @asynccontextmanager
    async def _fake_tx(*args, **kwargs):
        # The hierarchy lock runs on the transaction, so it cannot be None.
        tx = mocker.MagicMock()
        tx.execute_raw = mocker.AsyncMock()
        yield tx

    mocker.patch.object(wf, "transaction", _fake_tx)

    await wf.delete_folder("fld-1", "ws-001")

    assert calls == ["reparent_files", "soft_delete_folder"]
    # Files reparented to root (folderId=None) scoped to the workspace.
    _, kwargs = file_prisma.update_many.call_args
    assert kwargs["where"] == {"folderId": {"in": ["fld-1"]}, "workspaceId": "ws-001"}
    assert kwargs["data"] == {"folderId": None}
    # Folder soft-deleted (not hard-deleted) with a TOCTOU-safe isDeleted guard.
    _, kwargs = folder_prisma.update_many.call_args
    assert kwargs["where"] == {"id": {"in": ["fld-1"]}, "isDeleted": False}
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

    @asynccontextmanager
    async def _fake_tx(*args, **kwargs):
        # The hierarchy lock runs on the transaction, so it cannot be None.
        tx = mocker.MagicMock()
        tx.execute_raw = mocker.AsyncMock()
        yield tx

    mocker.patch.object(wf, "transaction", _fake_tx)

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

    @asynccontextmanager
    async def _fake_tx(*args, **kwargs):
        # The hierarchy lock runs on the transaction, so it cannot be None.
        tx = mocker.MagicMock()
        tx.execute_raw = mocker.AsyncMock()
        yield tx

    mocker.patch.object(wf, "transaction", _fake_tx)

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


def _tree(*records):
    """A prisma mock whose ``find_many`` answers with *records*."""
    return records


@pytest.mark.asyncio
async def test_move_folder_refuses_a_move_into_its_own_subtree(mocker):
    """The guard walks the destination's ancestors; without it the subtree is
    detached from the root and nothing can list it again."""
    parent = _folder_record(id="parent", name="Parent")
    child = _folder_record(id="child", name="Child", parentId="parent")
    grandchild = _folder_record(id="grandchild", name="Grandchild", parentId="child")

    folder_prisma = mocker.MagicMock()
    folder_prisma.find_many = mocker.AsyncMock(
        return_value=list(_tree(parent, child, grandchild))
    )
    # Everything past the guard is mocked, so removing the guard makes this
    # test fail on the missing raise rather than on an unmocked call.
    folder_prisma.find_first = mocker.AsyncMock(return_value=None)
    folder_prisma.update_many = mocker.AsyncMock(return_value=1)
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )
    mocker.patch.object(
        wf, "get_folder", mocker.AsyncMock(return_value=mocker.sentinel.moved)
    )
    records = {r.id: r for r in (parent, child, grandchild)}
    mocker.patch.object(
        wf,
        "_get_folder_record",
        mocker.AsyncMock(side_effect=lambda fid, ws: records[fid]),
    )

    for destination in ("grandchild", "child", "parent"):
        with pytest.raises(wf.FolderValidationError):
            await wf.move_folder("parent", "ws-001", destination)
    folder_prisma.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_move_folder_to_root_clears_the_parent(mocker):
    child = _folder_record(id="child", name="Child", parentId="parent")

    folder_prisma = mocker.MagicMock()
    folder_prisma.find_first = mocker.AsyncMock(return_value=None)
    folder_prisma.update_many = mocker.AsyncMock(return_value=1)
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )
    mocker.patch.object(wf, "_get_folder_record", mocker.AsyncMock(return_value=child))
    mocker.patch.object(
        wf, "get_folder", mocker.AsyncMock(return_value=mocker.sentinel.moved)
    )

    assert await wf.move_folder("child", "ws-001", None) is mocker.sentinel.moved
    _, kwargs = folder_prisma.update_many.call_args
    assert kwargs["where"] == {"id": "child", "isDeleted": False}
    assert kwargs["data"] == {"parentId": None}


@pytest.mark.asyncio
async def test_delete_folder_takes_the_whole_subtree(mocker):
    """A grandchild left behind would keep its files hidden under a folder no
    listing returns."""
    parent = _folder_record(id="parent", name="Parent")
    child = _folder_record(id="child", name="Child", parentId="parent")
    grandchild = _folder_record(id="grandchild", name="Grandchild", parentId="child")
    sibling = _folder_record(id="sibling", name="Sibling")

    file_prisma = mocker.MagicMock()
    file_prisma.update_many = mocker.AsyncMock()
    folder_prisma = mocker.MagicMock()
    folder_prisma.update_many = mocker.AsyncMock()
    folder_prisma.find_many = mocker.AsyncMock(
        return_value=[parent, child, grandchild, sibling]
    )
    mocker.patch.object(
        wf.UserWorkspaceFile, "prisma", mocker.MagicMock(return_value=file_prisma)
    )
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )
    mocker.patch.object(wf, "_get_folder_record", mocker.AsyncMock(return_value=parent))

    @asynccontextmanager
    async def _fake_tx(*args, **kwargs):
        # The hierarchy lock runs on the transaction, so it cannot be None.
        tx = mocker.MagicMock()
        tx.execute_raw = mocker.AsyncMock()
        yield tx

    mocker.patch.object(wf, "transaction", _fake_tx)

    await wf.delete_folder("parent", "ws-001")

    _, folder_kwargs = folder_prisma.update_many.call_args
    assert sorted(folder_kwargs["where"]["id"]["in"]) == [
        "child",
        "grandchild",
        "parent",
    ]
    _, file_kwargs = file_prisma.update_many.call_args
    assert sorted(file_kwargs["where"]["folderId"]["in"]) == [
        "child",
        "grandchild",
        "parent",
    ]
    assert file_kwargs["data"] == {"folderId": None}


@pytest.mark.asyncio
async def test_create_folder_scopes_the_name_clash_to_the_parent(mocker):
    folder_prisma = mocker.MagicMock()
    folder_prisma.find_first = mocker.AsyncMock(return_value=None)
    folder_prisma.create = mocker.AsyncMock(
        return_value=_folder_record(id="new", parentId="parent")
    )
    mocker.patch.object(
        wf.UserWorkspaceFolder, "prisma", mocker.MagicMock(return_value=folder_prisma)
    )
    mocker.patch.object(
        wf,
        "_get_folder_record",
        mocker.AsyncMock(return_value=_folder_record(id="parent")),
    )

    created = await wf.create_folder("ws-001", "Reports", parent_id="parent")

    assert created.parent_id == "parent"
    _, kwargs = folder_prisma.find_first.call_args
    assert kwargs["where"]["parentId"] == "parent"
    _, kwargs = folder_prisma.create.call_args
    assert kwargs["data"]["Parent"] == {"connect": {"id": "parent"}}
