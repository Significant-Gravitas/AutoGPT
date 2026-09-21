"""Real-database coverage for nested workspace folders.

Folders are a DB-level layer over files: a file's ``folderId`` and a folder's
``parentId`` decide the tree, and no storage path moves. These tests pin the
three things a tree adds over a flat list — creating a child, deleting a
subtree, and moving one without detaching it from the root.
"""

import uuid

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import UserWorkspaceFile, UserWorkspaceFolder

from backend.api.features.library.exceptions import (
    FolderAlreadyExistsError,
    FolderValidationError,
)
from backend.data.user import get_or_create_user
from backend.data.workspace import create_workspace_file, get_or_create_workspace
from backend.data.workspace_folder import (
    bulk_move_files_to_folder,
    create_folder,
    delete_folder,
    list_workspace_folders,
    move_folder,
)
from backend.util.test import SpinTestServer


async def _workspace() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"folders-{suffix}@example.com",
            "name": "Folder Owner",
        }
    )
    workspace = await get_or_create_workspace(user.id)
    return workspace.id


async def _file(workspace_id: str, folder_id: str | None = None) -> str:
    stored = await create_workspace_file(
        workspace_id=workspace_id,
        file_id=str(uuid.uuid4()),
        name="doc.pdf",
        path=f"/doc-{uuid.uuid4().hex[:8]}.pdf",
        storage_path=f"{workspace_id}/doc.pdf",
        mime_type="application/pdf",
        size_bytes=3,
    )
    if folder_id:
        await bulk_move_files_to_folder(workspace_id, [stored.id], folder_id)
    return stored.id


async def _folder_of(file_id: str) -> str | None:
    row = await UserWorkspaceFile.prisma().find_unique(where={"id": file_id})
    assert row is not None
    return row.folderId


@pytest.mark.asyncio(loop_scope="session")
async def test_a_child_folder_carries_its_parent(server: SpinTestServer):
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")
    child = await create_folder(workspace, "2026", parent_id=parent.id)

    assert child.parent_id == parent.id

    listed = {f.id: f for f in await list_workspace_folders(workspace)}
    assert listed[child.id].parent_id == parent.id
    assert listed[parent.id].parent_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_name_clash_is_scoped_to_the_parent(server: SpinTestServer):
    """One name per level, not per workspace."""
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")
    await create_folder(workspace, "2026", parent_id=parent.id)

    with pytest.raises(FolderAlreadyExistsError):
        await create_folder(workspace, "2026", parent_id=parent.id)

    # The same name one level up is a different folder, and is allowed.
    root_2026 = await create_folder(workspace, "2026")
    assert root_2026.parent_id is None
    with pytest.raises(FolderAlreadyExistsError):
        await create_folder(workspace, "2026")


@pytest.mark.asyncio(loop_scope="session")
async def test_the_database_refuses_a_duplicate_name_at_either_level(
    server: SpinTestServer,
):
    """The partial unique indexes, reached past the app-level check: they are
    what makes two concurrent creates safe."""
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")

    for parent_id in (None, parent.id):
        data: dict = {"name": "Duplicate", "Workspace": {"connect": {"id": workspace}}}
        if parent_id is not None:
            data["Parent"] = {"connect": {"id": parent_id}}
        await UserWorkspaceFolder.prisma().create(data=data)
        with pytest.raises(UniqueViolationError):
            await UserWorkspaceFolder.prisma().create(data=data)


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_a_folder_frees_its_name_at_either_level(
    server: SpinTestServer,
):
    """Needs this branch's migration: before it, only the ROOT index excluded
    soft-deleted rows, so re-creating a deleted child answered 409 forever."""
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")

    for parent_id in (None, parent.id):
        first = await create_folder(workspace, "2026", parent_id=parent_id)
        await delete_folder(first.id, workspace)
        again = await create_folder(workspace, "2026", parent_id=parent_id)
        assert again.id != first.id


@pytest.mark.asyncio(loop_scope="session")
async def test_deleting_a_folder_takes_its_subtree_and_frees_the_files(
    server: SpinTestServer,
):
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")
    child = await create_folder(workspace, "2026", parent_id=parent.id)
    grandchild = await create_folder(workspace, "Q1", parent_id=child.id)
    sibling = await create_folder(workspace, "Photos")

    deep_file = await _file(workspace, grandchild.id)
    sibling_file = await _file(workspace, sibling.id)

    await delete_folder(parent.id, workspace)

    remaining = {f.id for f in await list_workspace_folders(workspace)}
    assert remaining == {sibling.id}
    # A grandchild left behind would keep its file hidden under a folder no
    # listing returns; the file goes back to the root instead.
    assert await _folder_of(deep_file) is None
    assert await _folder_of(sibling_file) == sibling.id


@pytest.mark.asyncio(loop_scope="session")
async def test_moving_a_folder_into_its_own_subtree_is_refused(
    server: SpinTestServer,
):
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")
    child = await create_folder(workspace, "2026", parent_id=parent.id)
    grandchild = await create_folder(workspace, "Q1", parent_id=child.id)

    for destination in (grandchild.id, child.id, parent.id):
        with pytest.raises(FolderValidationError):
            await move_folder(parent.id, workspace, destination)

    listed = {f.id: f.parent_id for f in await list_workspace_folders(workspace)}
    assert listed[parent.id] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_moving_a_folder_under_a_parent_and_back_to_the_root(
    server: SpinTestServer,
):
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")
    loose = await create_folder(workspace, "Receipts")

    moved = await move_folder(loose.id, workspace, parent.id)
    assert moved.parent_id == parent.id

    back = await move_folder(loose.id, workspace, None)
    assert back.parent_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_move_onto_a_taken_name_is_refused(server: SpinTestServer):
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")
    await create_folder(workspace, "Receipts", parent_id=parent.id)
    loose = await create_folder(workspace, "Receipts")

    with pytest.raises(FolderAlreadyExistsError):
        await move_folder(loose.id, workspace, parent.id)
