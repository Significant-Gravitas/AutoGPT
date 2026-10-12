"""Real-database coverage for nested workspace folders.

Folders are a DB-level layer over files: a file's ``folderId`` and a folder's
``parentId`` decide the tree, and no storage path moves. These tests pin the
three things a tree adds over a flat list — creating a child, deleting a
subtree, and moving one without detaching it from the root.
"""

import asyncio
import contextlib
import uuid

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import UserWorkspace, UserWorkspaceFile, UserWorkspaceFolder

from backend.api.features.library.exceptions import (
    FolderAlreadyExistsError,
    FolderValidationError,
)
from backend.data import workspace_folder
from backend.data.user import get_or_create_user
from backend.data.workspace import create_workspace_file, get_or_create_workspace
from backend.data.workspace_folder import (
    apply_folder_update,
    bulk_move_files_to_folder,
    create_folder,
    delete_folder,
    list_workspace_folders,
    move_folder,
    resolve_attachable_workspace_folders,
)
from backend.data.workspace_scope import WorkspaceScope
from backend.util.exceptions import NotFoundError
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
async def test_two_moves_racing_into_each_other_cannot_build_a_cycle(
    server: SpinTestServer, mocker
):
    """Each mover reads the ancestor chain before either writes, so without the
    workspace move lock both pass their check and become each other's parent —
    a subtree no root-anchored listing can reach.

    Concurrency alone does not produce that interleaving, so the first mover is
    held inside its check until the second reaches one too. Under the lock the
    second never gets there, the wait times out, and the moves run in sequence,
    which is the whole point.
    """
    workspace = await _workspace()
    a = await create_folder(workspace, "A")
    b = await create_folder(workspace, "B")

    real_ancestors = workspace_folder._ancestor_ids
    both_checking = asyncio.Event()
    entered = 0

    async def gated(workspace_id: str, folder_id: str) -> list[str]:
        nonlocal entered
        entered += 1
        mine = entered
        if mine > 1:
            both_checking.set()
        chain = await real_ancestors(workspace_id, folder_id)
        if mine == 1:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(both_checking.wait(), timeout=2)
        return chain

    mocker.patch.object(workspace_folder, "_ancestor_ids", gated)

    outcomes = await asyncio.gather(
        move_folder(a.id, workspace, b.id),
        move_folder(b.id, workspace, a.id),
        return_exceptions=True,
    )

    assert entered == 2, "both movers must have reached the cycle check"
    refused = [o for o in outcomes if isinstance(o, FolderValidationError)]
    assert len(refused) == 1, outcomes

    parents = {f.id: f.parent_id for f in await list_workspace_folders(workspace)}
    assert None in (parents[a.id], parents[b.id]), parents


@pytest.mark.asyncio(loop_scope="session")
async def test_creating_under_an_unknown_parent_is_refused(server: SpinTestServer):
    """``parentId`` is written directly and the FK only proves the row exists,
    so a parent from another workspace has to be refused here or not at all."""
    workspace = await _workspace()

    with pytest.raises(NotFoundError):
        await create_folder(workspace, "Orphan", parent_id=str(uuid.uuid4()))


@pytest.mark.asyncio(loop_scope="session")
async def test_a_move_and_rename_in_one_patch_apply_together(
    server: SpinTestServer,
):
    workspace = await _workspace()
    destination = await create_folder(workspace, "Invoices")
    moving = await create_folder(workspace, "Source")

    updated = await apply_folder_update(
        moving.id, workspace, parent_id=destination.id, name="2026"
    )

    assert (updated.parent_id, updated.name) == (destination.id, "2026")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_refused_rename_leaves_the_folder_where_it_was(
    server: SpinTestServer,
):
    """The move used to be written before the rename was checked, so a PATCH
    doing both left the folder at its new parent under its old name and the
    client holding a 409 — half of a request it had been told failed."""
    workspace = await _workspace()
    destination = await create_folder(workspace, "Invoices")
    await create_folder(workspace, "Taken", parent_id=destination.id)
    moving = await create_folder(workspace, "Source")

    with pytest.raises(FolderAlreadyExistsError):
        await apply_folder_update(
            moving.id, workspace, parent_id=destination.id, name="Taken"
        )

    parents = {f.id: f.parent_id for f in await list_workspace_folders(workspace)}
    assert parents[moving.id] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_move_onto_a_taken_name_is_refused(server: SpinTestServer):
    workspace = await _workspace()
    parent = await create_folder(workspace, "Invoices")
    await create_folder(workspace, "Receipts", parent_id=parent.id)
    loose = await create_folder(workspace, "Receipts")

    with pytest.raises(FolderAlreadyExistsError):
        await move_folder(loose.id, workspace, parent.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_attaching_folders_drops_foreign_and_deleted_ones_and_gates_on_scope(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    workspace = await _workspace()
    owner = await UserWorkspace.prisma().find_unique(where={"id": workspace})
    assert owner is not None
    mine = await create_folder(workspace, "Invoices")
    deleted = await create_folder(workspace, "Old")
    await delete_folder(deleted.id, workspace)
    foreign = await create_folder(await _workspace(), "Theirs")
    ids = [mine.id, deleted.id, foreign.id]

    resolved = await resolve_attachable_workspace_folders(
        owner.userId, ids, expert_id=None
    )
    assert [f.id for f in resolved] == [mine.id]

    # A missing expert resolves to a fail-closed scope: no folders at all.
    assert (
        await resolve_attachable_workspace_folders(
            owner.userId, ids, expert_id=str(uuid.uuid4())
        )
        == []
    )

    async def granted(user_id: str, expert_id: str) -> WorkspaceScope:
        return WorkspaceScope(expert_id=expert_id, reads_user_files=True)

    monkeypatch.setattr(workspace_folder, "resolve_expert_workspace_scope", granted)
    resolved = await resolve_attachable_workspace_folders(
        owner.userId, ids, expert_id="exp-1"
    )
    assert [f.id for f in resolved] == [mine.id]
