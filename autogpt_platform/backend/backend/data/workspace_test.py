"""Real-database coverage for the path a soft-deleted workspace file leaves.

The unique index over ``(workspaceId, path)`` covers live rows only, so a
deleted file keeps the path it was deleted at and the next write takes it.
``write_file(overwrite=True)`` soft-deletes the row it replaces, so this is
the path two writes to one path race on.
"""

import uuid
from datetime import datetime, timezone

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import UserWorkspaceFile

from backend.data import workspace as workspace_db
from backend.data.user import get_or_create_user
from backend.data.workspace import (
    create_workspace_file,
    get_or_create_workspace,
    get_workspace_file_by_path,
    list_workspace_files,
    soft_delete_workspace_file,
)
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_one_path_survives_repeated_create_and_delete(server: SpinTestServer):
    """Needs this branch's migration: the old index spanned deleted rows, so
    only the rename-on-delete kept the path writable, and every round left
    another row that had to be renamed out of the way first."""
    workspace = await _workspace()

    written = []
    for _ in range(6):
        written.append(await _write(workspace, "/a.txt"))
        listed = await list_workspace_files(workspace)
        assert [f.id for f in listed] == [written[-1]]
        await soft_delete_workspace_file(written[-1], workspace)
        assert await list_workspace_files(workspace) == []

    assert len(set(written)) == 6
    kept = await list_workspace_files(workspace, include_deleted=True)
    assert {f.id for f in kept} == set(written)
    assert {f.path for f in kept} == {"/a.txt"}


@pytest.mark.asyncio(loop_scope="session")
async def test_two_deletes_at_one_instant_both_keep_the_path(
    server: SpinTestServer, mocker
):
    """The case the rename could not survive: it separated deleted rows by a
    timestamp, so two deletes close enough together collided on the very index
    the rename exists to dodge. Freezing the clock is stronger than the second
    the suffix once measured — nothing here reads it."""
    workspace = await _workspace()
    frozen = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
    mocker.patch.object(
        workspace_db,
        "datetime",
        mocker.MagicMock(now=mocker.MagicMock(return_value=frozen)),
    )

    first = await _write(workspace, "/a.txt")
    await soft_delete_workspace_file(first, workspace)
    second = await _write(workspace, "/a.txt")
    await soft_delete_workspace_file(second, workspace)

    rows = await list_workspace_files(workspace, include_deleted=True)
    assert {r.id for r in rows} == {first, second}
    assert {r.path for r in rows} == {"/a.txt"}
    assert {r.deleted_at for r in rows} == {frozen}


@pytest.mark.asyncio(loop_scope="session")
async def test_a_deleted_file_keeps_the_path_it_was_deleted_at(
    server: SpinTestServer,
):
    """What the rename cost: the record it preserved no longer said where the
    file had been, and recovering one meant parsing a suffix back off."""
    workspace = await _workspace()
    file_id = await _write(workspace, "/a.txt")
    await soft_delete_workspace_file(file_id, workspace)

    row = await UserWorkspaceFile.prisma().find_unique(where={"id": file_id})
    assert row is not None
    assert row.path == "/a.txt"
    assert row.isDeleted is True


@pytest.mark.asyncio(loop_scope="session")
async def test_the_database_still_refuses_two_live_files_at_one_path(
    server: SpinTestServer,
):
    """``write_file(overwrite=True)`` reads this violation to learn it must
    replace the file already there, so the index has to keep raising it."""
    workspace = await _workspace()
    await _write(workspace, "/a.txt")

    with pytest.raises(UniqueViolationError):
        await _write(workspace, "/a.txt")


@pytest.mark.asyncio(loop_scope="session")
async def test_reads_pass_over_the_deleted_rows_sharing_a_path(
    server: SpinTestServer,
):
    """One path may now hold several rows — one live, any number deleted — so
    every read promising a live file has to filter for itself."""
    workspace = await _workspace()
    older = await _write(workspace, "/a.txt")
    await soft_delete_workspace_file(older, workspace)
    newer = await _write(workspace, "/a.txt")
    await soft_delete_workspace_file(newer, workspace)
    live = await _write(workspace, "/a.txt")

    found = await get_workspace_file_by_path(workspace, "/a.txt")
    assert found is not None and found.id == live

    assert [f.id for f in await list_workspace_files(workspace)] == [live]
    assert {
        f.id for f in await list_workspace_files(workspace, include_deleted=True)
    } == {
        older,
        newer,
        live,
    }


async def _workspace() -> str:
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"file-paths-{suffix}@example.com",
            "name": "File Owner",
        }
    )
    workspace = await get_or_create_workspace(user.id)
    return workspace.id


async def _write(workspace_id: str, path: str) -> str:
    stored = await create_workspace_file(
        workspace_id=workspace_id,
        file_id=str(uuid.uuid4()),
        name=path.rsplit("/", 1)[-1],
        path=path,
        storage_path=f"{workspace_id}{path}",
        mime_type="text/plain",
        size_bytes=3,
    )
    return stored.id
