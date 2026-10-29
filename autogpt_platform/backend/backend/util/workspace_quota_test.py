"""Real-database proof that writes racing for the last of a workspace's storage
cannot all land: each is decided where its row commits, one at a time."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from backend.data import db
from backend.data import workspace as workspace_data
from backend.util.workspace import SkillPackageFile, WorkspaceManager
from backend.util.workspace_storage import LocalWorkspaceStorage

LIMIT = 1_000
SIZE = 400  # each write fits alone; only two fit together
WRITERS = 3


@pytest_asyncio.fixture(loop_scope="session")
async def manager() -> AsyncIterator[WorkspaceManager]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user_id = str(uuid.uuid4())
    await db.prisma.user.create(
        data={"id": user_id, "email": f"quota-{user_id}@example.com"}
    )
    workspace = await db.prisma.userworkspace.create(data={"userId": user_id})
    try:
        yield WorkspaceManager(user_id, workspace.id)
    finally:
        await db.prisma.user.delete(where={"id": user_id})
        if owned_connection:
            await db.disconnect()


async def _write_file(manager: WorkspaceManager, i: int, size: int = SIZE) -> None:
    await manager.write_file(b"x" * size, f"f{i}.bin", overwrite=True)


async def _write_package(manager: WorkspaceManager, i: int, size: int = SIZE) -> None:
    # A folder of its own per writer, so the skill-capacity lock cannot be
    # what serialises them.
    root = SkillPackageFile(
        path=f"/experts/e{i}/skills/s/SKILL.md", content=b"x" * size
    )
    await manager.write_skill_package(root, [], [])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "write", [_write_file, _write_package], ids=["file", "package"]
)
async def test_racing_writes_never_take_the_workspace_past_its_quota(
    manager: WorkspaceManager, write, tmp_path: Path
):
    with (
        _real_storage(tmp_path),
        patch.object(workspace_data, "_live_size", _held_until_all_have_read(WRITERS)),
    ):
        results = await asyncio.gather(
            *(write(manager, i) for i in range(WRITERS)), return_exceptions=True
        )

    refused = [r for r in results if isinstance(r, Exception)]
    assert all(str(r).startswith("Storage limit exceeded") for r in refused), refused
    assert len(refused) == WRITERS - LIMIT // SIZE
    live = await _live_files(manager)
    assert sum(f.sizeBytes for f in live) <= LIMIT
    # A refused write stored its blob before the commit refused it.
    assert len([p for p in tmp_path.rglob("*") if p.is_file()]) == len(live)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "write", [_write_file, _write_package], ids=["file", "package"]
)
async def test_replacing_a_file_near_the_quota_counts_only_the_new_copy(
    manager: WorkspaceManager, write, tmp_path: Path
):
    with _real_storage(tmp_path):
        await write(manager, 0, size=LIMIT - 1)
        await write(manager, 0, size=LIMIT - 1)

    assert [f.sizeBytes for f in await _live_files(manager)] == [LIMIT - 1]


@contextmanager
def _real_storage(tmp_path: Path) -> Iterator[None]:
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            AsyncMock(return_value=storage),
        ),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            AsyncMock(return_value=LIMIT),
        ),
        patch("backend.util.workspace.scan_content_safe", AsyncMock()),
        patch(
            "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
        ),
    ):
        yield


async def _live_files(manager: WorkspaceManager):
    return await db.prisma.userworkspacefile.find_many(
        where={"workspaceId": manager.workspace_id, "isDeleted": False}
    )


def _held_until_all_have_read(parties: int):
    """Hold each quota read until every writer has made one, so without the
    lock they all count the same usage. With it the others never get to read,
    so the first gives up waiting and lets the rest through."""
    barrier = asyncio.Barrier(parties)
    read = workspace_data._live_size

    async def live_size(*args, **kwargs) -> int:
        used = await read(*args, **kwargs)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=2)
        except TimeoutError:
            await barrier.abort()
        except asyncio.BrokenBarrierError:
            pass
        return used

    return live_size
