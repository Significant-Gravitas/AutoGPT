"""Real-database coverage for where ``store_media_file`` keeps a block's output."""

import posixpath
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from prisma.models import UserWorkspace

from backend.data import db
from backend.data.execution import ExecutionContext
from backend.util import file as file_module
from backend.util.file import get_exec_file_path, parse_workspace_uri, store_media_file
from backend.util.type import MediaFileType
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import LocalWorkspaceStorage


@pytest_asyncio.fixture(loop_scope="session")
async def workspace() -> AsyncIterator[UserWorkspace]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user_id = str(uuid.uuid4())
    await db.prisma.user.create(
        data={"id": user_id, "email": f"outputs-{user_id}@example.com"}
    )
    workspace = await db.prisma.userworkspace.create(data={"userId": user_id})
    try:
        yield workspace
    finally:
        await db.prisma.user.delete(where={"id": user_id})
        if owned_connection:
            await db.disconnect()


@pytest.fixture
def local_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Outputs stored under ``tmp_path``, with a quota set so every row is
    inserted inside the storage-quota transaction."""
    monkeypatch.setattr(file_module, "TEMP_DIR", (tmp_path / "temp").resolve())
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path / "storage"))
    with (
        patch(
            "backend.util.workspace.get_workspace_storage",
            AsyncMock(return_value=storage),
        ),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            AsyncMock(return_value=1_000_000),
        ),
        patch("backend.util.workspace.scan_content_safe", AsyncMock()),
        patch("backend.util.file.scan_content_safe", AsyncMock()),
        patch(
            "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
        ),
    ):
        yield


@pytest.mark.asyncio(loop_scope="session")
async def test_runs_storing_one_filename_each_keep_their_own_file(
    workspace: UserWorkspace, local_outputs: None
):
    refs = [
        await _store_run_output(workspace, f"run {run}".encode()) for run in range(3)
    ]

    manager = WorkspaceManager(workspace.userId, workspace.id)
    file_ids = [parse_workspace_uri(ref).file_ref for ref in refs]
    assert [await manager.read_file_by_id(i) for i in file_ids] == [
        b"run 0",
        b"run 1",
        b"run 2",
    ]
    files = [await manager.get_file_info(i) for i in file_ids]
    assert [f.name if f else None for f in files] == [
        "output.mp4",
        "output (1).mp4",
        "output (2).mp4",
    ]
    assert await manager.read_file("/output.mp4") == b"run 0"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_that_loses_its_free_name_mid_write_takes_the_next_one(
    workspace: UserWorkspace, local_outputs: None
):
    """Another write takes ``output (1).mp4`` between our look and our insert:
    the insert is refused in its quota transaction and lands at the next free
    name, and the file that won stays live."""
    free_path = WorkspaceManager._free_path
    racer_ids: list[str] = []

    async def free_path_lost_to_a_racer(self: WorkspaceManager, path: str) -> str:
        free = await free_path(self, path)
        if not racer_ids:
            racer = await self.write_file(
                content=b"racer", filename=posixpath.basename(free), path=free
            )
            racer_ids.append(racer.id)
        return free

    first = await _store_run_output(workspace, b"run 0")
    with patch.object(WorkspaceManager, "_free_path", free_path_lost_to_a_racer):
        second = await _store_run_output(workspace, b"run 1")

    manager = WorkspaceManager(workspace.userId, workspace.id)
    file_ids = [parse_workspace_uri(ref).file_ref for ref in (first, second)]
    file_ids += racer_ids
    files = [await manager.get_file_info(i) for i in file_ids]
    assert [f.name if f else None for f in files] == [
        "output.mp4",
        "output (2).mp4",
        "output (1).mp4",
    ]
    assert [await manager.read_file_by_id(i) for i in file_ids] == [
        b"run 0",
        b"run 1",
        b"racer",
    ]


async def _store_run_output(workspace: UserWorkspace, content: bytes) -> str:
    """Store ``output.mp4`` from a new run as a block's output."""
    graph_exec_id = str(uuid.uuid4())
    output = Path(get_exec_file_path(graph_exec_id, "output.mp4"))
    output.parent.mkdir(parents=True)
    output.write_bytes(content)
    context = ExecutionContext(
        user_id=workspace.userId,
        graph_exec_id=graph_exec_id,
        workspace_id=workspace.id,
    )
    return await store_media_file(
        MediaFileType("output.mp4"), context, return_format="for_block_output"
    )
