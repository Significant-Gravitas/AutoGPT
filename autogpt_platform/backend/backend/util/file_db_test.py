"""Real-database coverage for where ``store_media_file`` keeps a block's output."""

import uuid
from collections.abc import AsyncIterator
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


@pytest.mark.asyncio(loop_scope="session")
async def test_runs_storing_one_filename_each_keep_their_own_file(
    workspace: UserWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(file_module, "TEMP_DIR", (tmp_path / "temp").resolve())
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path / "storage"))
    refs = []
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
        for run in range(3):
            graph_exec_id = str(uuid.uuid4())
            output = Path(get_exec_file_path(graph_exec_id, "output.mp4"))
            output.parent.mkdir(parents=True)
            output.write_bytes(f"run {run}".encode())
            context = ExecutionContext(
                user_id=workspace.userId,
                graph_exec_id=graph_exec_id,
                workspace_id=workspace.id,
            )
            refs.append(
                await store_media_file(
                    MediaFileType("output.mp4"),
                    context,
                    return_format="for_block_output",
                )
            )

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
