import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from prisma.errors import UniqueViolationError

from backend.data import db
from backend.data.skill_capacity import MAX_SKILLS_PER_EXPERT, SkillLimitError
from backend.data.workspace_skill import (
    WorkspaceSkillWrite,
    publish_workspace_skill_file,
)
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import LocalWorkspaceStorage


@pytest_asyncio.fixture(loop_scope="session")
async def workspace_id() -> AsyncIterator[str]:
    owned_connection = not db.is_connected()
    if owned_connection:
        await db.connect()
    user_id = str(uuid.uuid4())
    await db.prisma.user.create(
        data={"id": user_id, "email": f"capacity-{user_id}@example.com"}
    )
    workspace = await db.prisma.userworkspace.create(data={"userId": user_id})
    try:
        yield workspace.id
    finally:
        await db.prisma.user.delete(where={"id": user_id})
        if owned_connection:
            await db.disconnect()


def write(
    workspace_id: str, slug: str, folder: str = "/skills", overwrite: bool = True
):
    file_id = str(uuid.uuid4())
    return WorkspaceSkillWrite(
        workspace_id=workspace_id,
        file_id=file_id,
        name="SKILL.md",
        path=f"{folder}/{slug}/SKILL.md",
        storage_path=f"test://{file_id}",
        mime_type="text/markdown",
        size_bytes=20,
        overwrite=overwrite,
        metadata={"kind": "copilot_skill", "description": "test"},
    )


async def seed(workspace_id: str, count: int, folder: str = "/skills"):
    await db.prisma.userworkspacefile.create_many(
        data=[
            {
                "workspaceId": workspace_id,
                "name": "SKILL.md",
                "path": f"{folder}/skill-{i}/SKILL.md",
                "storagePath": f"test://{i}",
                "mimeType": "text/markdown",
                "sizeBytes": 20,
            }
            for i in range(count)
        ]
    )


@asynccontextmanager
async def repeatable_transaction():
    async with db.transaction() as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        yield tx


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("repeatable_read", [False, True])
async def test_concurrent_new_roots_cannot_both_take_last_slot(
    workspace_id: str, repeatable_read: bool
):
    await seed(workspace_id, MAX_SKILLS_PER_EXPERT - 1)
    with (
        patch("backend.data.workspace_skill.transaction", repeatable_transaction)
        if repeatable_read
        else nullcontext()
    ):
        results = await asyncio.gather(
            *(
                publish_workspace_skill_file(write(workspace_id, f"new-{i}"))
                for i in range(8)
            )
        )
    assert sum(result.status == "stored" for result in results) == 1
    assert sum(result.status == "capacity" for result in results) == 7
    assert (
        await db.prisma.userworkspacefile.count(
            where={"workspaceId": workspace_id, "isDeleted": False}
        )
        == MAX_SKILLS_PER_EXPERT
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_upserts_at_capacity_never_expose_a_free_slot(workspace_id: str):
    await seed(workspace_id, MAX_SKILLS_PER_EXPERT)
    existing = await db.prisma.userworkspacefile.find_unique(
        where={
            "workspaceId_path": {
                "workspaceId": workspace_id,
                "path": "/skills/skill-0/SKILL.md",
            }
        }
    )
    assert existing is not None
    replacement = write(workspace_id, "skill-0")
    replaced, rejected = await asyncio.gather(
        publish_workspace_skill_file(replacement),
        publish_workspace_skill_file(write(workspace_id, "extra")),
    )
    assert replaced.status == "stored"
    assert replaced.file is not None
    assert replaced.file.id == replacement.file_id
    assert replaced.file.id != existing.id
    retired = await db.prisma.userworkspacefile.find_unique(where={"id": existing.id})
    assert retired is not None and retired.isDeleted
    assert replaced.file.storage_path == replacement.storage_path
    assert replaced.replaced_storage_path == existing.storagePath
    assert rejected.status == "capacity"
    assert (
        await db.prisma.userworkspacefile.count(
            where={"workspaceId": workspace_id, "isDeleted": False}
        )
        == MAX_SKILLS_PER_EXPERT
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_personal_and_expert_folders_have_independent_capacity(workspace_id: str):
    await seed(workspace_id, MAX_SKILLS_PER_EXPERT)
    first, second = await asyncio.gather(
        publish_workspace_skill_file(
            write(workspace_id, "same-name", "/experts/first/skills")
        ),
        publish_workspace_skill_file(
            write(workspace_id, "same-name", "/experts/second/skills")
        ),
    )
    assert first.status == second.status == "stored"


@pytest.mark.asyncio(loop_scope="session")
async def test_another_users_workspace_has_independent_capacity(workspace_id: str):
    await seed(workspace_id, MAX_SKILLS_PER_EXPERT)
    other_user = str(uuid.uuid4())
    await db.prisma.user.create(
        data={"id": other_user, "email": f"capacity-{other_user}@example.com"}
    )
    try:
        other_workspace = await db.prisma.userworkspace.create(
            data={"userId": other_user}
        )
        result = await publish_workspace_skill_file(write(other_workspace.id, "first"))
        assert result.status == "stored"
        assert result.file is not None
        assert result.file.workspace_id == other_workspace.id
    finally:
        await db.prisma.user.delete(where={"id": other_user})


@pytest.mark.asyncio(loop_scope="session")
async def test_non_overwrite_rejection_preserves_root(workspace_id: str):
    original = await publish_workspace_skill_file(write(workspace_id, "same"))
    rejected = await publish_workspace_skill_file(
        write(workspace_id, "same", overwrite=False)
    )
    assert original.file is not None
    assert rejected.status == "exists"
    row = await db.prisma.userworkspacefile.find_unique(where={"id": original.file.id})
    assert row is not None
    assert row.storagePath == original.file.storage_path


@pytest.mark.asyncio(loop_scope="session")
async def test_failed_replacement_rolls_back_root_retirement(workspace_id: str):
    original = await publish_workspace_skill_file(write(workspace_id, "original"))
    other = await publish_workspace_skill_file(write(workspace_id, "other"))
    assert original.file is not None and other.file is not None
    replacement = write(workspace_id, "original").model_copy(
        update={"file_id": other.file.id}
    )
    with pytest.raises(UniqueViolationError):
        await publish_workspace_skill_file(replacement)
    row = await db.prisma.userworkspacefile.find_unique(where={"id": original.file.id})
    assert row is not None
    assert not row.isDeleted
    assert row.path == original.file.path
    assert row.storagePath == original.file.storage_path


@pytest.mark.asyncio(loop_scope="session")
async def test_nested_skill_files_and_deleted_roots_do_not_consume_slots(
    workspace_id: str,
):
    await seed(workspace_id, MAX_SKILLS_PER_EXPERT - 1)
    await db.prisma.userworkspacefile.create(
        data={
            "workspaceId": workspace_id,
            "name": "SKILL.md",
            "path": "/skills/skill-0/references/SKILL.md",
            "storagePath": "test://nested",
            "mimeType": "text/markdown",
            "sizeBytes": 1,
        }
    )
    await db.prisma.userworkspacefile.create(
        data={
            "workspaceId": workspace_id,
            "name": "SKILL.md",
            "path": "/skills/deleted/SKILL.md",
            "storagePath": "test://deleted",
            "mimeType": "text/markdown",
            "sizeBytes": 1,
            "isDeleted": True,
        }
    )
    result = await publish_workspace_skill_file(write(workspace_id, "last"))
    assert result.status == "stored"


@pytest.mark.asyncio(loop_scope="session")
async def test_manager_publishes_and_replaces_real_files_at_capacity(
    workspace_id: str, tmp_path: Path
):
    workspace = await db.prisma.userworkspace.find_unique(where={"id": workspace_id})
    assert workspace is not None
    manager = WorkspaceManager(workspace.userId, workspace_id)
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
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
        patch(
            "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
        ),
    ):
        original = await manager.write_file(
            b"old", "SKILL.md", path="/skills/existing/SKILL.md", overwrite=True
        )
        await seed(workspace_id, MAX_SKILLS_PER_EXPERT - 1)
        with pytest.raises(SkillLimitError):
            await manager.write_file(
                b"extra", "SKILL.md", path="/skills/extra/SKILL.md", overwrite=True
            )
        assert len(list(tmp_path.rglob("SKILL.md"))) == 1
        replacement = await manager.write_file(
            b"new", "SKILL.md", path="/skills/existing/SKILL.md", overwrite=True
        )
        assert replacement.id != original.id
        assert await manager.read_file_by_id(replacement.id) == b"new"
        with pytest.raises(FileNotFoundError):
            await manager.read_file_by_id(original.id)
        assert len(list(tmp_path.rglob("SKILL.md"))) == 1
