import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.data.skill_capacity import SkillLimitError, SkillOwnedError
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_skill import WorkspaceSkillPublication
from backend.util.workspace import SkillPackageFile, WorkspaceManager


def stored_file():
    return WorkspaceFile(
        id="new-root",
        workspace_id="workspace",
        name="SKILL.md",
        path="/skills/example/SKILL.md",
        storage_path="test://new",
        mime_type="text/markdown",
        size_bytes=4,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["capacity", "exists", "stored", "error", "owned"])
async def test_skill_publication_cleans_only_the_appropriate_blob(status: str):
    database = AsyncMock()
    database.get_workspace_total_size.return_value = 0
    database.get_workspace_file_by_path.return_value = None
    publisher = AsyncMock()
    storage = AsyncMock()
    storage.store.return_value = "test://new"
    if status == "error":
        publisher.publish_workspace_skill_file.side_effect = RuntimeError(
            "DB unavailable"
        )
    elif status == "stored":
        publisher.publish_workspace_skill_file.return_value = WorkspaceSkillPublication(
            status="stored",
            file=stored_file(),
            replaced_storage_path="test://old",
            replaced_file_id="old-root",
        )
    else:
        publisher.publish_workspace_skill_file.return_value = WorkspaceSkillPublication(
            status=status
        )
    with (
        patch("backend.util.workspace.workspace_db", return_value=database),
        patch("backend.util.workspace.workspace_skill_db", return_value=publisher),
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
            "backend.api.features.workspace.embeddings.delete_workspace_file_embedding",
            AsyncMock(),
        ) as delete_embedding,
        patch(
            "backend.api.features.workspace.embeddings.schedule_workspace_file_embedding"
        ),
    ):
        manager = WorkspaceManager("user", "workspace")
        if status == "stored":
            result = await manager.write_file(
                b"body", "SKILL.md", path="/skills/example/SKILL.md", overwrite=True
            )
            assert result.id == "new-root"
            storage.delete.assert_awaited_once_with("test://old")
            delete_embedding.assert_awaited_once_with(
                file_id="old-root", user_id="user"
            )
        else:
            error_type = (
                SkillLimitError
                if status == "capacity"
                else (
                    SkillOwnedError
                    if status == "owned"
                    else ValueError if status == "exists" else RuntimeError
                )
            )
            with pytest.raises(error_type):
                await manager.write_file(
                    b"body", "SKILL.md", path="/skills/example/SKILL.md", overwrite=True
                )
            storage.delete.assert_awaited_once_with("test://new")
            delete_embedding.assert_not_awaited()
    database.create_workspace_file.assert_not_awaited()
    database.soft_delete_workspace_file.assert_not_awaited()


def package_manager(database, publisher, storage):
    return (
        patch("backend.util.workspace.workspace_db", return_value=database),
        patch("backend.util.workspace.workspace_skill_db", return_value=publisher),
        patch(
            "backend.util.workspace.get_workspace_storage",
            AsyncMock(return_value=storage),
        ),
        patch(
            "backend.util.workspace.get_workspace_storage_limit_bytes",
            AsyncMock(return_value=1_000_000),
        ),
        patch("backend.util.workspace.scan_content_safe", AsyncMock()),
    )


def package_write():
    return dict(
        root=SkillPackageFile(path="/skills/example/SKILL.md", content=b"root"),
        files=[
            SkillPackageFile(path="/skills/example/a.md", content=b"a"),
            SkillPackageFile(path="/skills/example/b.md", content=b"b"),
        ],
        removed_paths=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, published, error",
    [
        ("cancelled", False, asyncio.CancelledError),
        ("refused", False, SkillLimitError),
        ("rolled_back", False, RuntimeError),
        ("reply_lost", True, RuntimeError),
    ],
)
async def test_a_failed_package_write_drops_only_unpublished_blobs(
    failure: str, published: bool, error: type[BaseException]
):
    """A save that stops part-way, cancelled included, deletes the blobs it
    staged; one whose publication committed but whose reply was lost keeps
    them, since its rows now name them."""
    database = AsyncMock()
    database.get_workspace_total_size.return_value = 0
    database.get_workspace_file_by_path.return_value = None
    database.get_workspace_file.return_value = stored_file() if published else None
    publisher = AsyncMock()
    storage = AsyncMock()
    blobs = iter(["test://root", "test://a", "test://b"])

    async def store(**kwargs):
        if failure == "cancelled" and kwargs["filename"] == "b.md":
            raise asyncio.CancelledError
        return next(blobs)

    storage.store.side_effect = store
    if failure == "refused":
        publisher.publish_workspace_skill_file.return_value = WorkspaceSkillPublication(
            status="capacity"
        )
    else:
        publisher.publish_workspace_skill_file.side_effect = RuntimeError(
            "DB unavailable"
        )
    patches = package_manager(database, publisher, storage)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        with pytest.raises(error):
            await WorkspaceManager("user", "workspace").write_skill_package(
                **package_write()
            )

    deleted = {call.args[0] for call in storage.delete.await_args_list}
    if failure == "cancelled":
        publisher.publish_workspace_skill_file.assert_not_awaited()
        assert deleted == {"test://root", "test://a"}
    elif published:
        assert deleted == set()
    else:
        assert deleted == {"test://root", "test://a", "test://b"}


@pytest.mark.asyncio
async def test_a_cancel_during_refusal_cleanup_still_drops_the_blobs():
    """A refused package whose save is cancelled while it cleans up still
    deletes everything it staged."""
    checking, release = asyncio.Event(), asyncio.Event()

    async def held_check(*_args):
        checking.set()
        await release.wait()

    database = AsyncMock()
    database.get_workspace_total_size.return_value = 0
    database.get_workspace_file_by_path.return_value = None
    database.get_workspace_file.side_effect = held_check
    publisher = AsyncMock()
    publisher.publish_workspace_skill_file.return_value = WorkspaceSkillPublication(
        status="capacity"
    )
    storage = AsyncMock()
    blobs = iter(["test://root", "test://a", "test://b"])
    storage.store.side_effect = lambda **_kwargs: next(blobs)
    patches = package_manager(database, publisher, storage)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        save = asyncio.create_task(
            WorkspaceManager("user", "workspace").write_skill_package(**package_write())
        )
        await asyncio.wait_for(checking.wait(), 5)
        save.cancel()
        with pytest.raises(asyncio.CancelledError):
            await save
        release.set()
        for _ in range(10):
            await asyncio.sleep(0)

    deleted = {call.args[0] for call in storage.delete.await_args_list}
    assert deleted == {"test://root", "test://a", "test://b"}
