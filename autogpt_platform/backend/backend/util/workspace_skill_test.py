from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.data.skill_capacity import SkillLimitError, SkillOwnedError
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_skill import WorkspaceSkillPublication
from backend.util.workspace import WorkspaceManager


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
