from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools import skills
from backend.data.skill_capacity import SKILL_ORIGIN_MARKETPLACE
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_skill import WorkspaceSkillPublication, WorkspaceSkillWrite
from backend.util import workspace, workspace_storage
from backend.util.settings import Config


@pytest.fixture
def package_writes(monkeypatch):
    manager = workspace.WorkspaceManager("owner", "workspace")
    database = AsyncMock()
    database.get_workspace_total_size.return_value = 0
    database.get_workspace_file_by_path.return_value = None
    database.list_workspace_files.return_value = []
    publisher = AsyncMock()
    publisher.publish_workspace_skill_file.side_effect = _published
    storage = AsyncMock()
    storage.store.return_value = "test://package"
    monkeypatch.setattr(workspace_storage, "_local_storage", storage)
    monkeypatch.setattr(workspace, "workspace_db", lambda: database)
    monkeypatch.setattr(workspace, "workspace_skill_db", lambda: publisher)
    limit = AsyncMock(return_value=10_000_000)
    monkeypatch.setattr(workspace, "get_workspace_storage_limit_bytes", limit)
    scan = AsyncMock()
    monkeypatch.setattr(workspace, "scan_content_safe", scan)
    monkeypatch.setattr(manager, "_schedule_embedding", lambda *_: None)
    monkeypatch.setattr(
        skills, "_acquire_skill_write_lock", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        skills, "_get_user_skill_manager", AsyncMock(return_value=manager)
    )
    monkeypatch.setattr(
        skills, "_list_user_skills_from_workspace", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(skills, "invalidate_skills_index_cache", AsyncMock())
    monkeypatch.setenv("PRIVATE_USER_DATA_BUCKET", "")
    monkeypatch.setenv("MEDIA_GCS_BUCKET_NAME", "")
    monkeypatch.setenv("BEHAVE_AS", "local")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "2")
    return database, publisher, storage, limit, scan


def _published(write: WorkspaceSkillWrite) -> WorkspaceSkillPublication:
    return WorkspaceSkillPublication(
        status="stored",
        file=WorkspaceFile(
            id=write.file_id,
            workspace_id=write.workspace_id,
            name=write.name,
            path=write.path,
            storage_path=write.storage_path,
            mime_type=write.mime_type,
            size_bytes=write.size_bytes,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        ),
    )


async def _write_batch(files: list[skills.SkillFile] | None = None):
    return await skills.store_user_skills(
        "owner",
        [
            skills.SkillWrite(
                name=f"skill-{i}", description="Example", body="body", files=files
            )
            for i in range(3)
        ],
        origin=SKILL_ORIGIN_MARKETPLACE,
        notify_updates=False,
    )


@pytest.mark.asyncio
async def test_skill_batch_reads_configuration_once_and_checks_each_write(
    monkeypatch, package_writes
):
    database, publisher, storage, limit, scan = package_writes
    reads = []
    initialize = Config.__init__

    def record_config(self, *args, **kwargs):
        reads.append(None)
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(Config, "__init__", record_config)

    outcomes = await _write_batch()

    assert all(isinstance(outcome, skills.StoredSkill) for outcome in outcomes)
    assert len(reads) == 1
    assert [
        call.args[0].path
        for call in publisher.publish_workspace_skill_file.await_args_list
    ] == [f"/skills/skill-{i}/SKILL.md" for i in range(3)]
    assert limit.await_count == database.get_workspace_total_size.await_count == 3
    assert scan.await_count == storage.store.await_count == 3


@pytest.mark.asyncio
async def test_later_skill_batch_observes_a_changed_file_size_limit(
    monkeypatch, package_writes
):
    _, publisher, _, _, _ = package_writes

    files = [
        skills.SkillFile(relative_path="guide.txt", content=b"x" * (1024 * 1024 + 1))
    ]
    outcomes = await _write_batch(files)
    assert all(isinstance(outcome, skills.StoredSkill) for outcome in outcomes)

    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    refused = await _write_batch(files)

    assert all(isinstance(outcome, ValueError) for outcome in refused)
    assert all("1MB limit" in str(outcome) for outcome in refused)
    assert publisher.publish_workspace_skill_file.await_count == 3


@pytest.mark.asyncio
async def test_skill_batch_rechecks_quota_before_each_package(package_writes):
    _, publisher, _, limit, _ = package_writes
    limit.side_effect = [10_000_000, 1, 10_000_000]

    first, second, third = await _write_batch()

    assert isinstance(first, skills.StoredSkill)
    assert isinstance(second, ValueError)
    assert "Storage limit exceeded" in str(second)
    assert isinstance(third, skills.StoredSkill)
    assert [
        call.args[0].path
        for call in publisher.publish_workspace_skill_file.await_args_list
    ] == ["/skills/skill-0/SKILL.md", "/skills/skill-2/SKILL.md"]
