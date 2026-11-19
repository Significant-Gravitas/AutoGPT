"""Refused package writes must let a cold skill index become warm."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.store.exceptions import VirusDetectedError
from backend.copilot.tools import skills
from backend.copilot.tools.skill_updates_test import _active, _install, _package
from backend.copilot.tools.skills_index_cache_test import _skill
from backend.copilot.tools.skills_test import _cached_skills, _FakeWorkspaceManager
from backend.data.workspace_skill import WorkspaceSkillPublication
from backend.util import workspace


@pytest.fixture
def refused_package(monkeypatch, refusal):
    database = AsyncMock()
    database.get_workspace_total_size.return_value = 0
    database.get_workspace_file_by_path.return_value = None
    database.get_workspace_file.return_value = None
    publisher = AsyncMock()
    publisher.publish_workspace_skill_file.return_value = WorkspaceSkillPublication(
        status=refusal if refusal in {"capacity", "exists", "owned"} else "capacity"
    )
    storage = AsyncMock()
    storage.store.return_value = "test://package"
    monkeypatch.setattr(workspace, "workspace_db", lambda: database)
    monkeypatch.setattr(workspace, "workspace_skill_db", lambda: publisher)
    monkeypatch.setattr(
        workspace, "get_workspace_storage", AsyncMock(return_value=storage)
    )
    monkeypatch.setattr(
        workspace,
        "get_workspace_storage_limit_bytes",
        AsyncMock(return_value=1 if refusal == "quota" else 1_000_000),
    )
    monkeypatch.setattr(
        workspace,
        "scan_content_safe",
        AsyncMock(
            side_effect=VirusDetectedError("test") if refusal == "scan" else None
        ),
    )
    manager = workspace.WorkspaceManager("user-1", "workspace")
    return AsyncMock(wraps=manager.write_skill_package)


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["quota", "scan", "capacity", "exists", "owned"])
async def test_refused_marketplace_update_keeps_a_warm_index(
    monkeypatch, refused_package
):
    fake = _FakeWorkspaceManager()
    first = _package("checks", "v1", "old instructions")
    next_version = _package("checks", "v2", "new instructions")
    with _cached_skills(fake):
        await _install(first)
        marketplace = skills.skill_db()
        marketplace.get_active_versions.return_value = {"checks": _active(next_version)}
        marketplace.get_version_packages.return_value = {
            first.version_id: first,
            next_version.version_id: next_version,
        }
        monkeypatch.setattr(fake, "write_skill_package", refused_package)
        with patch.object(
            skills,
            "_list_user_skills_from_workspace",
            wraps=skills._list_user_skills_from_workspace,
        ) as read_folder:
            [listed] = await skills.list_user_skills("user-1")
            assert listed.baseline is not None
            assert listed.baseline.version_id == first.version_id
            assert listed.update == skills.UPDATE_AVAILABLE
            reads = read_folder.await_count
            assert await skills._read_skills_cache("user-1") is not None
            [warm] = await skills.list_user_skills("user-1")
            assert warm.baseline == listed.baseline
            assert warm.update == skills.UPDATE_AVAILABLE
            assert read_folder.await_count == reads
            refused_package.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["quota", "scan", "capacity", "exists", "owned"])
async def test_refused_assigned_copy_keeps_a_warm_index(monkeypatch, refused_package):
    fake = _FakeWorkspaceManager()
    fake.files["/skills/missing/SKILL.md"] = _skill("missing")
    with _cached_skills(fake):
        skills.experts_db().get_expert.return_value.skills = ["missing"]
        monkeypatch.setattr(fake, "write_skill_package", refused_package)
        with patch.object(
            skills,
            "_list_user_skills_from_workspace",
            wraps=skills._list_user_skills_from_workspace,
        ) as read_folder:
            assert await skills.list_user_skills("user-1", "expert-1") == []
            reads = read_folder.await_count
            assert await skills._read_skills_cache("user-1", "expert-1") == []
            assert await skills.list_user_skills("user-1", "expert-1") == []
            assert read_folder.await_count == reads
            refused_package.assert_awaited_once()
