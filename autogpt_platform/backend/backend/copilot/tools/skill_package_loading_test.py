from unittest.mock import AsyncMock

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools import skills
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.skill_package_versions_test import (
    package_workspace as workspace_fixture,
)

package_workspace = workspace_fixture


async def _save_package():
    await skills.store_user_skill(
        "user-1",
        name="checks",
        description="Run checks",
        body="Run scripts/check.py",
        files=[
            skills.SkillFile(
                relative_path="scripts/check.py", content=b"print('verified')"
            )
        ],
    )


@pytest.mark.asyncio
async def test_read_reuses_the_scoped_workspace_manager(
    package_workspace, fake_learning_store, monkeypatch
):
    await _save_package()
    lookup = AsyncMock(return_value=package_workspace)
    monkeypatch.setattr(skills, "_get_user_skill_manager", lookup)
    response = await skills.ReadSkillTool()._execute(
        "user-1", ChatSession.new("user-1", dry_run=False), name="checks"
    )
    assert isinstance(response, skills.ReadSkillResponse)
    assert response.version == 1
    lookup.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_second_listing_is_not_reported_as_a_changed_package(
    package_workspace, fake_learning_store, monkeypatch
):
    await _save_package()
    files = await skills._list_package_files(package_workspace, "/skills", "checks")
    monkeypatch.setattr(
        skills,
        "_list_package_files",
        AsyncMock(side_effect=[files, OSError("unavailable")]),
    )
    sync = AsyncMock()
    monkeypatch.setattr(skills, "_sync_skill_package", sync)
    response = await skills.ReadSkillTool()._execute(
        "user-1", ChatSession.new("user-1", dry_run=False), name="checks"
    )
    assert isinstance(response, ErrorResponse)
    assert response.error == "skill_package_unavailable"
    sync.assert_not_awaited()
