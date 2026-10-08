from unittest.mock import AsyncMock

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools import skills
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.skills import SkillFile
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.data.skill_publication import VersionDraft
from backend.data.skill_version_files import SkillVersionFile


@pytest.fixture
def package_workspace(monkeypatch):
    manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )
    with _patch_skills_path(manager):
        yield manager


@pytest.mark.asyncio
async def test_script_only_change_is_an_immutable_new_version(
    package_workspace, fake_learning_store
):
    for text in ("print('first')", "print('second')"):
        await skills.store_user_skill(
            "user-1",
            name="checks",
            description="Run checks",
            body="Run scripts/check.py",
            files=[SkillFile(relative_path="scripts/check.py", content=text.encode())],
        )
    versions = await fake_learning_store.list_versions("user-1", "personal", "checks")
    assert len(versions) == 2 and versions[0].content_hash == versions[1].content_hash
    assert versions[0].files[0].content == b"print('second')"
    assert versions[1].files[0].content == b"print('first')"


@pytest.mark.asyncio
async def test_interrupted_package_is_not_loaded_with_mixed_file_versions(
    package_workspace, fake_learning_store
):
    await skills.store_user_skill(
        "user-1",
        name="checks",
        description="Run checks",
        body="Run scripts/check.py",
        files=[SkillFile(relative_path="scripts/check.py", content=b"print('first')")],
    )
    head = await fake_learning_store.get_head("user-1", "personal", "checks")
    old = await fake_learning_store.get_version("user-1", head.current_version_id)
    pending = await fake_learning_store.commit_version_safe(
        "user-1",
        head=head,
        draft=VersionDraft(
            content=old.content,
            files=[
                SkillVersionFile.from_content("scripts/check.py", b"print('second')")
            ],
            description="Run checks",
            origin="saved_overnight",
        ),
        expected_current_version=head.current_version,
    )
    assert pending.committed
    package_workspace.files["/skills/checks/scripts/check.py"] = b"print('second')"
    response = await skills.ReadSkillTool()._execute(
        "user-1", ChatSession.new("user-1", dry_run=False), name="checks"
    )
    assert isinstance(response, ErrorResponse)
    assert response.error == "skill_registry_unavailable"


@pytest.mark.asyncio
async def test_untracked_script_edit_is_not_attributed_to_the_old_version(
    package_workspace, fake_learning_store
):
    await skills.store_user_skill(
        "user-1",
        name="checks",
        description="Run checks",
        body="Run scripts/check.py",
        files=[SkillFile(relative_path="scripts/check.py", content=b"print('first')")],
    )
    text = package_workspace.files["/skills/checks/SKILL.md"].decode()
    initial = await skills.resolve_loaded_skill_version("user-1", None, "checks", text)
    assert initial.version == 1
    package_workspace.files["/skills/checks/scripts/check.py"] = b"print('owner edit')"
    edited = await skills.resolve_loaded_skill_version("user-1", None, "checks", text)
    assert edited.version_id is None
