from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.copilot.learning import nightly, publish
from backend.copilot.model import ChatSession
from backend.copilot.tools import skills
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.skills import (
    ParsedSkill,
    ReadSkillResponse,
    render_skill_markdown,
)
from backend.data import skill_publication as publication
from backend.data import skill_versions as versions
from backend.data.skill_learning_test import _cleanup, _create_user


@pytest.mark.asyncio(loop_scope="session")
async def test_read_repairs_owner_write_with_learning_disabled_and_respects_use_pause(
    server, tmp_path, monkeypatch
):
    user = str(uuid4())
    name = "recoverable-owner-edit"
    await _create_user(user)
    monkeypatch.setattr("backend.util.workspace.scan_content_safe", AsyncMock())
    flag = AsyncMock(return_value=False)
    monkeypatch.setattr(nightly, "is_feature_enabled", flag)
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(skills, "workdir_root", lambda _: str(tmp_path / "sandbox"))
    monkeypatch.setattr(skills, "read_workdir_bytes", AsyncMock(return_value=None))
    monkeypatch.setattr(skills, "remove_from_workdir", AsyncMock(return_value=[]))
    monkeypatch.setattr(skills, "set_executable", AsyncMock(return_value=[]))

    async def materialize(path, content, session_id):
        target = Path(path)
        assert target.is_relative_to(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(skills, "save_to_workdir", materialize)
    try:
        await skills.store_user_skill(
            user, name=name, description="Recover", body="Old body"
        )
        session = ChatSession.new(user, dry_run=False)
        head = await versions.get_head(user, "personal", name)
        content = render_skill_markdown(
            ParsedSkill(name=name, description="Recover", body="Restored body")
        )
        committed = await publication.commit_version_safe(
            user,
            head=head,
            draft=publication.VersionDraft(
                content=content, files=[], description="Recover", origin="restored"
            ),
            expected_current_version=head.current_version,
        )
        version = committed.version
        assert (
            await publication.get_pending_publication_for_skill(
                "foreign", "personal", name
            )
            is None
        )
        assert (
            await publication.get_pending_publication_for_skill(
                user, "another-scope", name
            )
            is None
        )
        with monkeypatch.context() as failure:
            failure.setattr(
                publish,
                "store_user_skill",
                AsyncMock(side_effect=OSError("storage down")),
            )
            assert (
                await publish.write_committed_version(user, version, None)
            ).status == "write_failed"
            unreadable = await skills.ReadSkillTool()._execute(user, session, name=name)
            assert isinstance(unreadable, ErrorResponse)
            assert (
                await versions.get_version(user, version.id)
            ).state == "pending_write"
        await versions.update_head_policy(user, "personal", name, use_paused=True)
        assert (
            await publication.get_pending_publication_for_skill(user, "personal", name)
            is None
        )
        paused = await skills.ReadSkillTool()._execute(user, session, name=name)
        assert isinstance(paused, ErrorResponse) and paused.error == "skill_use_paused"
        assert (await versions.get_version(user, version.id)).state == "pending_write"
        await versions.update_head_policy(user, "personal", name, use_paused=False)
        loaded = await skills.ReadSkillTool()._execute(user, session, name=name)
        assert isinstance(loaded, ReadSkillResponse), loaded
        assert loaded.version_id == version.id, loaded
        assert loaded.body.strip() == "Restored body"
        assert (await versions.get_version(user, version.id)).state == "ready"
        flag.assert_not_awaited()
    finally:
        await _cleanup(user)
