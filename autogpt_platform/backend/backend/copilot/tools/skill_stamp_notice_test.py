from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skill_updates_test import USER, _marketplace, _package
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path


@pytest.fixture
def update_markers(monkeypatch):
    updated: set[str] = set()

    async def remember(_user_id, _expert_id, names):
        updated.update(names)

    async def consume(_user_id, _expert_id):
        names = sorted(updated)
        updated.clear()
        return names

    monkeypatch.setattr(skills, "_remember_updated", remember)
    monkeypatch.setattr(skills, "_consume_updated", consume)
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tracked", [False, True])
async def test_baseline_stamp_is_silent_but_later_body_update_is_reported(
    update_markers, tracked
):
    manager = _FakeWorkspaceManager()
    v1 = _package("checks", "v1", "Run the checks\n")
    v2 = _package("checks", "v2", "Run the checks and verify the output\n")
    with _patch_skills_path(manager) as patched:
        await skills.store_user_skill(
            USER,
            name="checks",
            description="",
            body="",
            skill_markdown=v1.skill_markdown,
            files=[],
            origin=skills.SKILL_ORIGIN_MARKETPLACE,
            version_origin="imported" if tracked else None,
        )
        index = await skills.skills_context_block(USER)
        assert index is not None
        history = [index.text]
        await skills.build_skills_update_notice(
            USER, prior_contents=history, session_id="chat-1"
        )

        _marketplace(patched, v1)
        notice = await skills.build_skills_update_notice(
            USER, prior_contents=history, session_id="chat-1"
        )
        assert (
            manager.metadata["/skills/checks/SKILL.md"]["listing_version_id"]
            == v1.version_id
        )
        assert manager.files["/skills/checks/SKILL.md"] == v1.skill_markdown.encode()
        assert notice == ""

        _marketplace(patched, v1, v2)
        notice = await skills.build_skills_update_notice(
            USER, prior_contents=history, session_id="chat-1"
        )
        assert "Updated skills: checks" in notice
        assert manager.files["/skills/checks/SKILL.md"] == v2.skill_markdown.encode()
        assert (
            await skills.build_skills_update_notice(
                USER, prior_contents=history, session_id="chat-1"
            )
            == ""
        )
