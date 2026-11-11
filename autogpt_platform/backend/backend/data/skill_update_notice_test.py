"""A cold turn that brings a marketplace copy up to date tells the chat once,
on real storage: the copy's new version lands in the database before the
turn reads the skill heads, so the version it marks seen in Redis is the new
one and the next turn has nothing to repeat. Only the marketplace catalog is
a stand-in."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from backend.copilot.tools import skills
from backend.copilot.tools.skill_updates_test import _active, _baseline, _package
from backend.data import skill_versions as versions
from backend.data.skill_learning_test import _cleanup, _create_user


@pytest.mark.asyncio(loop_scope="session")
async def test_a_cold_turn_that_updates_a_copy_announces_the_update_once(
    server, monkeypatch
):
    user = str(uuid4())
    chat = str(uuid4())
    await _create_user(user)
    monkeypatch.setattr("backend.util.workspace.scan_content_safe", AsyncMock())
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )
    marketplace = MagicMock(
        get_active_versions=AsyncMock(return_value={}),
        get_version_packages=AsyncMock(return_value={}),
        find_version_by_hash=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(skills, "skill_db", lambda: marketplace)
    v1 = _package("cold-email", "v1", "# Cold\n")
    v2 = _package("cold-email", "v2", "# Cold, revised\n")
    history: list[str] = []

    async def turn() -> str:
        return await skills.build_skills_update_notice(
            user, prior_contents=history, session_id=chat
        )

    try:
        await skills.store_user_skill(
            user,
            name=v1.slug,
            description="",
            body="",
            files=list(v1.files),
            origin=skills.SKILL_ORIGIN_MARKETPLACE,
            skill_markdown=v1.skill_markdown,
            baseline=_baseline(v1),
        )
        index = await skills.build_skills_context(user)
        history.append(f"<available_skills>\n{index}\n</available_skills>\n\nhi")
        # Takes the install's own queued name and records v1 as seen.
        await turn()
        installed = await versions.get_head(user, "personal", v1.slug)

        marketplace.get_active_versions = AsyncMock(return_value={v2.slug: _active(v2)})
        marketplace.get_version_packages = AsyncMock(return_value={v2.version_id: v2})
        # The index cache has expired, so the next turn reads cold.
        await skills.invalidate_skills_index_cache(user)
        updating = await turn()
        updated = await versions.get_head(user, "personal", v1.slug)
        after = await turn()
    finally:
        await _cleanup(user)

    assert installed is not None and updated is not None
    assert updated.current_version_id != installed.current_version_id
    assert "Updated skills: cold-email." in updating
    assert after == ""
