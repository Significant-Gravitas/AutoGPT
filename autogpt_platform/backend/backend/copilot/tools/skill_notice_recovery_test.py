import json
from unittest.mock import AsyncMock

import pytest

from backend.copilot.tools import skills


@pytest.fixture
def notice_registry(monkeypatch):
    skill = skills.ParsedSkill(
        name="private-checks", description="Checks", body="Run checks"
    )
    monkeypatch.setattr(skills, "list_user_skills", AsyncMock(return_value=[skill]))
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )
    consume = AsyncMock(return_value=[])
    mark = AsyncMock(wraps=skills.mark_index_revision_seen)
    monkeypatch.setattr(skills, "_consume_updated", consume)
    monkeypatch.setattr(skills, "mark_index_revision_seen", mark)
    return consume, mark


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_id", [None, "expert-1"])
async def test_partial_registry_outage_preserves_unseen_updates(
    notice_registry, fake_learning_store, monkeypatch, expert_id
):
    head = await fake_learning_store.ensure_head("user-1", expert_id, "private-checks")
    head.current_version_id = "v1"
    index = await skills.skills_context_block("user-1", expert_id)
    assert index is not None
    history = [index.text]
    await skills.build_skills_update_notice(
        "user-1", expert_id, history, session_id="session-1"
    )
    before = await skills.read_seen_index_revision("session-1")
    assert before is not None and json.loads(before) == {"private-checks": "v1"}
    consume, mark = notice_registry
    consume.reset_mock()
    mark.reset_mock()
    head.current_version_id = "v2"

    with monkeypatch.context() as outage:
        outage.setattr(
            fake_learning_store,
            "list_use_paused_skill_names",
            AsyncMock(side_effect=OSError("registry unavailable")),
        )
        assert (
            await skills.build_skills_update_notice(
                "user-1", expert_id, history, session_id="session-1"
            )
            == ""
        )
        assert await skills.read_seen_index_revision("session-1") == before
        consume.assert_not_awaited()
        mark.assert_not_awaited()
        visible = {
            skill.name for skill in await skills.list_all_skills("user-1", expert_id)
        }
        assert "private-checks" not in visible
        assert "agent_building_guide" in visible

    recovered = await skills.build_skills_update_notice(
        "user-1", expert_id, history, session_id="session-1"
    )
    assert "Updated skills: private-checks" in recovered
    assert "Removed skills" not in recovered
    mark.assert_awaited_once()
    revision = await skills.read_seen_index_revision("session-1")
    assert revision is not None and json.loads(revision) == {"private-checks": "v2"}
    assert (
        await skills.build_skills_update_notice(
            "user-1", expert_id, history, session_id="session-1"
        )
        == ""
    )
