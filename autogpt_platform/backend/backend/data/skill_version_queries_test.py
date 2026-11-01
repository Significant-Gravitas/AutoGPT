from uuid import uuid4

import pytest

from backend.data import skill_versions as versions
from backend.data.skill_learning_test import _cleanup, _create_user
from backend.data.skill_version_files import SkillVersionFile


async def _version(user_id, *, skill_name="projection-check", state="ready"):
    head = await versions.ensure_head(user_id, None, skill_name)
    return await versions.create_version(
        user_id,
        head=head,
        content="Run scripts/check.py",
        description="Check package history",
        triggers=[],
        origin="saved_overnight",
        state=state,
        files=[SkillVersionFile.from_content("scripts/check.py", b"print('verified')")],
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_history_queries_omit_snapshots_but_exact_lookup_preserves_them(server):
    user, foreign = str(uuid4()), str(uuid4())
    await _create_user(user)
    await _create_user(foreign)
    try:
        first = await _version(user)
        decision = await _version(user, state="needs_decision")
        pending = await _version(user, state="pending_write")
        await _version(user, skill_name="other-skill")
        await _version(foreign)
        history = await versions.list_versions(
            user, "personal", "projection-check", limit=2
        )
        assert [v.id for v in history] == [pending.id, decision.id]
        assert (
            await versions.list_versions(user, "another-owner", "projection-check")
            == []
        )
        recent = await versions.list_recent_versions(
            user, owner_key="personal", origin="saved_overnight", state="ready"
        )
        assert len(recent) == 2 and all(v.user_id == user for v in recent)
        decisions = await versions.list_open_decisions(user)
        assert [v.id for v in decisions] == [decision.id]
        states = await versions.list_versions_in_states(user, ["pending_write"])
        assert [v.id for v in states] == [pending.id]
        assert all(v.files is None for v in [*history, *recent, *decisions, *states])
        assert (await versions.get_version(user, first.id)).files == first.files
        assert await versions.get_version(foreign, first.id) is None
        assert await versions.list_versions_in_states(user, []) == []
    finally:
        await _cleanup(user, foreign)
