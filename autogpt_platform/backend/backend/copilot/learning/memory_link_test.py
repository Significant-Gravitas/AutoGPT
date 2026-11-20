"""A learned skill version's reference episode goes into its own memory scope:
the account's for an account skill, the expert's for an expert's, through
``enqueue_episode`` as it is declared (the episode's scope its first
argument)."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.skill_versions import SkillVersionRecord

from . import memory_link


def _version(expert_id: str | None) -> SkillVersionRecord:
    return SkillVersionRecord(
        id="version-1",
        user_id="user-1",
        expert_id=expert_id,
        owner_key="owner",
        skill_name="import-csv",
        head_id="head-1",
        version=2,
        content="# Import a CSV",
        content_hash="hash",
        description="Import a CSV",
        origin="learned",
        actor_user_id=None,
        summary="",
        base_version_id=None,
        restored_from_version_id=None,
        review_id=None,
        state="published",
        state_reason="",
        blocked_pattern_class=None,
        blocked_step=None,
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_id", [None, "expert-1"], ids=["account", "expert"])
async def test_a_learned_version_is_linked_in_its_own_scope(expert_id):
    with (patch.object(memory_link, "enqueue_episode", autospec=True) as enqueue,):
        enqueue.return_value = True
        linked = await memory_link.link_version_to_memory("user-1", _version(expert_id))

    assert linked is True
    assert enqueue.await_args.args[0] == MemoryScope.build("user-1", expert_id)
    assert enqueue.await_args.args[1] == "skill_learning_version-1"
    assert enqueue.await_args.kwargs["name"] == "dream_skill_version-1"
