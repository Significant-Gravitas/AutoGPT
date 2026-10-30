"""Revocation: make every version that depends on a withdrawn source
unavailable, then restore an independent version or pause the skill."""

from __future__ import annotations

import logging

from backend.copilot.tools.skills import invalidate_skills_index_cache
from backend.data.db_accessors import skill_versions_db
from backend.data.skill_learning import LearningSourceRecord

from .owner_actions import restore_version

logger = logging.getLogger(__name__)


async def invalidate_versions_for_source(
    user_id: str, source: LearningSourceRecord, reason: str
) -> list[str]:
    """Make every version depending on ``source`` (and descendants)
    unavailable; restore a prior eligible version or pause use."""
    versions = skill_versions_db()
    affected = await versions.invalidate_source_versions(user_id, source.id, reason)
    affected_ids = {v.id for v in affected}
    heads_touched = {(v.owner_key, v.skill_name, v.expert_id) for v in affected}
    for owner_key, skill_name, expert_id in heads_touched:
        head = await versions.get_head(user_id, owner_key, skill_name)
        replaced = head.current_version_id if head is not None else None
        if head is None or replaced is None or replaced not in affected_ids:
            continue
        fallback = await versions.latest_independent_version_id(user_id, head.id)
        if fallback is None:
            # Conditional: a deletion (or a newer version) landing since the
            # head was read must not inherit this pause.
            await versions.pause_use_if_current(user_id, head.id, replaced)
            await invalidate_skills_index_cache(user_id, expert_id)
            continue
        await restore_version(
            user_id=user_id,
            expert_id=expert_id,
            skill_name=skill_name,
            version_id=fallback,
            actor_user_id=user_id,
            reason="Restored after a source became unavailable",
            suppress_behavior=False,
            replacing_version_id=replaced,
        )
        await invalidate_skills_index_cache(user_id, expert_id)
    return [v.id for v in affected]
