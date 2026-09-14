"""Revocation: make every version that depends on a withdrawn source
unavailable, then restore an independent version or pause the skill."""

from __future__ import annotations

import logging

from backend.copilot.tools.skills import invalidate_skills_index_cache
from backend.data.db_accessors import skill_versions_db
from backend.data.skill_learning import LearningSourceRecord
from backend.data.skill_versions import SkillHeadRecord, SkillVersionRecord

from .owner_actions import restore_version

logger = logging.getLogger(__name__)


async def invalidate_versions_for_source(
    user_id: str, source: LearningSourceRecord, reason: str
) -> list[str]:
    """Make every version depending on ``source`` (and descendants)
    unavailable; restore a prior eligible version or pause use."""
    versions = skill_versions_db()
    live = await versions.list_versions_in_states(
        user_id, ["ready", "needs_decision", "pending_write", "candidate"]
    )
    affected = _dependents(live, source.id)
    for version in affected:
        await versions.set_version_state(user_id, version.id, "invalidated", reason)
    heads_touched = {(v.owner_key, v.skill_name, v.expert_id) for v in affected}
    for owner_key, skill_name, expert_id in heads_touched:
        head = await versions.get_head(user_id, owner_key, skill_name)
        if head is None or head.current_version_id not in {v.id for v in affected}:
            continue
        fallback = _latest_unaffected(live, affected, head)
        if fallback is None:
            await versions.update_head_policy(
                user_id, owner_key, skill_name, use_paused=True
            )
            await invalidate_skills_index_cache(user_id, expert_id)
            continue
        await restore_version(
            user_id=user_id,
            expert_id=expert_id,
            skill_name=skill_name,
            version_id=fallback.id,
            actor_user_id=user_id,
            reason="Restored after a source became unavailable",
        )
    return [v.id for v in affected]


def _dependents(
    live: list[SkillVersionRecord], source_id: str
) -> list[SkillVersionRecord]:
    by_id = {v.id: v for v in live}
    affected: dict[str, SkillVersionRecord] = {
        v.id: v for v in live if any(s.get("source_id") == source_id for s in v.sources)
    }
    changed = True
    while changed:
        changed = False
        for version in live:
            if version.id in affected:
                continue
            parent = version.base_version_id or version.restored_from_version_id
            if parent in affected and parent in by_id:
                affected[version.id] = version
                changed = True
    return list(affected.values())


def _latest_unaffected(
    live: list[SkillVersionRecord],
    affected: list[SkillVersionRecord],
    head: SkillHeadRecord,
) -> SkillVersionRecord | None:
    bad = {v.id for v in affected}
    candidates = [
        v
        for v in live
        if v.head_id == head.id and v.id not in bad and v.state == "ready" and v.content
    ]
    return max(candidates, key=lambda v: v.version, default=None)
