"""Version history for writes that come through the ordinary skill registry.

Every write to a skill appends an immutable version: the copilot's own
``store_skill`` during work, a REST upload or an expert copy (imported), or
an owner edit. Human-controlled writes turn automatic improvements off for
that skill unless the editor explicitly keeps them on, so an overnight
review can never silently replace a version a person authored.
"""

from __future__ import annotations

import logging

from backend.data.db_accessors import skill_publication_db, skill_versions_db
from backend.data.skill_learning import owner_key_for
from backend.data.skill_publication import VersionDraft
from backend.data.skill_versions import (
    SkillHeadRecord,
    SkillVersionRecord,
    content_hash,
)

logger = logging.getLogger(__name__)

HUMAN_CONTROLLED_ORIGINS = frozenset({"edited", "imported"})
AUTOMATED_ORIGINS = frozenset({"saved_overnight", "requested"})


async def record_registry_write(
    user_id: str,
    *,
    expert_id: str | None,
    skill_name: str,
    rendered: str,
    description: str,
    triggers: list[str],
    origin: str,
    actor_user_id: str | None,
    summary: str,
    keep_auto_improve: bool | None,
) -> SkillVersionRecord | None:
    """Append the version for a workspace write that already succeeded.

    Retries the head compare-and-swap a few times because a registry write
    is the truth of what the file now says; a concurrent automated commit
    that loses the race is superseded by the human write, never the other
    way round. Never raises — history must not fail the write it records.
    """
    try:
        versions = skill_versions_db()
        head = await versions.ensure_head(user_id, expert_id, skill_name)
        if head.content_hash == content_hash(rendered) and head.current_version_id:
            return None
        auto_improve = _auto_improve_after(origin, keep_auto_improve)
        for _ in range(3):
            result = await skill_publication_db().commit_version_safe(
                user_id,
                head=head,
                draft=VersionDraft(
                    content=rendered,
                    description=description,
                    triggers=triggers,
                    origin=origin,
                    summary=summary,
                    actor_user_id=actor_user_id,
                    base_version_id=head.current_version_id,
                ),
                expected_current_version=head.current_version,
                auto_improve=auto_improve,
            )
            if result.committed and result.version is not None:
                await skill_publication_db().complete_publication(
                    user_id, version_id=result.version.id, review_id=None
                )
                return result.version
            refreshed = await versions.get_head(
                user_id, owner_key_for(expert_id), skill_name
            )
            if refreshed is None:
                return None
            head = refreshed
    except Exception:
        logger.warning(
            "Skill version history not recorded for %s (user %s)",
            skill_name,
            user_id[:12],
            exc_info=True,
        )
    return None


def _auto_improve_after(origin: str, keep_auto_improve: bool | None) -> bool | None:
    if keep_auto_improve is not None:
        return keep_auto_improve
    if origin in HUMAN_CONTROLLED_ORIGINS:
        return False
    return None


async def head_for(
    user_id: str, expert_id: str | None, skill_name: str
) -> SkillHeadRecord | None:
    return await skill_versions_db().get_head(
        user_id, owner_key_for(expert_id), skill_name
    )
