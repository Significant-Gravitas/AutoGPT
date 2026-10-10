"""Retrieval-side invariants and freshness for learned skills.

Paused and archived versions are never returned as active procedures, a
load is recorded as a load (never a success), and an existing conversation
sees an overnight update at its next user turn without a new chat.
"""

from __future__ import annotations

import hashlib
import logging

from pydantic import BaseModel, ConfigDict

from backend.data.db_accessors import skill_use_db, skill_versions_db
from backend.data.redis_client import get_redis_async
from backend.data.skill_learning import owner_key_for
from backend.data.skill_versions import SkillHeadRecord, SkillVersionRecord

logger = logging.getLogger(__name__)

SKILLS_INDEX_SEEN_KEY = "copilot:skills_index_seen:{session_id}"
SKILLS_INDEX_SEEN_TTL_S = 30 * 24 * 3600

ORIGIN_LABELS = {
    "saved_overnight": "Saved overnight",
    "saved_during_work": "Saved during work",
    "requested": "Requested by user",
    "edited": "Edited by owner",
    "restored": "Restored by owner",
    "imported": "Imported",
}


def origin_label(origin: str, *, viewer_is_actor: bool = False) -> str:
    label = ORIGIN_LABELS.get(origin, origin.replace("_", " ").capitalize())
    if viewer_is_actor and origin in ("requested", "edited", "restored"):
        return label.replace("by user", "by you").replace("by owner", "by you")
    return label


class IndexEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str | None


async def paused_skill_names(user_id: str, expert_id: str | None) -> set[str] | None:
    """Skills the owner asked to stop using.

    Returns ``None`` when the registry cannot be read. Callers must treat
    that as *unknown* and fail closed for the owner's skills: unknown
    eligibility never makes paused or revoked content usable. Built-in
    default skills and every ordinary tool stay available regardless.
    """
    try:
        names = await skill_versions_db().list_use_paused_skill_names(
            user_id, owner_key_for(expert_id)
        )
        return set(names)
    except Exception:
        logger.warning(
            "Could not read paused skills for user %s — hiding owner skills "
            "until the registry answers",
            user_id[:12],
            exc_info=True,
        )
        return None


async def resolve_current_version(
    user_id: str, expert_id: str | None, skill_name: str
) -> tuple[SkillHeadRecord, SkillVersionRecord | None] | None:
    """Head + current version record for a learned skill, or ``None``."""
    head = await skill_versions_db().get_head(
        user_id, owner_key_for(expert_id), skill_name
    )
    if head is None:
        return None
    version = None
    if head.current_version_id:
        version = await skill_versions_db().get_version(
            user_id, head.current_version_id
        )
    return head, version


async def record_skill_loaded(
    user_id: str,
    expert_id: str | None,
    skill_name: str,
    *,
    version_id: str | None,
    session_id: str | None,
) -> None:
    """A load event only. Never raises; never counts as success."""
    try:
        await skill_use_db().record_use_event(
            user_id,
            expert_id=expert_id,
            skill_name=skill_name,
            kind="loaded",
            version_id=version_id,
            session_id=session_id,
        )
    except Exception:
        logger.debug("skill load event not recorded", exc_info=True)


def index_revision(entries: list[IndexEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda e: e.name):
        digest.update(f"{entry.name}@{entry.version or ''}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


async def read_seen_index_revision(session_id: str) -> str | None:
    try:
        redis = await get_redis_async()
        raw = await redis.get(SKILLS_INDEX_SEEN_KEY.format(session_id=session_id))
    except Exception:
        return None
    if raw is None:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


async def mark_index_revision_seen(session_id: str, revision: str) -> None:
    try:
        redis = await get_redis_async()
        await redis.set(
            SKILLS_INDEX_SEEN_KEY.format(session_id=session_id),
            revision,
            ex=SKILLS_INDEX_SEEN_TTL_S,
        )
    except Exception:
        logger.debug("skills index seen marker not written", exc_info=True)
