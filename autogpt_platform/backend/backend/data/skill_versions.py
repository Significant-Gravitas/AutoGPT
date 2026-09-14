"""Data layer for versioned learned skills.

The workspace ``SKILL.md`` stays the content surface the copilot reads; this
module keeps the immutable version history, the compare-and-swap head that
publication races against, per-skill learning policy, suppression records,
and load/outcome events attached to exact versions.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import prisma.errors
import prisma.models
import prisma.types
from pydantic import BaseModel, Field

from backend.data.skill_learning import owner_key_for
from backend.util.json import SafeJson

ACTIVE_VERSION_STATES = frozenset({"ready"})
OPEN_DECISION_STATE = "needs_decision"
PENDING_WRITE_STATE = "pending_write"


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


class SkillHeadRecord(BaseModel):
    id: str
    user_id: str
    expert_id: str | None
    owner_key: str
    skill_name: str
    current_version: int
    current_version_id: str | None
    content_hash: str | None
    auto_improve: bool
    learning_paused_at: datetime | None
    use_paused_at: datetime | None
    updated_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.SkillHead) -> "SkillHeadRecord":
        return cls(
            id=row.id,
            user_id=row.userId,
            expert_id=row.expertId,
            owner_key=row.ownerKey,
            skill_name=row.skillName,
            current_version=row.currentVersion,
            current_version_id=row.currentVersionId,
            content_hash=row.contentHash,
            auto_improve=row.autoImprove,
            learning_paused_at=row.learningPausedAt,
            use_paused_at=row.usePausedAt,
            updated_at=row.updatedAt,
        )


class SkillVersionRecord(BaseModel):
    id: str
    user_id: str
    expert_id: str | None
    owner_key: str
    skill_name: str
    head_id: str
    version: int
    content: str
    content_hash: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    origin: str
    actor_user_id: str | None
    summary: str
    base_version_id: str | None
    restored_from_version_id: str | None
    review_id: str | None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    state: str
    state_reason: str
    blocked_pattern_class: str | None
    blocked_step: str | None
    created_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.SkillVersion) -> "SkillVersionRecord":
        return cls(
            id=row.id,
            user_id=row.userId,
            expert_id=row.expertId,
            owner_key=row.ownerKey,
            skill_name=row.skillName,
            head_id=row.headId,
            version=row.version,
            content=row.content,
            content_hash=row.contentHash,
            description=row.description,
            triggers=list(row.triggers or []),
            origin=row.origin,
            actor_user_id=row.actorUserId,
            summary=row.summary,
            base_version_id=row.baseVersionId,
            restored_from_version_id=row.restoredFromVersionId,
            review_id=row.reviewId,
            sources=_json_list(row.sources),
            evidence=_json_list(row.evidence),
            limits=list(row.limits or []),
            state=row.state,
            state_reason=row.stateReason,
            blocked_pattern_class=row.blockedPatternClass,
            blocked_step=row.blockedStep,
            created_at=row.createdAt,
        )


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


async def get_head(
    user_id: str, owner_key: str, skill_name: str
) -> SkillHeadRecord | None:
    row = await prisma.models.SkillHead.prisma().find_unique(
        where={
            "userId_ownerKey_skillName": {
                "userId": user_id,
                "ownerKey": owner_key,
                "skillName": skill_name,
            }
        }
    )
    return SkillHeadRecord.from_db(row) if row else None


async def ensure_head(
    user_id: str, expert_id: str | None, skill_name: str
) -> SkillHeadRecord:
    owner_key = owner_key_for(expert_id)
    existing = await get_head(user_id, owner_key, skill_name)
    if existing is not None:
        return existing
    try:
        row = await prisma.models.SkillHead.prisma().create(
            data={
                "userId": user_id,
                "expertId": expert_id,
                "ownerKey": owner_key,
                "skillName": skill_name,
            }
        )
        return SkillHeadRecord.from_db(row)
    except prisma.errors.UniqueViolationError:
        head = await get_head(user_id, owner_key, skill_name)
        if head is None:
            raise
        return head


async def list_heads(
    user_id: str, owner_key: str | None = None
) -> list[SkillHeadRecord]:
    where: prisma.types.SkillHeadWhereInput = {"userId": user_id}
    if owner_key is not None:
        where["ownerKey"] = owner_key
    rows = await prisma.models.SkillHead.prisma().find_many(where=where)
    return [SkillHeadRecord.from_db(row) for row in rows]


async def list_use_paused_skill_names(user_id: str, owner_key: str) -> list[str]:
    rows = await prisma.models.SkillHead.prisma().find_many(
        where={"userId": user_id, "ownerKey": owner_key, "NOT": [{"usePausedAt": None}]}
    )
    return [row.skillName for row in rows]


async def update_head_policy(
    user_id: str,
    owner_key: str,
    skill_name: str,
    *,
    auto_improve: bool | None = None,
    learning_paused: bool | None = None,
    use_paused: bool | None = None,
) -> SkillHeadRecord | None:
    head = await get_head(user_id, owner_key, skill_name)
    if head is None:
        return None
    now = datetime.now(timezone.utc)
    data: prisma.types.SkillHeadUpdateInput = {}
    if auto_improve is not None:
        data["autoImprove"] = auto_improve
    if learning_paused is not None:
        data["learningPausedAt"] = now if learning_paused else None
    if use_paused is not None:
        data["usePausedAt"] = now if use_paused else None
    if not data:
        return head
    row = await prisma.models.SkillHead.prisma().update(
        where={"id": head.id}, data=data
    )
    return SkillHeadRecord.from_db(row) if row else head


async def publish_version_if_current(
    user_id: str,
    head_id: str,
    *,
    expected_current_version: int,
    version_id: str,
    version_number: int,
    new_content_hash: str,
) -> bool:
    """Compare-and-swap the head pointer. False means a concurrent change won."""
    updated = await prisma.models.SkillHead.prisma().update_many(
        where={
            "id": head_id,
            "userId": user_id,
            "currentVersion": expected_current_version,
        },
        data={
            "currentVersion": version_number,
            "currentVersionId": version_id,
            "contentHash": new_content_hash,
        },
    )
    return updated > 0


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


async def next_version_number(head_id: str) -> int:
    latest = await prisma.models.SkillVersion.prisma().find_first(
        where={"headId": head_id}, order={"version": "desc"}
    )
    return (latest.version if latest else 0) + 1


async def create_version(
    user_id: str,
    *,
    head: SkillHeadRecord,
    content: str,
    description: str,
    triggers: list[str],
    origin: str,
    summary: str = "",
    state: str = "ready",
    state_reason: str = "",
    actor_user_id: str | None = None,
    base_version_id: str | None = None,
    restored_from_version_id: str | None = None,
    review_id: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    limits: list[str] | None = None,
    blocked_pattern_class: str | None = None,
    blocked_step: str | None = None,
) -> SkillVersionRecord:
    """Append an immutable version row (numbers are monotonic per head)."""
    for _ in range(3):
        number = await next_version_number(head.id)
        try:
            row = await prisma.models.SkillVersion.prisma().create(
                data={
                    "userId": user_id,
                    "expertId": head.expert_id,
                    "ownerKey": head.owner_key,
                    "skillName": head.skill_name,
                    "headId": head.id,
                    "version": number,
                    "content": content,
                    "contentHash": content_hash(content),
                    "description": description,
                    "triggers": list(triggers),
                    "origin": origin,
                    "actorUserId": actor_user_id,
                    "summary": summary[:1000],
                    "baseVersionId": base_version_id,
                    "restoredFromVersionId": restored_from_version_id,
                    "reviewId": review_id,
                    "sources": SafeJson(sources or []),
                    "evidence": SafeJson(evidence or []),
                    "limits": list(limits or []),
                    "state": state,
                    "stateReason": state_reason[:1000],
                    "blockedPatternClass": blocked_pattern_class,
                    "blockedStep": blocked_step,
                }
            )
            return SkillVersionRecord.from_db(row)
        except prisma.errors.UniqueViolationError:
            continue
    raise RuntimeError("could not allocate a skill version number")


async def set_version_state(
    user_id: str, version_id: str, state: str, reason: str = ""
) -> None:
    await prisma.models.SkillVersion.prisma().update_many(
        where={"id": version_id, "userId": user_id},
        data={"state": state, "stateReason": reason[:1000]},
    )


async def get_version(user_id: str, version_id: str) -> SkillVersionRecord | None:
    row = await prisma.models.SkillVersion.prisma().find_first(
        where={"id": version_id, "userId": user_id}
    )
    return SkillVersionRecord.from_db(row) if row else None


async def list_versions(
    user_id: str, owner_key: str, skill_name: str, *, limit: int = 50
) -> list[SkillVersionRecord]:
    rows = await prisma.models.SkillVersion.prisma().find_many(
        where={"userId": user_id, "ownerKey": owner_key, "skillName": skill_name},
        order={"version": "desc"},
        take=limit,
    )
    return [SkillVersionRecord.from_db(row) for row in rows]


async def list_recent_versions(
    user_id: str,
    *,
    owner_key: str | None = None,
    origin: str | None = None,
    state: str | None = None,
    limit: int = 50,
) -> list[SkillVersionRecord]:
    where: prisma.types.SkillVersionWhereInput = {"userId": user_id}
    if owner_key is not None:
        where["ownerKey"] = owner_key
    if origin is not None:
        where["origin"] = origin
    if state is not None:
        where["state"] = state
    rows = await prisma.models.SkillVersion.prisma().find_many(
        where=where, order={"createdAt": "desc"}, take=limit
    )
    return [SkillVersionRecord.from_db(row) for row in rows]


async def list_open_decisions(
    user_id: str, owner_key: str | None = None
) -> list[SkillVersionRecord]:
    return await list_recent_versions(
        user_id, owner_key=owner_key, state=OPEN_DECISION_STATE, limit=100
    )


async def list_versions_in_states(
    user_id: str, states: list[str], *, limit: int = 500
) -> list[SkillVersionRecord]:
    rows = await prisma.models.SkillVersion.prisma().find_many(
        where={"userId": user_id, "state": {"in": states}},
        order={"createdAt": "desc"},
        take=limit,
    )
    return [SkillVersionRecord.from_db(row) for row in rows]
