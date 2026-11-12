"""Suppression records and load/outcome events for learned skills.

A suppression identifies the *behaviour* an owner removed so it is not
silently reapplied; a use event attaches a load or an explicit outcome to
the exact version used. A load is never a success claim.
"""

from __future__ import annotations

from datetime import datetime

import prisma.errors
import prisma.models
from pydantic import BaseModel, Field

from backend.data.skill_learning import owner_key_for


class SkillUseEventRecord(BaseModel):
    id: str
    skill_name: str
    version_id: str | None
    session_id: str | None
    kind: str
    detail: str
    actor_user_id: str | None
    created_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.SkillUseEvent) -> "SkillUseEventRecord":
        return cls(
            id=row.id,
            skill_name=row.skillName,
            version_id=row.versionId,
            session_id=row.sessionId,
            kind=row.kind,
            detail=row.detail,
            actor_user_id=row.actorUserId,
            created_at=row.createdAt,
        )


class SkillSuppressionRecord(BaseModel):
    id: str
    skill_name: str
    behavior_fingerprint: str
    behavior_tokens: list[str] = Field(default_factory=list)
    evidence_fingerprints: list[str] = Field(default_factory=list)
    reason: str
    created_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.SkillSuppression) -> "SkillSuppressionRecord":
        return cls(
            id=row.id,
            skill_name=row.skillName,
            behavior_fingerprint=row.behaviorFingerprint,
            behavior_tokens=list(row.behaviorTokens or []),
            evidence_fingerprints=list(row.evidenceFingerprints or []),
            reason=row.reason,
            created_at=row.createdAt,
        )


async def add_suppression(
    user_id: str,
    *,
    expert_id: str | None,
    skill_name: str,
    behavior_fingerprint: str,
    behavior_tokens: list[str],
    evidence_fingerprints: list[str],
    actor_user_id: str | None,
    reason: str,
) -> SkillSuppressionRecord:
    owner_key = owner_key_for(expert_id)
    try:
        row = await prisma.models.SkillSuppression.prisma().create(
            data={
                "userId": user_id,
                "expertId": expert_id,
                "ownerKey": owner_key,
                "skillName": skill_name,
                "behaviorFingerprint": behavior_fingerprint,
                "behaviorTokens": list(behavior_tokens),
                "evidenceFingerprints": list(evidence_fingerprints),
                "actorUserId": actor_user_id,
                "reason": reason[:1000],
            }
        )
        return SkillSuppressionRecord.from_db(row)
    except prisma.errors.UniqueViolationError:
        existing = await find_suppression(
            user_id, owner_key, skill_name, behavior_fingerprint
        )
        if existing is None:
            raise
        return existing


async def find_suppression(
    user_id: str, owner_key: str, skill_name: str, behavior_fingerprint: str
) -> SkillSuppressionRecord | None:
    row = await prisma.models.SkillSuppression.prisma().find_first(
        where={
            "userId": user_id,
            "ownerKey": owner_key,
            "skillName": skill_name,
            "behaviorFingerprint": behavior_fingerprint,
        }
    )
    return SkillSuppressionRecord.from_db(row) if row else None


async def list_suppressions(
    user_id: str, owner_key: str, skill_name: str
) -> list[SkillSuppressionRecord]:
    rows = await prisma.models.SkillSuppression.prisma().find_many(
        where={"userId": user_id, "ownerKey": owner_key, "skillName": skill_name}
    )
    return [SkillSuppressionRecord.from_db(row) for row in rows]


async def record_use_event(
    user_id: str,
    *,
    expert_id: str | None,
    skill_name: str,
    kind: str,
    version_id: str | None = None,
    session_id: str | None = None,
    detail: str = "",
    actor_user_id: str | None = None,
) -> SkillUseEventRecord:
    row = await prisma.models.SkillUseEvent.prisma().create(
        data={
            "userId": user_id,
            "expertId": expert_id,
            "ownerKey": owner_key_for(expert_id),
            "skillName": skill_name,
            "versionId": version_id,
            "sessionId": session_id,
            "kind": kind,
            "detail": detail[:1000],
            "actorUserId": actor_user_id,
        }
    )
    return SkillUseEventRecord.from_db(row)


async def list_use_events(
    user_id: str, owner_key: str, skill_name: str, *, limit: int = 200
) -> list[SkillUseEventRecord]:
    rows = await prisma.models.SkillUseEvent.prisma().find_many(
        where={"userId": user_id, "ownerKey": owner_key, "skillName": skill_name},
        order={"createdAt": "desc"},
        take=limit,
    )
    return [SkillUseEventRecord.from_db(row) for row in rows]
