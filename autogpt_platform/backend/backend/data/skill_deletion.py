"""Record an owner's deletion of a skill in its version history.

Deleting a skill removes its workspace package, but the head would still
name the last version and the older versions would stay ``ready``:
excluding an old source would then restore one of them, and a learned
``create`` for the name would conflict with the head for good.
:func:`record_skill_deletion` ends the history in one transaction instead.
It suppresses the deleted behaviour, appends a ``deleted`` marker version,
clears the head's pointer and per-skill policy, settles any publication
still waiting for its workspace write, and archives what is left.
"""

from __future__ import annotations

import prisma.errors
import prisma.models
from prisma import Prisma
from pydantic import BaseModel, Field

from backend.data.db import query_raw_with_schema, transaction
from backend.data.skill_content_approvals import content_approvals_json
from backend.data.skill_learning import owner_key_for
from backend.data.skill_publication_settlement import settle_publication_in
from backend.data.skill_version_files import version_files_json
from backend.data.skill_versions import (
    PENDING_WRITE_STATE,
    SkillVersionRecord,
    content_hash,
    next_version_number,
)
from backend.util.json import SafeJson

DELETED_STATE = "deleted"
DELETED_ORIGIN = "deleted"
DELETED_REASON = "the skill was deleted"


class DeletedBehavior(BaseModel):
    """One deleted behaviour, as its suppression record will hold it."""

    behavior_fingerprint: str
    behavior_tokens: list[str] = Field(default_factory=list)
    evidence_fingerprints: list[str] = Field(default_factory=list)


class SkillDeletion(BaseModel):
    """``committed=False`` means the head moved after the caller read it, so
    nothing was recorded; read it again and retry."""

    committed: bool
    marker: SkillVersionRecord | None = None
    archived_version_ids: list[str] = Field(default_factory=list)
    abandoned_version_ids: list[str] = Field(default_factory=list)


class _HeadChanged(Exception):
    """Internal: rolls the deletion back when the head moved."""


async def record_skill_deletion(
    user_id: str,
    *,
    expert_id: str | None,
    skill_name: str,
    expected_current_version_id: str | None,
    behaviors: list[DeletedBehavior],
    actor_user_id: str | None,
) -> SkillDeletion:
    """End *skill_name*'s history and suppress *behaviors*, or change nothing.

    Locks are taken in the order a publication takes them, the new version
    row and then the head, so the two never deadlock. Whichever moves the
    head first wins. The loser finds the head changed, or its version number
    taken, and rolls back: a publication reports a conflict, and a deletion
    returns ``committed=False``. A skill with no head gets one, so every
    deletion leaves a marker. The next version under the name starts a new
    chain: the head names no version, and the marker links to none, so
    revoking a source of the deleted history can never reach it.
    """
    try:
        async with transaction() as tx:
            return await _record(
                tx,
                user_id,
                expert_id=expert_id,
                skill_name=skill_name,
                expected_current_version_id=expected_current_version_id,
                behaviors=behaviors,
                actor_user_id=actor_user_id,
            )
    except (_HeadChanged, prisma.errors.UniqueViolationError):
        return SkillDeletion(committed=False)


async def _record(
    tx: Prisma,
    user_id: str,
    *,
    expert_id: str | None,
    skill_name: str,
    expected_current_version_id: str | None,
    behaviors: list[DeletedBehavior],
    actor_user_id: str | None,
) -> SkillDeletion:
    owner_key = owner_key_for(expert_id)
    head = await tx.skillhead.find_unique(
        where={
            "userId_ownerKey_skillName": {
                "userId": user_id,
                "ownerKey": owner_key,
                "skillName": skill_name,
            }
        }
    )
    current_id = head.currentVersionId if head is not None else None
    if current_id != expected_current_version_id:
        raise _HeadChanged()
    if behaviors:
        await _suppress(tx, user_id, expert_id, skill_name, behaviors, actor_user_id)
    if head is None:
        # A skill saved before version history has no head. It gets one, so
        # the deletion is on record and a learning run that read the package
        # earlier cannot import it as an existing skill afterwards.
        head = await tx.skillhead.create(
            data={
                "userId": user_id,
                "expertId": expert_id,
                "ownerKey": owner_key,
                "skillName": skill_name,
            }
        )
    marker = await _append_marker(tx, head, actor_user_id)
    ended = await tx.skillhead.update_many(
        where={
            "id": head.id,
            "currentVersion": head.currentVersion,
            "currentVersionId": current_id,
        },
        data={
            "currentVersion": marker.version,
            "currentVersionId": None,
            "contentHash": None,
            "autoImprove": True,
            "learningPausedAt": None,
            "usePausedAt": None,
        },
    )
    if not ended:
        raise _HeadChanged()
    abandoned = await _settle_pending(tx, user_id, head.id)
    # A ``ready`` version is what a source revocation would restore, and an
    # open proposal would edit a skill that is gone. Archived after the
    # pending ones are settled: ``complete_publication`` runs outside the
    # write lock, and a version it turns ready in between is caught here.
    archived = await query_raw_with_schema(
        """UPDATE {schema_prefix}"SkillVersion"
        SET "state" = 'archived', "stateReason" = $3
        WHERE "userId" = $1 AND "headId" = $2
              AND "state" IN ('ready', 'needs_decision')
        RETURNING "id" """,
        user_id,
        head.id,
        DELETED_REASON,
        client=tx,
    )
    return SkillDeletion(
        committed=True,
        marker=SkillVersionRecord.from_db(marker),
        archived_version_ids=[str(row["id"]) for row in archived],
        abandoned_version_ids=abandoned,
    )


async def _suppress(
    tx: Prisma,
    user_id: str,
    expert_id: str | None,
    skill_name: str,
    behaviors: list[DeletedBehavior],
    actor_user_id: str | None,
) -> None:
    await tx.skillsuppression.create_many(
        data=[
            {
                "userId": user_id,
                "expertId": expert_id,
                "ownerKey": owner_key_for(expert_id),
                "skillName": skill_name,
                "behaviorFingerprint": behavior.behavior_fingerprint,
                "behaviorTokens": list(behavior.behavior_tokens),
                "evidenceFingerprints": list(behavior.evidence_fingerprints),
                "actorUserId": actor_user_id,
                "reason": "deleted by the owner",
            }
            for behavior in behaviors
        ],
        skip_duplicates=True,
    )


async def _append_marker(
    tx: Prisma, head: prisma.models.SkillHead, actor_user_id: str | None
) -> prisma.models.SkillVersion:
    # A version number a concurrent publication already holds fails the
    # insert, which rolls the deletion back for the caller to retry.
    return await tx.skillversion.create(
        data={
            "userId": head.userId,
            "expertId": head.expertId,
            "ownerKey": head.ownerKey,
            "skillName": head.skillName,
            "headId": head.id,
            "version": await next_version_number(head.id),
            "content": "",
            "contentHash": content_hash(""),
            "packageFiles": version_files_json(None),
            "contentApprovals": content_approvals_json([]),
            "description": "",
            "triggers": [],
            "origin": DELETED_ORIGIN,
            "actorUserId": actor_user_id,
            "summary": (
                f"Deleted v{head.currentVersion}"
                if head.currentVersionId
                else "Deleted"
            ),
            "sources": SafeJson([]),
            "evidence": SafeJson([]),
            "limits": [],
            "state": DELETED_STATE,
        }
    )


async def _settle_pending(tx: Prisma, user_id: str, head_id: str) -> list[str]:
    """Settle publications still waiting for their workspace write.

    ``suppressed`` is terminal, so the review is not re-run for a skill its
    owner deleted. A version that turned ready meanwhile is not pending any
    more and is left to the archive step.
    """
    pending = await tx.skillversion.find_many(
        where={"userId": user_id, "headId": head_id, "state": PENDING_WRITE_STATE}
    )
    return [
        version.id
        for version in pending
        if await settle_publication_in(
            tx,
            user_id,
            version_id=version.id,
            review_id=version.reviewId,
            state="stale",
            disposition="suppressed",
            reason=DELETED_REASON,
        )
    ]
