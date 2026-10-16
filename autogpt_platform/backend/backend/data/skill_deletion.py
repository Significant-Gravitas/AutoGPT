"""Record an owner's deletion of a skill in its version history.

Deleting a skill removes its workspace package, but the head would still
name the last version and the older versions would stay ``ready``:
excluding an old source would then restore one of them, and a learned
``create`` for the name would conflict with the head for good.
:func:`record_skill_deletion` ends the history in one transaction instead.
It suppresses the deleted behaviour, appends a ``deleted`` marker version,
clears the head's pointer and per-skill policy, archives what is left, and
settles any publication still waiting for its workspace write.
"""

from __future__ import annotations

import prisma.errors
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
    """``committed=False`` means the head no longer names the version the
    caller read, so nothing was recorded; read it again and retry."""

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

    The head row is locked first, so a publication's compare-and-swap either
    lands before (and the caller sees ``committed=False``) or fails against
    the marker's version number afterwards. A skill with no head has no
    history to end; only its suppressions are recorded. The next version
    under the name starts a new chain: the head names no version, and the
    marker links to none, so revoking a source of the deleted history can
    never reach it.
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
    heads = await query_raw_with_schema(
        'SELECT "id", "currentVersion", "currentVersionId" '
        'FROM {schema_prefix}"SkillHead" '
        'WHERE "userId" = $1 AND "ownerKey" = $2 AND "skillName" = $3 FOR UPDATE',
        user_id,
        owner_key,
        skill_name,
        client=tx,
    )
    current_id = heads[0]["currentVersionId"] if heads else None
    if current_id != expected_current_version_id:
        raise _HeadChanged()
    if behaviors:
        await tx.skillsuppression.create_many(
            data=[
                {
                    "userId": user_id,
                    "expertId": expert_id,
                    "ownerKey": owner_key,
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
    if not heads:
        return SkillDeletion(committed=True)
    head_id = heads[0]["id"]
    number = await next_version_number(head_id)
    marker = await tx.skillversion.create(
        data={
            "userId": user_id,
            "expertId": expert_id,
            "ownerKey": owner_key,
            "skillName": skill_name,
            "headId": head_id,
            "version": number,
            "content": "",
            "contentHash": content_hash(""),
            "packageFiles": version_files_json(None),
            "contentApprovals": content_approvals_json([]),
            "description": "",
            "triggers": [],
            "origin": DELETED_ORIGIN,
            "actorUserId": actor_user_id,
            "summary": (
                f"Deleted v{heads[0]['currentVersion']}" if current_id else "Deleted"
            ),
            "sources": SafeJson([]),
            "evidence": SafeJson([]),
            "limits": [],
            "state": DELETED_STATE,
        }
    )
    await tx.skillhead.update(
        where={"id": head_id},
        data={
            "currentVersion": number,
            "currentVersionId": None,
            "contentHash": None,
            "autoImprove": True,
            "learningPausedAt": None,
            "usePausedAt": None,
        },
    )
    # A ``ready`` version is what a source revocation would restore, and an
    # open proposal would edit a skill that is gone.
    archived = await query_raw_with_schema(
        """UPDATE {schema_prefix}"SkillVersion"
        SET "state" = 'archived', "stateReason" = $3
        WHERE "userId" = $1 AND "headId" = $2
              AND "state" IN ('ready', 'needs_decision')
        RETURNING "id" """,
        user_id,
        head_id,
        DELETED_REASON,
        client=tx,
    )
    pending = await tx.skillversion.find_many(
        where={"userId": user_id, "headId": head_id, "state": PENDING_WRITE_STATE}
    )
    for version in pending:
        # ``suppressed`` is terminal: the review is not re-run for a skill
        # its owner deleted.
        await settle_publication_in(
            tx,
            user_id,
            version_id=version.id,
            review_id=version.reviewId,
            state="stale",
            disposition="suppressed",
            reason=DELETED_REASON,
        )
    return SkillDeletion(
        committed=True,
        marker=SkillVersionRecord.from_db(marker),
        archived_version_ids=[str(row["id"]) for row in archived],
        abandoned_version_ids=[version.id for version in pending],
    )
