"""Transactional publication of learned skill versions.

``commit_version`` appends a ``pending_write`` version, swaps the head
pointer by compare-and-swap, and stamps the review ledger — all in one
database transaction that also re-checks the source's live eligibility.
The workspace write happens afterwards; ``complete_publication`` and
``abandon_publication`` close the two-phase commit, and
``list_pending_publications`` feeds crash recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import prisma.errors
import prisma.models
import prisma.types
from pydantic import BaseModel, Field

from backend.data.db import query_raw_with_schema, transaction
from backend.data.skill_learning import LearningSourceRecord, canonical_revision
from backend.data.skill_reviews import review_data, upsert_review_in
from backend.data.skill_versions import (
    PENDING_WRITE_STATE,
    SkillHeadRecord,
    SkillVersionRecord,
    content_hash,
    list_versions_in_states,
    next_version_number,
)
from backend.util.json import SafeJson

# Ledger disposition between the atomic pointer swap and the workspace
# write; ``reconcile`` completes or reverts it after a crash.
APPLIED_PENDING_DISPOSITION = "applied_pending"


class VersionDraft(BaseModel):
    """Everything needed to append one version row."""

    content: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    origin: str
    summary: str = ""
    actor_user_id: str | None = None
    base_version_id: str | None = None
    restored_from_version_id: str | None = None
    review_id: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)


class CommitResult(BaseModel):
    committed: bool
    version: SkillVersionRecord | None = None
    review_id: str | None = None
    reason: str = ""


class ReviewStamp(BaseModel):
    """Ledger row written atomically with a publication."""

    source: LearningSourceRecord
    source_revision: str
    policy_version: int
    run_id: str
    change_fingerprint: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_microdollars: int | None = None


async def commit_version(
    user_id: str,
    *,
    head: SkillHeadRecord,
    draft: VersionDraft,
    expected_current_version: int,
    review: ReviewStamp | None = None,
    auto_improve: bool | None = None,
) -> CommitResult:
    """Atomically append a ``pending_write`` version, swap the head pointer,
    and stamp the ledger.

    The head swap is a compare-and-swap on ``expected_current_version`` so a
    concurrent human edit or a second review can never overwrite each other;
    the loser sees ``committed=False`` and nothing is written. The workspace
    file is written by the caller afterwards; until then the ledger row
    carries ``applied_pending`` and the version stays ``pending_write`` so a
    crash is recoverable without a second model call.
    """
    async with transaction() as tx:
        if review is not None:
            await _require_committable_source(tx, user_id, review)
            await _require_no_terminal_review(tx, user_id, review)
        number = await next_version_number(head.id)
        try:
            row = await tx.skillversion.create(
                data={
                    "userId": user_id,
                    "expertId": head.expert_id,
                    "ownerKey": head.owner_key,
                    "skillName": head.skill_name,
                    "headId": head.id,
                    "version": number,
                    "content": draft.content,
                    "contentHash": content_hash(draft.content),
                    "description": draft.description,
                    "triggers": list(draft.triggers),
                    "origin": draft.origin,
                    "actorUserId": draft.actor_user_id,
                    "summary": draft.summary[:1000],
                    "baseVersionId": draft.base_version_id,
                    "restoredFromVersionId": draft.restored_from_version_id,
                    "reviewId": draft.review_id,
                    "sources": SafeJson(draft.sources),
                    "evidence": SafeJson(draft.evidence),
                    "limits": list(draft.limits),
                    "state": PENDING_WRITE_STATE,
                }
            )
        except prisma.errors.UniqueViolationError:
            return CommitResult(committed=False, reason="version number contended")
        head_data: prisma.types.SkillHeadUpdateInput = {
            "currentVersion": number,
            "currentVersionId": row.id,
            "contentHash": row.contentHash,
        }
        if auto_improve is not None:
            head_data["autoImprove"] = auto_improve
        head_where: prisma.types.SkillHeadWhereInput = {
            "id": head.id,
            "userId": user_id,
            "currentVersion": expected_current_version,
        }
        if review is not None:
            # An automated change also loses to a concurrent "pause
            # learning" on the skill; owner edits and restores are not
            # subject to that policy.
            head_where["learningPausedAt"] = None
        swapped = await tx.skillhead.update_many(where=head_where, data=head_data)
        if swapped == 0:
            # Roll the whole transaction back: no version row, no ledger.
            raise _CommitConflict("the skill changed while this version was prepared")
        review_id: str | None = None
        if review is not None:
            ledger = await upsert_review_in(
                tx,
                user_id,
                source=review.source,
                source_revision=review.source_revision,
                policy_version=review.policy_version,
                data=review_data(
                    run_id=review.run_id,
                    disposition=APPLIED_PENDING_DISPOSITION,
                    reason="pointer swapped; workspace write pending",
                    change_fingerprint=review.change_fingerprint,
                    skill_name=head.skill_name,
                    applied_version_id=row.id,
                    model=review.model,
                    input_tokens=review.input_tokens,
                    output_tokens=review.output_tokens,
                    cost_microdollars=review.cost_microdollars,
                    completed=False,
                ),
            )
            review_id = ledger.id
            row = (
                await tx.skillversion.update(
                    where={"id": row.id}, data={"reviewId": review_id}
                )
                or row
            )
    return CommitResult(
        committed=True, version=SkillVersionRecord.from_db(row), review_id=review_id
    )


class _CommitConflict(Exception):
    """Internal: aborts the publication transaction (lost CAS, stale
    source eligibility, or an already-settled review)."""


async def _require_committable_source(
    tx: Any, user_id: str, review: ReviewStamp
) -> None:
    """Lock the source row and re-check eligibility inside the transaction.

    The model-layer revalidate ran earlier; a pause, exclusion, revocation,
    or new revision landing between that check and this commit bumps the
    epoch or changes the revision, and must abort the commit.
    """
    rows = await query_raw_with_schema(
        'SELECT "eligibility", "epoch", "revision" '
        'FROM {schema_prefix}"SkillLearningSource" '
        'WHERE "id" = $1 AND "userId" = $2 FOR UPDATE',
        review.source.id,
        user_id,
        client=tx,
    )
    if not rows:
        raise _CommitConflict("source no longer accessible")
    live = rows[0]
    if live["eligibility"] != "eligible":
        raise _CommitConflict(f"source is {live['eligibility']}")
    if int(live["epoch"]) != review.source.epoch:
        raise _CommitConflict("source eligibility changed since it was reviewed")
    if live["revision"] != canonical_revision(review.source_revision):
        raise _CommitConflict("a newer source revision exists")


async def _require_no_terminal_review(
    tx: Any, user_id: str, review: ReviewStamp
) -> None:
    """A revision whose ledger row is already settled is never re-applied."""
    existing = await tx.skilllearningreview.find_unique(
        where={
            "sourceId_sourceRevision_policyVersion": {
                "sourceId": review.source.id,
                "sourceRevision": canonical_revision(review.source_revision),
                "policyVersion": review.policy_version,
            }
        }
    )
    if existing is None:
        return
    if existing.userId != user_id:
        raise _CommitConflict("review belongs to another user")
    if existing.disposition in ("applied", APPLIED_PENDING_DISPOSITION):
        raise _CommitConflict("this revision was already applied")


async def commit_version_safe(
    user_id: str,
    *,
    head: SkillHeadRecord,
    draft: VersionDraft,
    expected_current_version: int,
    review: ReviewStamp | None = None,
    auto_improve: bool | None = None,
) -> CommitResult:
    """``commit_version`` with the CAS loss reported instead of raised."""
    try:
        return await commit_version(
            user_id,
            head=head,
            draft=draft,
            expected_current_version=expected_current_version,
            review=review,
            auto_improve=auto_improve,
        )
    except _CommitConflict as conflict:
        return CommitResult(committed=False, reason=str(conflict))


async def list_pending_publications(user_id: str) -> list[SkillVersionRecord]:
    """Versions whose pointer swap landed but whose workspace write did not."""
    return await list_versions_in_states(user_id, [PENDING_WRITE_STATE])


async def complete_publication(
    user_id: str, *, version_id: str, review_id: str | None, reason: str = ""
) -> None:
    """Mark the workspace write done: version ready, ledger row applied."""
    await prisma.models.SkillVersion.prisma().update_many(
        where={"id": version_id, "userId": user_id, "state": PENDING_WRITE_STATE},
        data={"state": "ready", "stateReason": reason[:1000]},
    )
    if review_id is not None:
        await prisma.models.SkillLearningReview.prisma().update_many(
            where={"id": review_id, "userId": user_id},
            data={
                "disposition": "applied",
                "reason": reason[:2000],
                "completedAt": datetime.now(timezone.utc),
            },
        )


async def abandon_publication(
    user_id: str, *, version_id: str, review_id: str | None, reason: str
) -> None:
    """A pending version that can no longer be written: stale, and the
    ledger records the conflict so the source is reviewed again later."""
    await prisma.models.SkillVersion.prisma().update_many(
        where={"id": version_id, "userId": user_id, "state": PENDING_WRITE_STATE},
        data={"state": "stale", "stateReason": reason[:1000]},
    )
    if review_id is not None:
        await prisma.models.SkillLearningReview.prisma().update_many(
            where={"id": review_id, "userId": user_id},
            data={
                "disposition": "conflict",
                "reason": reason[:2000],
                "completedAt": datetime.now(timezone.utc),
            },
        )
