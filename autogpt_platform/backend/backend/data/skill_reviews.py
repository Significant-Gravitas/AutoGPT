"""Durable review ledger for skill learning.

One row per (source, revision, policy version); a retryable failure
updates the same row so a crashed worker never re-applies a completed
change. Every mutation enforces the owning ``user_id``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import prisma
import prisma.models
import prisma.types
from pydantic import BaseModel

from backend.data.skill_learning import (
    LearningAccessError,
    LearningSourceRecord,
    canonical_revision,
    list_pending_sources,
    require_owned_source,
)
from backend.util.json import SafeJson

TERMINAL_REVIEW_DISPOSITIONS = frozenset(
    {
        "applied",
        "skipped",
        "no_novel_procedure",
        "blocked_content",
        "suppressed",
        "needs_decision",
        "stale_eligibility",
        "inaccessible_evidence",
    }
)


class LearningReviewRecord(BaseModel):
    id: str
    user_id: str
    expert_id: str | None
    owner_key: str
    source_id: str
    source_revision: str
    policy_version: int
    run_id: str
    disposition: str
    reason: str
    attempts: int
    model_attempts: int = 0
    replay: dict[str, Any] | None = None
    change_fingerprint: str | None
    skill_name: str | None
    applied_version_id: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    cost_microdollars: int | None
    created_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_db(cls, row: prisma.models.SkillLearningReview) -> "LearningReviewRecord":
        return cls(
            id=row.id,
            user_id=row.userId,
            expert_id=row.expertId,
            owner_key=row.ownerKey,
            source_id=row.sourceId,
            source_revision=row.sourceRevision,
            policy_version=row.policyVersion,
            run_id=row.runId,
            disposition=row.disposition,
            reason=row.reason,
            attempts=row.attempts,
            model_attempts=row.modelAttempts,
            replay=row.reviewReplay if isinstance(row.reviewReplay, dict) else None,
            change_fingerprint=row.changeFingerprint,
            skill_name=row.skillName,
            applied_version_id=row.appliedVersionId,
            model=row.model,
            input_tokens=row.inputTokens,
            output_tokens=row.outputTokens,
            cost_microdollars=(
                int(row.costMicrodollars) if row.costMicrodollars is not None else None
            ),
            created_at=row.createdAt,
            completed_at=row.completedAt,
        )


def review_data(
    *,
    run_id: str,
    disposition: str,
    reason: str = "",
    change_fingerprint: str | None = None,
    skill_name: str | None = None,
    applied_version_id: str | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_microdollars: int | None = None,
    completed: bool = True,
) -> dict[str, Any]:
    """Column payload shared by the standalone upsert and the publication tx."""
    result = {
        "runId": run_id,
        "disposition": disposition,
        "reason": reason[:2000],
        "changeFingerprint": change_fingerprint,
        "skillName": skill_name,
        "appliedVersionId": applied_version_id,
        "model": model,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "costMicrodollars": cost_microdollars,
        "completedAt": datetime.now(timezone.utc) if completed else None,
    }
    if (
        model is None
        and not input_tokens
        and not output_tokens
        and cost_microdollars is None
    ):
        for key in ("model", "inputTokens", "outputTokens", "costMicrodollars"):
            result.pop(key)
    if completed or disposition in ("applied_pending", "conflict"):
        result["reviewReplay"] = SafeJson(None)
    return result


async def upsert_review_in(
    client: Any,
    user_id: str,
    *,
    source: LearningSourceRecord,
    source_revision: str,
    policy_version: int,
    data: dict[str, Any],
) -> prisma.models.SkillLearningReview:
    """Ledger upsert on ``client`` (a transaction or the default client).

    The source must already be verified as the caller's — see
    :func:`require_owned_source`.
    """
    rev = canonical_revision(source_revision)
    existing = await client.skilllearningreview.find_unique(
        where={
            "sourceId_sourceRevision_policyVersion": {
                "sourceId": source.id,
                "sourceRevision": rev,
                "policyVersion": policy_version,
            }
        }
    )
    if existing is None:
        return await client.skilllearningreview.create(
            data={
                "userId": user_id,
                "expertId": source.expert_id,
                "ownerKey": source.owner_key,
                "sourceId": source.id,
                "sourceRevision": rev,
                "policyVersion": policy_version,
                **data,
            }
        )
    if existing.userId != user_id:
        raise LearningAccessError("review belongs to another user")
    row = await client.skilllearningreview.update(
        where={"id": existing.id},
        data={**data, "attempts": existing.attempts + 1},
    )
    return row or existing


async def upsert_review(
    user_id: str,
    *,
    source_id: str,
    source_revision: str,
    policy_version: int,
    run_id: str,
    disposition: str,
    reason: str = "",
    change_fingerprint: str | None = None,
    skill_name: str | None = None,
    applied_version_id: str | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_microdollars: int | None = None,
    completed: bool = True,
) -> LearningReviewRecord:
    """Write the ledger row for one review attempt (update in place on retry)."""
    source = await require_owned_source(user_id, source_id)
    row = await upsert_review_in(
        prisma.get_client(),
        user_id,
        source=source,
        source_revision=source_revision,
        policy_version=policy_version,
        data=review_data(
            run_id=run_id,
            disposition=disposition,
            reason=reason,
            change_fingerprint=change_fingerprint,
            skill_name=skill_name,
            applied_version_id=applied_version_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_microdollars=cost_microdollars,
            completed=completed,
        ),
    )
    return LearningReviewRecord.from_db(row)


async def get_review_for_revision(
    user_id: str, source_id: str, source_revision: str, policy_version: int
) -> LearningReviewRecord | None:
    """The ledger row for one (source, revision, policy), if any."""
    row = await prisma.models.SkillLearningReview.prisma().find_first(
        where={
            "userId": user_id,
            "sourceId": source_id,
            "sourceRevision": canonical_revision(source_revision),
            "policyVersion": policy_version,
        }
    )
    return LearningReviewRecord.from_db(row) if row else None


async def get_review(user_id: str, review_id: str) -> LearningReviewRecord | None:
    row = await prisma.models.SkillLearningReview.prisma().find_first(
        where={"id": review_id, "userId": user_id}
    )
    return LearningReviewRecord.from_db(row) if row else None


async def list_reviews(
    user_id: str,
    *,
    owner_key: str | None = None,
    disposition: str | None = None,
    limit: int = 50,
) -> list[LearningReviewRecord]:
    where: prisma.types.SkillLearningReviewWhereInput = {"userId": user_id}
    if owner_key is not None:
        where["ownerKey"] = owner_key
    if disposition is not None:
        where["disposition"] = disposition
    rows = await prisma.models.SkillLearningReview.prisma().find_many(
        where=where, order={"createdAt": "desc"}, take=limit
    )
    return [LearningReviewRecord.from_db(row) for row in rows]


class LearningRunSummary(BaseModel):
    pending_sources: int
    oldest_pending_at: datetime | None
    last_review_at: datetime | None
    last_applied_at: datetime | None
    retrying_reviews: int
    cost_microdollars_30d: int


async def summarize_learning(user_id: str) -> LearningRunSummary:
    """Operator-facing status: backlog, freshness, retries, and recent cost."""
    pending = await list_pending_sources(user_id, limit=500)
    reviews = await list_reviews(user_id, limit=500)
    since = datetime.now(timezone.utc).timestamp() - 30 * 86400
    return LearningRunSummary(
        pending_sources=len(pending),
        oldest_pending_at=min((s.updated_at for s in pending), default=None),
        last_review_at=reviews[0].created_at if reviews else None,
        last_applied_at=next(
            (r.created_at for r in reviews if r.disposition == "applied"), None
        ),
        retrying_reviews=sum(1 for r in reviews if r.completed_at is None),
        cost_microdollars_30d=sum(
            r.cost_microdollars or 0
            for r in reviews
            if r.created_at.timestamp() >= since
        ),
    )


async def reserve_model_review(
    user_id: str,
    *,
    source_id: str,
    source_revision: str,
    policy_version: int,
    run_id: str,
    max_attempts: int,
    expected_epoch: int,
) -> LearningReviewRecord | None:
    from backend.data.db import query_raw_with_schema, transaction

    async with transaction() as tx:
        owned = await query_raw_with_schema(
            'SELECT * FROM {schema_prefix}"SkillLearningSource" '
            'WHERE "id" = $1 AND "userId" = $2 FOR UPDATE',
            source_id,
            user_id,
            client=tx,
            model=prisma.models.SkillLearningSource,
        )
        if not owned:
            return None
        source = LearningSourceRecord.from_db(owned[0])
        rev = canonical_revision(source_revision)
        if (
            source.revision != rev
            or source.epoch != expected_epoch
            or source.eligibility != "eligible"
        ):
            return None
        key: prisma.types.SkillLearningReviewWhereUniqueInput = {
            "sourceId_sourceRevision_policyVersion": {
                "sourceId": source_id,
                "sourceRevision": rev,
                "policyVersion": policy_version,
            }
        }
        existing = await tx.skilllearningreview.find_unique(where=key)
        if existing is not None and (
            existing.userId != user_id
            or existing.modelAttempts >= max_attempts
            or existing.disposition in TERMINAL_REVIEW_DISPOSITIONS
            or existing.disposition == "applied_pending"
            or existing.reviewReplay is not None
        ):
            return None
        if existing is None:
            row = await tx.skilllearningreview.create(
                data={
                    "userId": user_id,
                    "expertId": source.expert_id,
                    "ownerKey": source.owner_key,
                    "sourceId": source_id,
                    "sourceRevision": rev,
                    "policyVersion": policy_version,
                    "runId": run_id,
                    "disposition": "reviewing",
                    "attempts": 0,
                    "modelAttempts": 1,
                }
            )
        else:
            row = await tx.skilllearningreview.update(
                where={"id": existing.id},
                data={
                    "runId": run_id,
                    "disposition": "reviewing",
                    "reason": "review started",
                    "modelAttempts": {"increment": 1},
                    "completedAt": None,
                },
            )
    return LearningReviewRecord.from_db(row) if row else None


async def checkpoint_review_usage(
    user_id: str,
    review_id: str,
    run_id: str,
    *,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cost_microdollars: int | None,
) -> bool:
    updated = await prisma.models.SkillLearningReview.prisma().update_many(
        where={
            "id": review_id,
            "userId": user_id,
            "runId": run_id,
            "completedAt": None,
            "disposition": "reviewing",
        },
        data={
            "model": model,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "costMicrodollars": cost_microdollars,
        },
    )
    return updated > 0


async def set_review_replay(
    user_id: str,
    review_id: str,
    run_id: str,
    replay: dict[str, Any] | None,
) -> bool:
    import json

    from backend.util.json import SafeJson

    if replay is not None and len(json.dumps(replay).encode("utf-8")) > 524_288:
        raise ValueError("review replay exceeds its storage budget")
    updated = await prisma.models.SkillLearningReview.prisma().update_many(
        where={
            "id": review_id,
            "userId": user_id,
            "runId": run_id,
            "completedAt": None,
        },
        data={
            "reviewReplay": SafeJson(replay) if replay is not None else SafeJson(None)
        },
    )
    return updated > 0
