"""Data layer for skill-learning sources and the durable review ledger.

A *source* row is one revision of something the nightly learner may learn
from (an ordinary chat session today). A *review* row is the ledger entry
for one (source, revision, policy version) triple; retries update the row
in place so a crashed worker never re-applies a completed change.

Every mutation takes the owning ``user_id`` and enforces it in the WHERE
clause; a caller holding a foreign row id gets :class:`LearningAccessError`
or a no-op, never a write.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import prisma.errors
import prisma.models
import prisma.types
from pydantic import BaseModel, Field

from backend.data.db import execute_raw_with_schema, query_raw_with_schema, transaction
from backend.util.json import SafeJson

PERSONAL_OWNER_KEY = "personal"
MAX_EVIDENCE_REFS = 40
MAX_OUTCOME_SIGNALS = 40
_REVISION_PAD = 12


class LearningAccessError(PermissionError):
    """The caller does not own the learning row it tried to change."""


def owner_key_for(expert_id: str | None) -> str:
    """Owner half of the learning unique keys: the expert id or ``personal``."""
    return PERSONAL_OWNER_KEY if expert_id is None else expert_id


def canonical_revision(revision: str) -> str:
    """Zero-pad numeric revisions so string comparison orders them correctly.

    Adapters hand in opaque revision strings; chat sessions use message
    sequence numbers. Storing ``"000000000042"`` lets the cursor guards in
    SQL (``"processedRevision" < $rev``) and the merge below compare with
    plain string ordering, while non-numeric revisions pass through.
    """
    text = revision.strip()
    if text.isdigit():
        return text.zfill(_REVISION_PAD)
    return text


def _json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


class LearningSourceRecord(BaseModel):
    id: str
    user_id: str
    expert_id: str | None
    owner_key: str
    source_kind: str
    source_id: str
    revision: str
    processed_revision: str | None
    origin: str
    eligibility: str
    epoch: int
    approval_event_id: str | None
    approval_actor_id: str | None
    approved_revision: str | None
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    outcome_signals: list[dict[str, Any]] = Field(default_factory=list)
    excluded_at: datetime | None
    excluded_by_user_id: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.SkillLearningSource) -> "LearningSourceRecord":
        return cls(
            id=row.id,
            user_id=row.userId,
            expert_id=row.expertId,
            owner_key=row.ownerKey,
            source_kind=row.sourceKind,
            source_id=row.sourceId,
            revision=row.revision,
            processed_revision=row.processedRevision,
            origin=row.origin,
            eligibility=row.eligibility,
            epoch=row.epoch,
            approval_event_id=row.approvalEventId,
            approval_actor_id=row.approvalActorId,
            approved_revision=row.approvedRevision,
            evidence_refs=_json_list(row.evidenceRefs),
            outcome_signals=_json_list(row.outcomeSignals),
            excluded_at=row.excludedAt,
            excluded_by_user_id=row.excludedByUserId,
            created_at=row.createdAt,
            updated_at=row.updatedAt,
        )

    @property
    def has_unprocessed_revision(self) -> bool:
        return (
            self.processed_revision is None or self.processed_revision < self.revision
        )


def _merge_bounded(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]], cap: int
) -> list[dict[str, Any]]:
    """Append new items (by ``kind``/``ref`` identity) and keep the newest ``cap``."""
    seen = {(item.get("kind"), item.get("ref")) for item in existing}
    merged = list(existing)
    for item in incoming:
        key = (item.get("kind"), item.get("ref"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged[-cap:]


_SOURCE_UNIQUE_WHERE = "userId_sourceKind_sourceId"


def _unique_where(
    user_id: str, source_kind: str, source_id: str
) -> prisma.types.SkillLearningSourceWhereUniqueInput:
    return {
        _SOURCE_UNIQUE_WHERE: {
            "userId": user_id,
            "sourceKind": source_kind,
            "sourceId": source_id,
        }
    }


async def upsert_source_revision(
    user_id: str,
    *,
    expert_id: str | None,
    source_kind: str,
    source_id: str,
    revision: str,
    evidence_refs: list[dict[str, Any]],
    outcome_signals: list[dict[str, Any]],
    origin: str = "ordinary",
) -> LearningSourceRecord:
    """Record a revision of a source, merging bounded evidence under a row lock.

    * ``revision`` never moves backwards: an out-of-order turn merges its
      evidence but leaves the newer revision in place.
    * ``requested`` origin is sticky so a later ordinary turn cannot
      downgrade the label.
    * A source that reappears under a different owner scope is marked
      inaccessible (epoch bumped) rather than silently re-owned.
    * Exclusion is never undone here.
    """
    owner_key = owner_key_for(expert_id)
    rev = canonical_revision(revision)
    try:
        row = await prisma.models.SkillLearningSource.prisma().create(
            data={
                "userId": user_id,
                "expertId": expert_id,
                "ownerKey": owner_key,
                "sourceKind": source_kind,
                "sourceId": source_id,
                "revision": rev,
                "origin": origin,
                "evidenceRefs": SafeJson(evidence_refs[-MAX_EVIDENCE_REFS:]),
                "outcomeSignals": SafeJson(outcome_signals[-MAX_OUTCOME_SIGNALS:]),
            }
        )
        return LearningSourceRecord.from_db(row)
    except prisma.errors.UniqueViolationError:
        pass

    async with transaction() as tx:
        locked = await query_raw_with_schema(
            'SELECT "id" FROM {schema_prefix}"SkillLearningSource" '
            'WHERE "userId" = $1 AND "sourceKind" = $2 AND "sourceId" = $3 '
            "FOR UPDATE",
            user_id,
            source_kind,
            source_id,
            client=tx,
        )
        if not locked:
            raise LearningAccessError("learning source not found for this user")
        existing = await tx.skilllearningsource.find_unique(
            where=_unique_where(user_id, source_kind, source_id)
        )
        if existing is None:
            raise LearningAccessError("learning source not found for this user")
        if existing.ownerKey != owner_key:
            row = await tx.skilllearningsource.update(
                where={"id": existing.id},
                data={"eligibility": "inaccessible", "epoch": existing.epoch + 1},
            )
            return LearningSourceRecord.from_db(row or existing)
        data: prisma.types.SkillLearningSourceUpdateInput = {
            "evidenceRefs": SafeJson(
                _merge_bounded(
                    _json_list(existing.evidenceRefs), evidence_refs, MAX_EVIDENCE_REFS
                )
            ),
            "outcomeSignals": SafeJson(
                _merge_bounded(
                    _json_list(existing.outcomeSignals),
                    outcome_signals,
                    MAX_OUTCOME_SIGNALS,
                )
            ),
        }
        if rev > existing.revision:
            data["revision"] = rev
        if "requested" in (existing.origin, origin):
            data["origin"] = "requested"
        row = await tx.skilllearningsource.update(where={"id": existing.id}, data=data)
        return LearningSourceRecord.from_db(row or existing)


async def get_source(user_id: str, source_id: str) -> LearningSourceRecord | None:
    row = await prisma.models.SkillLearningSource.prisma().find_first(
        where={"id": source_id, "userId": user_id}
    )
    return LearningSourceRecord.from_db(row) if row else None


async def get_source_by_ref(
    user_id: str, source_kind: str, source_ref: str
) -> LearningSourceRecord | None:
    row = await prisma.models.SkillLearningSource.prisma().find_first(
        where={"userId": user_id, "sourceKind": source_kind, "sourceId": source_ref}
    )
    return LearningSourceRecord.from_db(row) if row else None


_PENDING_PREDICATE = (
    '"userId" = $1 AND "eligibility" = \'eligible\' '
    'AND ("processedRevision" IS NULL OR "processedRevision" < "revision")'
)


async def list_pending_owner_keys(user_id: str) -> list[str]:
    """Owners (personal + experts) that have at least one pending source."""
    rows = await query_raw_with_schema(
        'SELECT DISTINCT "ownerKey" FROM {schema_prefix}"SkillLearningSource" '
        f"WHERE {_PENDING_PREDICATE}",
        user_id,
    )
    return sorted(str(row["ownerKey"]) for row in rows)


async def list_pending_sources(
    user_id: str, *, limit: int = 100, owner_key: str | None = None
) -> list[LearningSourceRecord]:
    """Eligible sources whose newest revision has no durable disposition yet.

    The pending predicate is evaluated in SQL (a column-to-column compare
    Prisma cannot express) so completed rows never crowd newer work out of
    the page. Pass ``owner_key`` to page one owner's backlog so a large
    backlog for one expert cannot hide another's fresh work.
    """
    if owner_key is None:
        rows = await query_raw_with_schema(
            'SELECT "id" FROM {schema_prefix}"SkillLearningSource" '
            f"WHERE {_PENDING_PREDICATE} "
            'ORDER BY "updatedAt" ASC LIMIT $2',
            user_id,
            limit,
        )
    else:
        rows = await query_raw_with_schema(
            'SELECT "id" FROM {schema_prefix}"SkillLearningSource" '
            f'WHERE {_PENDING_PREDICATE} AND "ownerKey" = $3 '
            'ORDER BY "updatedAt" ASC LIMIT $2',
            user_id,
            limit,
            owner_key,
        )
    ids = [str(row["id"]) for row in rows]
    if not ids:
        return []
    found = await prisma.models.SkillLearningSource.prisma().find_many(
        where={"id": {"in": ids}, "userId": user_id}
    )
    by_id = {row.id: LearningSourceRecord.from_db(row) for row in found}
    return [by_id[i] for i in ids if i in by_id]


async def set_source_eligibility(
    user_id: str,
    source_id: str,
    eligibility: str,
    *,
    excluded_by_user_id: str | None = None,
) -> LearningSourceRecord | None:
    """Change eligibility with an atomic epoch bump so in-flight reviews see it."""
    if eligibility == "excluded":
        updated = await execute_raw_with_schema(
            'UPDATE {schema_prefix}"SkillLearningSource" '
            'SET "eligibility" = $3, "epoch" = "epoch" + 1, '
            '"excludedAt" = NOW(), "excludedByUserId" = $4, "updatedAt" = NOW() '
            'WHERE "id" = $1 AND "userId" = $2',
            source_id,
            user_id,
            eligibility,
            excluded_by_user_id,
        )
    else:
        updated = await execute_raw_with_schema(
            'UPDATE {schema_prefix}"SkillLearningSource" '
            'SET "eligibility" = $3, "epoch" = "epoch" + 1, "updatedAt" = NOW() '
            'WHERE "id" = $1 AND "userId" = $2',
            source_id,
            user_id,
            eligibility,
        )
    if not updated:
        return None
    return await get_source(user_id, source_id)


async def set_source_approval(
    user_id: str,
    source_id: str,
    *,
    approval_event_id: str | None,
    approval_actor_id: str | None,
    approved_revision: str | None,
    eligibility: str,
) -> LearningSourceRecord | None:
    """Record (or withdraw) an approval checkpoint for an approval-aware source."""
    updated = await execute_raw_with_schema(
        'UPDATE {schema_prefix}"SkillLearningSource" '
        'SET "approvalEventId" = $3, "approvalActorId" = $4, '
        '"approvedRevision" = $5, "eligibility" = $6, "epoch" = "epoch" + 1, '
        '"updatedAt" = NOW() WHERE "id" = $1 AND "userId" = $2',
        source_id,
        user_id,
        approval_event_id,
        approval_actor_id,
        canonical_revision(approved_revision) if approved_revision else None,
        eligibility,
    )
    if not updated:
        return None
    return await get_source(user_id, source_id)


async def advance_source_cursor(user_id: str, source_id: str, revision: str) -> bool:
    """Mark ``revision`` as durably dispositioned. Never moves backwards.

    Returns ``True`` when the cursor advanced, ``False`` when it was already
    at or past ``revision`` (or the row is not the caller's).
    """
    rev = canonical_revision(revision)
    updated = await execute_raw_with_schema(
        'UPDATE {schema_prefix}"SkillLearningSource" '
        'SET "processedRevision" = $3 WHERE "id" = $1 AND "userId" = $2 '
        'AND ("processedRevision" IS NULL OR "processedRevision" < $3)',
        source_id,
        user_id,
        rev,
    )
    return updated > 0


async def require_owned_source(user_id: str, source_id: str) -> LearningSourceRecord:
    source = await get_source(user_id, source_id)
    if source is None:
        raise LearningAccessError("learning source not found for this user")
    return source
