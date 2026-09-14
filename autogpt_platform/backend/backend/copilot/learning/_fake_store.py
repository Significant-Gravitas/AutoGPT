"""In-memory stand-in for the learning data layer, for unit tests.

Implements the subset of ``backend.data.skill_learning`` and
``backend.data.skill_versions`` the runtime calls through the
``db_accessors`` seams, with the same ownership, cursor-monotonicity, and
compare-and-swap semantics. Tests patch the accessor names in the module
under test to a ``FakeLearningStore`` instance.
"""

from __future__ import annotations

import itertools
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.data.skill_learning import (
    LearningAccessError,
    LearningSourceRecord,
    canonical_revision,
    owner_key_for,
)
from backend.data.skill_publication import (
    APPLIED_PENDING_DISPOSITION,
    CommitResult,
    ReviewStamp,
    VersionDraft,
)
from backend.data.skill_reviews import LearningReviewRecord, LearningRunSummary
from backend.data.skill_use import SkillSuppressionRecord, SkillUseEventRecord
from backend.data.skill_versions import (
    PENDING_WRITE_STATE,
    SkillHeadRecord,
    SkillVersionRecord,
    content_hash,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FakeLearningStore:
    def __init__(self) -> None:
        self.sources: dict[str, LearningSourceRecord] = {}
        self.reviews: dict[str, LearningReviewRecord] = {}
        self.heads: dict[str, SkillHeadRecord] = {}
        self.versions: dict[str, SkillVersionRecord] = {}
        self.suppressions: list[SkillSuppressionRecord] = []
        self.use_events: list[SkillUseEventRecord] = []
        self._clock = itertools.count()

    # ---- sources -------------------------------------------------------

    async def upsert_source_revision(
        self,
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
        rev = canonical_revision(revision)
        key = (user_id, source_kind, source_id)
        existing = next(
            (
                s
                for s in self.sources.values()
                if (s.user_id, s.source_kind, s.source_id) == key
            ),
            None,
        )
        if existing is None:
            record = LearningSourceRecord(
                id=str(uuid.uuid4()),
                user_id=user_id,
                expert_id=expert_id,
                owner_key=owner_key_for(expert_id),
                source_kind=source_kind,
                source_id=source_id,
                revision=rev,
                processed_revision=None,
                origin=origin,
                eligibility="eligible",
                epoch=0,
                approval_event_id=None,
                approval_actor_id=None,
                approved_revision=None,
                evidence_refs=list(evidence_refs),
                outcome_signals=list(outcome_signals),
                excluded_at=None,
                excluded_by_user_id=None,
                created_at=_now(),
                updated_at=_now(),
            )
            self.sources[record.id] = record
            return record
        if existing.owner_key != owner_key_for(expert_id):
            updated = existing.model_copy(
                update={"eligibility": "inaccessible", "epoch": existing.epoch + 1}
            )
            self.sources[updated.id] = updated
            return updated
        seen = {(e.get("kind"), e.get("ref")) for e in existing.evidence_refs}
        refs = list(existing.evidence_refs) + [
            e for e in evidence_refs if (e.get("kind"), e.get("ref")) not in seen
        ]
        seen_s = {(e.get("kind"), e.get("ref")) for e in existing.outcome_signals}
        sigs = list(existing.outcome_signals) + [
            e for e in outcome_signals if (e.get("kind"), e.get("ref")) not in seen_s
        ]
        updated = existing.model_copy(
            update={
                "revision": max(existing.revision, rev),
                "origin": (
                    "requested" if "requested" in (existing.origin, origin) else origin
                ),
                "evidence_refs": refs,
                "outcome_signals": sigs,
                "updated_at": _now(),
            }
        )
        self.sources[updated.id] = updated
        return updated

    async def get_source(
        self, user_id: str, source_id: str
    ) -> LearningSourceRecord | None:
        record = self.sources.get(source_id)
        return record if record and record.user_id == user_id else None

    async def get_source_by_ref(
        self, user_id: str, source_kind: str, source_ref: str
    ) -> LearningSourceRecord | None:
        return next(
            (
                s
                for s in self.sources.values()
                if s.user_id == user_id
                and s.source_kind == source_kind
                and s.source_id == source_ref
            ),
            None,
        )

    async def require_owned_source(
        self, user_id: str, source_id: str
    ) -> LearningSourceRecord:
        record = await self.get_source(user_id, source_id)
        if record is None:
            raise LearningAccessError("learning source not found for this user")
        return record

    def _pending(self, user_id: str) -> list[LearningSourceRecord]:
        rows = [
            s
            for s in self.sources.values()
            if s.user_id == user_id
            and s.eligibility == "eligible"
            and s.has_unprocessed_revision
        ]
        return sorted(rows, key=lambda s: s.updated_at)

    async def list_pending_owner_keys(self, user_id: str) -> list[str]:
        return sorted({s.owner_key for s in self._pending(user_id)})

    async def list_pending_sources(
        self, user_id: str, *, limit: int = 100, owner_key: str | None = None
    ) -> list[LearningSourceRecord]:
        rows = self._pending(user_id)
        if owner_key is not None:
            rows = [s for s in rows if s.owner_key == owner_key]
        return rows[:limit]

    async def set_source_eligibility(
        self,
        user_id: str,
        source_id: str,
        eligibility: str,
        *,
        excluded_by_user_id: str | None = None,
    ) -> LearningSourceRecord | None:
        record = await self.get_source(user_id, source_id)
        if record is None:
            return None
        updated = record.model_copy(
            update={
                "eligibility": eligibility,
                "epoch": record.epoch + 1,
                "excluded_at": (
                    _now() if eligibility == "excluded" else record.excluded_at
                ),
                "excluded_by_user_id": excluded_by_user_id,
            }
        )
        self.sources[source_id] = updated
        return updated

    async def set_source_approval(
        self,
        user_id: str,
        source_id: str,
        *,
        approval_event_id: str | None,
        approval_actor_id: str | None,
        approved_revision: str | None,
        eligibility: str,
    ) -> LearningSourceRecord | None:
        record = await self.get_source(user_id, source_id)
        if record is None:
            return None
        updated = record.model_copy(
            update={
                "approval_event_id": approval_event_id,
                "approval_actor_id": approval_actor_id,
                "approved_revision": (
                    canonical_revision(approved_revision) if approved_revision else None
                ),
                "eligibility": eligibility,
                "epoch": record.epoch + 1,
            }
        )
        self.sources[source_id] = updated
        return updated

    async def advance_source_cursor(
        self, user_id: str, source_id: str, revision: str
    ) -> bool:
        record = await self.get_source(user_id, source_id)
        rev = canonical_revision(revision)
        if record is None or (
            record.processed_revision is not None and record.processed_revision >= rev
        ):
            return False
        self.sources[source_id] = record.model_copy(update={"processed_revision": rev})
        return True

    # ---- reviews -------------------------------------------------------

    def _review_key(self, source_id: str, revision: str, policy: int) -> str:
        return f"{source_id}|{canonical_revision(revision)}|{policy}"

    async def upsert_review(
        self,
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
        source = await self.require_owned_source(user_id, source_id)
        key = self._review_key(source_id, source_revision, policy_version)
        existing = self.reviews.get(key)
        if existing is not None and existing.user_id != user_id:
            raise LearningAccessError("review belongs to another user")
        record = LearningReviewRecord(
            id=existing.id if existing else str(uuid.uuid4()),
            user_id=user_id,
            expert_id=source.expert_id,
            owner_key=source.owner_key,
            source_id=source_id,
            source_revision=canonical_revision(source_revision),
            policy_version=policy_version,
            run_id=run_id,
            disposition=disposition,
            reason=reason,
            attempts=(existing.attempts + 1) if existing else 1,
            change_fingerprint=change_fingerprint,
            skill_name=skill_name,
            applied_version_id=applied_version_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_microdollars=cost_microdollars,
            created_at=existing.created_at if existing else _now(),
            completed_at=_now() if completed else None,
        )
        self.reviews[key] = record
        return record

    async def get_review_for_revision(
        self, user_id: str, source_id: str, source_revision: str, policy_version: int
    ) -> LearningReviewRecord | None:
        record = self.reviews.get(
            self._review_key(source_id, source_revision, policy_version)
        )
        return record if record and record.user_id == user_id else None

    async def get_review(
        self, user_id: str, review_id: str
    ) -> LearningReviewRecord | None:
        return next(
            (
                r
                for r in self.reviews.values()
                if r.id == review_id and r.user_id == user_id
            ),
            None,
        )

    async def list_reviews(
        self,
        user_id: str,
        *,
        owner_key: str | None = None,
        disposition: str | None = None,
        limit: int = 50,
    ) -> list[LearningReviewRecord]:
        rows = [
            r
            for r in self.reviews.values()
            if r.user_id == user_id
            and (owner_key is None or r.owner_key == owner_key)
            and (disposition is None or r.disposition == disposition)
        ]
        return sorted(rows, key=lambda r: r.created_at, reverse=True)[:limit]

    async def summarize_learning(self, user_id: str) -> LearningRunSummary:
        pending = self._pending(user_id)
        reviews = await self.list_reviews(user_id, limit=500)
        return LearningRunSummary(
            pending_sources=len(pending),
            oldest_pending_at=min((s.updated_at for s in pending), default=None),
            last_review_at=reviews[0].created_at if reviews else None,
            last_applied_at=next(
                (r.created_at for r in reviews if r.disposition == "applied"), None
            ),
            retrying_reviews=sum(1 for r in reviews if r.completed_at is None),
            cost_microdollars_30d=sum(r.cost_microdollars or 0 for r in reviews),
        )

    # ---- heads ---------------------------------------------------------

    async def get_head(
        self, user_id: str, owner_key: str, skill_name: str
    ) -> SkillHeadRecord | None:
        return self.heads.get(f"{user_id}|{owner_key}|{skill_name}")

    async def ensure_head(
        self, user_id: str, expert_id: str | None, skill_name: str
    ) -> SkillHeadRecord:
        owner_key = owner_key_for(expert_id)
        existing = await self.get_head(user_id, owner_key, skill_name)
        if existing is not None:
            return existing
        head = SkillHeadRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            expert_id=expert_id,
            owner_key=owner_key,
            skill_name=skill_name,
            current_version=0,
            current_version_id=None,
            content_hash=None,
            auto_improve=True,
            learning_paused_at=None,
            use_paused_at=None,
            updated_at=_now(),
        )
        self.heads[f"{user_id}|{owner_key}|{skill_name}"] = head
        return head

    def _put_head(self, head: SkillHeadRecord) -> None:
        self.heads[f"{head.user_id}|{head.owner_key}|{head.skill_name}"] = head

    async def list_heads(
        self, user_id: str, owner_key: str | None = None
    ) -> list[SkillHeadRecord]:
        return [
            h
            for h in self.heads.values()
            if h.user_id == user_id and (owner_key is None or h.owner_key == owner_key)
        ]

    async def list_use_paused_skill_names(
        self, user_id: str, owner_key: str
    ) -> list[str]:
        return [
            h.skill_name
            for h in await self.list_heads(user_id, owner_key)
            if h.use_paused_at is not None
        ]

    async def update_head_policy(
        self,
        user_id: str,
        owner_key: str,
        skill_name: str,
        *,
        auto_improve: bool | None = None,
        learning_paused: bool | None = None,
        use_paused: bool | None = None,
    ) -> SkillHeadRecord | None:
        head = await self.get_head(user_id, owner_key, skill_name)
        if head is None:
            return None
        update: dict[str, Any] = {}
        if auto_improve is not None:
            update["auto_improve"] = auto_improve
        if learning_paused is not None:
            update["learning_paused_at"] = _now() if learning_paused else None
        if use_paused is not None:
            update["use_paused_at"] = _now() if use_paused else None
        head = head.model_copy(update=update)
        self._put_head(head)
        return head

    # ---- versions ------------------------------------------------------

    def _next_number(self, head_id: str) -> int:
        numbers = [v.version for v in self.versions.values() if v.head_id == head_id]
        return (max(numbers) if numbers else 0) + 1

    async def create_version(
        self,
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
        version = SkillVersionRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            expert_id=head.expert_id,
            owner_key=head.owner_key,
            skill_name=head.skill_name,
            head_id=head.id,
            version=self._next_number(head.id),
            content=content,
            content_hash=content_hash(content),
            description=description,
            triggers=list(triggers),
            origin=origin,
            actor_user_id=actor_user_id,
            summary=summary,
            base_version_id=base_version_id,
            restored_from_version_id=restored_from_version_id,
            review_id=review_id,
            sources=list(sources or []),
            evidence=list(evidence or []),
            limits=list(limits or []),
            state=state,
            state_reason=state_reason,
            blocked_pattern_class=blocked_pattern_class,
            blocked_step=blocked_step,
            created_at=_now().replace(microsecond=next(self._clock) % 1_000_000),
        )
        self.versions[version.id] = version
        return version

    async def commit_version_safe(
        self,
        user_id: str,
        *,
        head: SkillHeadRecord,
        draft: VersionDraft,
        expected_current_version: int,
        review: ReviewStamp | None = None,
        auto_improve: bool | None = None,
    ) -> CommitResult:
        live = await self.get_head(user_id, head.owner_key, head.skill_name)
        if live is None or live.current_version != expected_current_version:
            return CommitResult(
                committed=False,
                reason="the skill changed while this version was prepared",
            )
        if review is not None:
            source = await self.get_source(user_id, review.source.id)
            if source is None:
                return CommitResult(
                    committed=False, reason="source no longer accessible"
                )
            if source.eligibility != "eligible":
                return CommitResult(
                    committed=False, reason=f"source is {source.eligibility}"
                )
            if source.epoch != review.source.epoch:
                return CommitResult(
                    committed=False,
                    reason="source eligibility changed since it was reviewed",
                )
            if source.revision != canonical_revision(review.source_revision):
                return CommitResult(
                    committed=False, reason="a newer source revision exists"
                )
            settled = await self.get_review_for_revision(
                user_id, review.source.id, review.source_revision, review.policy_version
            )
            if settled is not None and settled.disposition in (
                "applied",
                APPLIED_PENDING_DISPOSITION,
            ):
                return CommitResult(
                    committed=False, reason="this revision was already applied"
                )
            if live.learning_paused_at is not None:
                return CommitResult(
                    committed=False, reason="learning paused for this skill"
                )
        version = await self.create_version(
            user_id,
            head=live,
            content=draft.content,
            description=draft.description,
            triggers=draft.triggers,
            origin=draft.origin,
            summary=draft.summary,
            state=PENDING_WRITE_STATE,
            actor_user_id=draft.actor_user_id,
            base_version_id=draft.base_version_id,
            restored_from_version_id=draft.restored_from_version_id,
            sources=draft.sources,
            evidence=draft.evidence,
            limits=draft.limits,
        )
        update: dict[str, Any] = {
            "current_version": version.version,
            "current_version_id": version.id,
            "content_hash": version.content_hash,
        }
        if auto_improve is not None:
            update["auto_improve"] = auto_improve
        self._put_head(live.model_copy(update=update))
        review_id = None
        if review is not None:
            ledger = await self.upsert_review(
                user_id,
                source_id=review.source.id,
                source_revision=review.source_revision,
                policy_version=review.policy_version,
                run_id=review.run_id,
                disposition=APPLIED_PENDING_DISPOSITION,
                reason="pointer swapped; workspace write pending",
                change_fingerprint=review.change_fingerprint,
                skill_name=head.skill_name,
                applied_version_id=version.id,
                model=review.model,
                input_tokens=review.input_tokens,
                output_tokens=review.output_tokens,
                cost_microdollars=review.cost_microdollars,
                completed=False,
            )
            review_id = ledger.id
            version = version.model_copy(update={"review_id": review_id})
            self.versions[version.id] = version
        return CommitResult(committed=True, version=version, review_id=review_id)

    def _find_review(
        self, review_id: str | None
    ) -> tuple[str, LearningReviewRecord] | None:
        if review_id is None:
            return None
        return next(
            ((k, r) for k, r in self.reviews.items() if r.id == review_id), None
        )

    async def complete_publication(
        self, user_id: str, *, version_id: str, review_id: str | None, reason: str = ""
    ) -> None:
        version = self.versions.get(version_id)
        if (
            version
            and version.user_id == user_id
            and version.state == PENDING_WRITE_STATE
        ):
            self.versions[version_id] = version.model_copy(
                update={"state": "ready", "state_reason": reason}
            )
        found = self._find_review(review_id)
        if found:
            key, review = found
            self.reviews[key] = review.model_copy(
                update={
                    "disposition": "applied",
                    "reason": reason,
                    "completed_at": _now(),
                }
            )

    async def abandon_publication(
        self, user_id: str, *, version_id: str, review_id: str | None, reason: str
    ) -> None:
        version = self.versions.get(version_id)
        if (
            version
            and version.user_id == user_id
            and version.state == PENDING_WRITE_STATE
        ):
            self.versions[version_id] = version.model_copy(
                update={"state": "stale", "state_reason": reason}
            )
        found = self._find_review(review_id)
        if found:
            key, review = found
            self.reviews[key] = review.model_copy(
                update={
                    "disposition": "conflict",
                    "reason": reason,
                    "completed_at": _now(),
                }
            )

    async def list_pending_publications(self, user_id: str) -> list[SkillVersionRecord]:
        return await self.list_versions_in_states(user_id, [PENDING_WRITE_STATE])

    async def set_version_state(
        self, user_id: str, version_id: str, state: str, reason: str = ""
    ) -> None:
        version = self.versions.get(version_id)
        if version and version.user_id == user_id:
            self.versions[version_id] = version.model_copy(
                update={"state": state, "state_reason": reason}
            )

    async def get_version(
        self, user_id: str, version_id: str
    ) -> SkillVersionRecord | None:
        version = self.versions.get(version_id)
        return version if version and version.user_id == user_id else None

    async def list_versions(
        self, user_id: str, owner_key: str, skill_name: str, *, limit: int = 50
    ) -> list[SkillVersionRecord]:
        rows = [
            v
            for v in self.versions.values()
            if v.user_id == user_id
            and v.owner_key == owner_key
            and v.skill_name == skill_name
        ]
        return sorted(rows, key=lambda v: v.version, reverse=True)[:limit]

    async def list_recent_versions(
        self,
        user_id: str,
        *,
        owner_key: str | None = None,
        origin: str | None = None,
        state: str | None = None,
        limit: int = 50,
    ) -> list[SkillVersionRecord]:
        rows = [
            v
            for v in self.versions.values()
            if v.user_id == user_id
            and (owner_key is None or v.owner_key == owner_key)
            and (origin is None or v.origin == origin)
            and (state is None or v.state == state)
        ]
        return sorted(rows, key=lambda v: v.created_at, reverse=True)[:limit]

    async def list_open_decisions(
        self, user_id: str, owner_key: str | None = None
    ) -> list[SkillVersionRecord]:
        return await self.list_recent_versions(
            user_id, owner_key=owner_key, state="needs_decision", limit=100
        )

    async def list_versions_in_states(
        self, user_id: str, states: list[str], *, limit: int = 500
    ) -> list[SkillVersionRecord]:
        rows = [
            v
            for v in self.versions.values()
            if v.user_id == user_id and v.state in states
        ]
        return sorted(rows, key=lambda v: v.created_at, reverse=True)[:limit]

    # ---- suppressions + use events -------------------------------------

    async def add_suppression(
        self,
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
        record = SkillSuppressionRecord(
            id=str(uuid.uuid4()),
            skill_name=skill_name,
            behavior_fingerprint=behavior_fingerprint,
            behavior_tokens=list(behavior_tokens),
            evidence_fingerprints=list(evidence_fingerprints),
            reason=reason,
            created_at=_now(),
        )
        self.suppressions.append(record)
        self._suppression_scope = getattr(self, "_suppression_scope", {})
        self._suppression_scope[record.id] = (user_id, owner_key_for(expert_id))
        return record

    def _scoped_suppressions(self, user_id: str, owner_key: str, skill_name: str):
        scope = getattr(self, "_suppression_scope", {})
        return [
            s
            for s in self.suppressions
            if scope.get(s.id) == (user_id, owner_key) and s.skill_name == skill_name
        ]

    async def find_suppression(
        self, user_id: str, owner_key: str, skill_name: str, behavior_fingerprint: str
    ) -> SkillSuppressionRecord | None:
        return next(
            (
                s
                for s in self._scoped_suppressions(user_id, owner_key, skill_name)
                if s.behavior_fingerprint == behavior_fingerprint
            ),
            None,
        )

    async def list_suppressions(
        self, user_id: str, owner_key: str, skill_name: str
    ) -> list[SkillSuppressionRecord]:
        return self._scoped_suppressions(user_id, owner_key, skill_name)

    async def record_use_event(
        self,
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
        event = SkillUseEventRecord(
            id=str(uuid.uuid4()),
            skill_name=skill_name,
            version_id=version_id,
            session_id=session_id,
            kind=kind,
            detail=detail,
            actor_user_id=actor_user_id,
            created_at=_now(),
        )
        self.use_events.append(event)
        return event

    async def list_use_events(
        self, user_id: str, owner_key: str, skill_name: str, *, limit: int = 200
    ) -> list[SkillUseEventRecord]:
        return [e for e in self.use_events if e.skill_name == skill_name][-limit:]
