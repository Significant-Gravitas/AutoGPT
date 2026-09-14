"""Publication engine: turns a validated proposal into a ready skill version.

Every path ends in the same two-phase commit:

1. ``commit_version`` (data layer, one transaction): append a
   ``pending_write`` version, compare-and-swap the head pointer, and stamp
   the review ledger ``applied_pending``.
2. Write the workspace ``SKILL.md`` (the registry the copilot reads), then
   ``complete_publication`` marks the version ready and the ledger applied.

A crash between the phases is recovered by :func:`reconcile_pending`
without a second model call: the pointer already names the version, so
the write is simply repeated. Before phase 1 the engine re-checks live
source eligibility, content checks, suppression, and the skill's own
policy — a queued proposal never publishes on a stale snapshot.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from backend.copilot.tools.skills import (
    ExpectedHead,
    ParsedSkill,
    SkillContentBlockedError,
    SkillVersionConflictError,
    canonicalize_skill,
    invalidate_skills_index_cache,
    parse_skill_markdown,
    read_skill_bundle_files,
    read_user_skill_markdown,
    render_skill_markdown,
    store_user_skill,
)
from backend.data.db_accessors import (
    skill_publication_db,
    skill_use_db,
    skill_versions_db,
)
from backend.data.skill_learning import owner_key_for
from backend.data.skill_publication import ReviewStamp, VersionDraft
from backend.data.skill_versions import (
    SkillHeadRecord,
    SkillVersionRecord,
    content_hash,
)

from .content_checks import check_metadata, check_skill_bundle
from .contract import (
    ApprovalCheckpoint,
    LearningScope,
    check_approval_precondition,
    get_source_adapter,
)
from .fingerprint import (
    UNCERTAIN_EQUIVALENCE_THRESHOLD,
    behavior_fingerprint,
    behavior_tokens,
    token_overlap,
)

logger = logging.getLogger(__name__)

PublishStatus = Literal[
    "applied",
    "needs_decision",
    "conflict",
    "stale_eligibility",
    "blocked_content",
    "suppressed",
    "paused",
    "write_failed",
]


class SourceSnapshot(BaseModel):
    """The eligibility snapshot a proposal was reviewed under."""

    source_id: str
    source_kind: str
    source_ref: str
    revision: str
    # ``None`` for a snapshot rebuilt from stored version dependencies: the
    # live epoch is then adopted and only state + revision are checked.
    epoch: int | None
    approval: ApprovalCheckpoint | None = None

    def as_dependency(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "revision": self.revision,
            "epoch": self.epoch,
        }


class PublishRequest(BaseModel):
    user_id: str
    expert_id: str | None
    skill_name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    body: str
    summary: str
    origin: str
    sources: list[SourceSnapshot] = Field(default_factory=list)
    evidence: list[dict[str, str]] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    supported_refs: list[str] = Field(default_factory=list)
    review: ReviewStamp | None = None


class PublishOutcome(BaseModel):
    status: PublishStatus
    version: SkillVersionRecord | None = None
    reason: str = ""
    pattern_class: str | None = None
    blocked_step: str | None = None


# ---------------------------------------------------------------------------
# Automated publication (nightly / requested)
# ---------------------------------------------------------------------------


async def publish_learned_version(request: PublishRequest) -> PublishOutcome:
    """Publish an automated change, or file it as a proposal when the skill
    is human-controlled. Every precondition is re-checked live.

    Model-derived metadata (name, description, triggers, summary) is checked
    before a head or any record exists under those values; a blocked
    attempt persists only the pattern class and an ordinal location.
    """
    metadata_failure = check_metadata(
        {
            "name": request.skill_name,
            "description": request.description,
            "triggers": " ".join(request.triggers),
            "summary": request.summary,
            "limits": " ".join(request.limits),
        }
    )
    if metadata_failure is not None:
        return PublishOutcome(
            status="blocked_content",
            reason=metadata_failure.describe(),
            pattern_class=metadata_failure.pattern_class,
            blocked_step=metadata_failure.step,
        )
    parsed = canonicalize_skill(
        ParsedSkill(
            name=request.skill_name,
            description=request.description,
            body=request.body,
            triggers=tuple(request.triggers),
        )
    )
    rendered = render_skill_markdown(parsed)
    blocked = await _check_bundle(request, rendered)
    if blocked is not None:
        return blocked

    versions = skill_versions_db()
    head = await versions.ensure_head(
        request.user_id, request.expert_id, request.skill_name
    )
    if head.learning_paused_at is not None:
        return PublishOutcome(status="paused", reason="learning paused for this skill")

    suppression = await _check_suppression(request, head)
    if suppression is not None and suppression.status == "suppressed":
        return suppression

    stale = await require_fresh_eligibility(request)
    if stale is not None:
        return stale

    drifted = await _record_untracked_edit(request, head)
    if drifted is not None:
        return drifted

    draft = VersionDraft(
        content=rendered,
        description=request.description,
        triggers=request.triggers,
        origin=request.origin,
        summary=request.summary,
        base_version_id=head.current_version_id,
        sources=[s.as_dependency() for s in request.sources],
        evidence=request.evidence,
        limits=request.limits,
    )
    if not head.auto_improve or suppression is not None:
        return await _file_proposal(request, head, draft, suppression)
    return await commit_and_write(request.user_id, head, draft, request.review)


async def _check_bundle(
    request: PublishRequest, rendered: str
) -> PublishOutcome | None:
    bundle = await read_skill_bundle_files(
        request.user_id, request.skill_name, expert_id=request.expert_id
    )
    bundle["SKILL.md"] = rendered
    failure = check_skill_bundle(bundle)
    if failure is None:
        return None
    # Record the blocked attempt without any rejected content: history can
    # show the pattern class and ordinal step while no body, description,
    # trigger, or summary from the rejected output is ever persisted. The
    # metadata passed ``check_metadata`` above, so the head name is safe.
    head = await skill_versions_db().ensure_head(
        request.user_id, request.expert_id, request.skill_name
    )
    version = await skill_versions_db().create_version(
        request.user_id,
        head=head,
        content="",
        description="Blocked by content check",
        triggers=[],
        origin=request.origin,
        summary="Blocked by content check",
        state="blocked_content",
        state_reason=failure.describe(),
        base_version_id=head.current_version_id,
        sources=[s.as_dependency() for s in request.sources],
        blocked_pattern_class=failure.pattern_class,
        blocked_step=f"{failure.file}: {failure.step}",
    )
    return PublishOutcome(
        status="blocked_content",
        version=version,
        reason=failure.describe(),
        pattern_class=failure.pattern_class,
        blocked_step=failure.step,
    )


async def _check_suppression(
    request: PublishRequest, head: SkillHeadRecord
) -> PublishOutcome | None:
    """Exact match → suppressed; uncertain overlap → must become a proposal."""
    fingerprint = behavior_fingerprint(request.skill_name, request.body)
    exact = await skill_use_db().find_suppression(
        request.user_id, head.owner_key, request.skill_name, fingerprint
    )
    if exact is not None:
        return PublishOutcome(
            status="suppressed",
            reason="this behaviour was removed by the owner; not reapplied",
        )
    tokens = behavior_tokens(request.body)
    for record in await skill_use_db().list_suppressions(
        request.user_id, head.owner_key, request.skill_name
    ):
        if (
            token_overlap(tokens, record.behavior_tokens)
            >= UNCERTAIN_EQUIVALENCE_THRESHOLD
        ):
            return PublishOutcome(
                status="needs_decision",
                reason="closely resembles a change the owner removed; needs a decision",
            )
    return None


async def require_fresh_eligibility(request: PublishRequest) -> PublishOutcome | None:
    scope = LearningScope(
        user_id=request.user_id,
        expert_id=request.expert_id,
        owner_key=owner_key_for(request.expert_id),
    )
    for snapshot in request.sources:
        adapter = get_source_adapter(snapshot.source_kind)
        if adapter is None:
            return PublishOutcome(
                status="stale_eligibility", reason="unknown source kind"
            )
        eligibility = await adapter.revalidate(
            source_id=snapshot.source_id,
            revision=snapshot.revision,
            scope=scope,
            approval_event_id=(
                snapshot.approval.event_id if snapshot.approval else None
            ),
        )
        pinned = (
            snapshot
            if snapshot.epoch is not None
            else snapshot.model_copy(update={"epoch": eligibility.epoch})
        )
        problem = check_approval_precondition(
            adapter, eligibility, _as_source_revision(pinned, scope)
        )
        if problem is not None:
            return PublishOutcome(status="stale_eligibility", reason=problem)
    return None


def _as_source_revision(snapshot: SourceSnapshot, scope: LearningScope):
    from .contract import SourceRevision

    return SourceRevision(
        source_id=snapshot.source_id,
        source_kind=snapshot.source_kind,
        source_ref=snapshot.source_ref,
        scope=scope,
        revision=snapshot.revision,
        epoch=snapshot.epoch if snapshot.epoch is not None else -1,
        approval=snapshot.approval,
    )


async def _record_untracked_edit(
    request: PublishRequest, head: SkillHeadRecord
) -> PublishOutcome | None:
    """A workspace SKILL.md whose bytes no longer match the head is a human
    edit that bypassed version tracking: record it and report a conflict."""
    if head.content_hash is None:
        return None
    raw = await read_user_skill_markdown(
        request.user_id, request.skill_name, expert_id=request.expert_id
    )
    if raw is None or content_hash(raw) == head.content_hash:
        return None
    current = parse_skill_markdown(raw, fallback_name=request.skill_name)
    if current is None:
        return None
    await store_user_skill(
        request.user_id,
        name=current.name,
        description=current.description,
        body=current.body,
        triggers=list(current.triggers),
        version=current.version,
        expert_id=request.expert_id,
        version_origin="edited",
        actor_user_id=request.user_id,
        summary="Edited outside version tracking",
    )
    return PublishOutcome(
        status="conflict",
        reason="the skill was edited concurrently; the edit was kept",
    )


async def _file_proposal(
    request: PublishRequest,
    head: SkillHeadRecord,
    draft: VersionDraft,
    uncertain: PublishOutcome | None,
) -> PublishOutcome:
    """One open proposal per skill: a newer one supersedes the older."""
    versions = skill_versions_db()
    for open_version in await versions.list_open_decisions(
        request.user_id, head.owner_key
    ):
        if open_version.skill_name == head.skill_name:
            await versions.set_version_state(
                request.user_id,
                open_version.id,
                "archived",
                "superseded by a newer proposal",
            )
    reason = (
        uncertain.reason
        if uncertain is not None
        else "automatic improvements are off for this skill"
    )
    version = await versions.create_version(
        request.user_id,
        head=head,
        content=draft.content,
        description=draft.description,
        triggers=draft.triggers,
        origin=draft.origin,
        summary=draft.summary,
        state="needs_decision",
        state_reason=reason,
        base_version_id=head.current_version_id,
        sources=draft.sources,
        evidence=draft.evidence,
        limits=draft.limits,
    )
    return PublishOutcome(status="needs_decision", version=version, reason=reason)


async def commit_and_write(
    user_id: str,
    head: SkillHeadRecord,
    draft: VersionDraft,
    review: ReviewStamp | None,
    *,
    auto_improve: bool | None = None,
) -> PublishOutcome:
    result = await skill_publication_db().commit_version_safe(
        user_id,
        head=head,
        draft=draft,
        expected_current_version=head.current_version,
        review=review,
        auto_improve=auto_improve,
    )
    if not result.committed or result.version is None:
        return PublishOutcome(status="conflict", reason=result.reason)
    return await write_committed_version(user_id, result.version, result.review_id)


async def write_committed_version(
    user_id: str, version: SkillVersionRecord, review_id: str | None
) -> PublishOutcome:
    """Phase 2: the workspace write. Idempotent, so reconcile can repeat it.

    The write is conditional on the head still naming this version, checked
    inside the registry's per-owner write lock: a publication resumed after
    a newer human correction is abandoned instead of putting its older
    bytes on top of that correction.
    """
    parsed = parse_skill_markdown(version.content, fallback_name=version.skill_name)
    if parsed is None:
        await skill_publication_db().abandon_publication(
            user_id, version_id=version.id, review_id=review_id, reason="unparseable"
        )
        return PublishOutcome(
            status="write_failed", reason="stored content unparseable"
        )
    try:
        await store_user_skill(
            user_id,
            name=parsed.name,
            description=parsed.description,
            body=parsed.body,
            triggers=list(parsed.triggers),
            version=parsed.version,
            expert_id=version.expert_id,
            version_origin=None,
            expected_head=ExpectedHead(version.id),
        )
    except SkillContentBlockedError as exc:
        await skill_publication_db().abandon_publication(
            user_id, version_id=version.id, review_id=review_id, reason=str(exc)
        )
        return PublishOutcome(status="blocked_content", reason=str(exc))
    except SkillVersionConflictError:
        reason = "superseded by a newer version before its workspace write"
        await skill_publication_db().abandon_publication(
            user_id, version_id=version.id, review_id=review_id, reason=reason
        )
        return PublishOutcome(status="conflict", reason=reason)
    except Exception as exc:
        logger.warning(
            "Skill workspace write failed for %s v%s (user %s); left pending",
            version.skill_name,
            version.version,
            user_id[:12],
            exc_info=True,
        )
        return PublishOutcome(status="write_failed", reason=str(exc)[:200])
    await skill_publication_db().complete_publication(
        user_id, version_id=version.id, review_id=review_id, reason=version.summary
    )
    await invalidate_skills_index_cache(user_id, version.expert_id)
    return PublishOutcome(status="applied", version=version, reason=version.summary)


async def reconcile_pending(user_id: str) -> int:
    """Finish (or abandon) publications interrupted after the pointer swap."""
    versions = skill_versions_db()
    completed = 0
    for pending in await skill_publication_db().list_pending_publications(user_id):
        head = await versions.get_head(user_id, pending.owner_key, pending.skill_name)
        if head is None or head.current_version_id != pending.id:
            await skill_publication_db().abandon_publication(
                user_id,
                version_id=pending.id,
                review_id=pending.review_id,
                reason="superseded before its workspace write completed",
            )
            continue
        outcome = await write_committed_version(user_id, pending, pending.review_id)
        if outcome.status == "applied":
            completed += 1
    return completed
