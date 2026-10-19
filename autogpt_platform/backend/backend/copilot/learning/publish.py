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

from backend.api.features.store.exceptions import VirusDetectedError
from backend.copilot.tools.skills import (
    ExpectedHead,
    ParsedSkill,
    SkillContentBlockedError,
    SkillFile,
    SkillPackage,
    SkillPackageError,
    SkillVersionConflictError,
    canonicalize_skill,
    invalidate_skills_index_cache,
    parse_skill_markdown,
    read_user_skill_files,
    read_user_skill_markdown,
    render_skill_markdown,
    saved_skill_limit_reached,
    store_user_skill,
    validate_package,
)
from backend.data.db_accessors import (
    skill_publication_db,
    skill_use_db,
    skill_versions_db,
)
from backend.data.skill_capacity import MAX_SKILLS_PER_EXPERT, SkillLimitError
from backend.data.skill_content_approvals import SkillContentApproval
from backend.data.skill_learning import owner_key_for
from backend.data.skill_publication import ReviewStamp, VersionDraft
from backend.data.skill_version_files import version_package_hash
from backend.data.skill_versions import (
    SkillHeadRecord,
    SkillVersionRecord,
    content_hash,
)

from .content_approvals import current_content_approvals
from .content_checks import (
    check_metadata,
    check_skill_bundle,
    collect_content_approvals,
    safe_diagnostic,
)
from .fingerprint import (
    UNCERTAIN_EQUIVALENCE_THRESHOLD,
    behavior_fingerprint,
    behavior_tokens,
    token_overlap,
)
from .history import record_registry_write
from .packages import restore_files, snapshot_files
from .publication_eligibility import require_fresh_eligibility
from .publication_models import PublishOutcome, PublishRequest
from .publication_models import PublishStatus as PublishStatus
from .publication_models import SourceSnapshot as SourceSnapshot
from .publication_state import require_settled_head
from .publication_validation import (
    _validate_request,
    _validate_skill_metadata,
    _validate_version_content,
)

logger = logging.getLogger(__name__)

SKILL_LIMIT_REASON = (
    f"the skill limit is reached ({MAX_SKILLS_PER_EXPERT} saved skills); "
    "delete a skill you no longer use to make room"
)

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
    invalid = _validate_request(request)
    if invalid is not None:
        return invalid
    unsettled = await require_settled_head(
        request.user_id,
        await skill_versions_db().get_head(
            request.user_id, owner_key_for(request.expert_id), request.skill_name
        ),
    )
    if unsettled is not None:
        return unsettled
    try:
        current_text = await read_user_skill_markdown(
            request.user_id,
            request.skill_name,
            expert_id=request.expert_id,
            strict=True,
        )
    except UnicodeDecodeError:
        return PublishOutcome(
            status="invalid_proposal", reason="existing skill is not valid UTF-8"
        )
    current = parse_skill_markdown(current_text) if current_text else None
    if request.create_only and current_text is not None:
        return PublishOutcome(
            status="conflict", reason="the target skill already exists"
        )
    try:
        current_files = await read_user_skill_files(
            request.user_id, request.skill_name, expert_id=request.expert_id
        )
    except SkillPackageError as exc:
        return PublishOutcome(
            status="invalid_proposal", reason=safe_diagnostic(str(exc))
        )
    if (
        request.expected_package_hash is not None
        and version_package_hash(current_text or "", snapshot_files(current_files))
        != request.expected_package_hash
    ):
        return PublishOutcome(
            status="conflict", reason="the skill package changed after review"
        )
    approvals: list[SkillContentApproval] = []
    if current_text:
        previous_bundle = {
            f.relative_path: f.content.decode("utf-8", errors="replace")
            for f in current_files
        }
        previous_bundle["SKILL.md"] = current_text
        metadata_failure = check_metadata(
            {str(i): f.relative_path for i, f in enumerate(current_files)}
        )
        failure = metadata_failure or check_skill_bundle(previous_bundle)
        if failure is not None and metadata_failure is None:
            approvals = await current_content_approvals(
                request.user_id, request.expert_id, request.skill_name
            )
            failure = check_skill_bundle(previous_bundle, approved_matches=approvals)
        if failure is not None:
            return PublishOutcome(
                status="blocked_content",
                reason=failure.describe(),
                pattern_class=failure.pattern_class,
                blocked_step=failure.step,
            )
    parsed = canonicalize_skill(
        ParsedSkill(
            name=request.skill_name,
            description=request.description,
            body=request.body,
            triggers=tuple(request.triggers),
            version=current.version if current else None,
            extra=current.extra if current else {},
        )
    )
    rendered = render_skill_markdown(parsed)
    files = current_files if request.files is None else request.files
    try:
        validate_package(SkillPackage(skill_md=rendered, files=files))
    except ValueError as exc:
        return PublishOutcome(
            status="invalid_proposal", reason=safe_diagnostic(str(exc))
        )
    blocked = await _check_bundle(request, rendered, files, approvals)
    if blocked is not None:
        return blocked
    if await saved_skill_limit_reached(
        request.user_id, request.expert_id, request.skill_name
    ):
        # Checked before anything is written, so a skill that would need a
        # saved slot the owner no longer has (a new one, or an installed copy
        # an edit moves into the saved budget) leaves nothing behind. Nightly
        # keeps the paid review for its next retry.
        return PublishOutcome(status="paused", reason=SKILL_LIMIT_REASON)

    versions = skill_versions_db()
    head = await versions.ensure_head(
        request.user_id, request.expert_id, request.skill_name
    )
    if request.create_only and head.current_version_id is not None:
        return PublishOutcome(
            status="conflict", reason="the target skill already has a version"
        )
    if head.deleted and current_text is not None:
        # This run read the package before its owner deleted it; importing
        # those bytes now would bring the deleted skill back.
        return PublishOutcome(
            status="conflict", reason="the skill was deleted after it was read"
        )
    if head.current_version_id is None and current is not None and current_text:
        await record_registry_write(
            request.user_id,
            expert_id=request.expert_id,
            skill_name=request.skill_name,
            rendered=current_text,
            description=current.description,
            triggers=list(current.triggers),
            origin="imported",
            actor_user_id=None,
            summary="Existing skill preserved before learning",
            keep_auto_improve=False,
            files=snapshot_files(current_files),
            only_if_head_at=head.current_version,
        )
        head = await versions.ensure_head(
            request.user_id, request.expert_id, request.skill_name
        )
        if head.current_version_id is None:
            return PublishOutcome(
                status="write_failed",
                reason="existing skill history could not be preserved",
            )
    if head.learning_paused_at is not None:
        return PublishOutcome(status="paused", reason="learning paused for this skill")

    suppression = await _check_suppression(request, head, files)
    if suppression is not None and suppression.status == "suppressed":
        return suppression

    stale = await require_fresh_eligibility(request, retry_changed_source=True)
    if stale is not None:
        return stale

    drifted = await _record_untracked_edit(request, head)
    if drifted is not None:
        return drifted

    draft = VersionDraft(
        content=rendered,
        files=snapshot_files(files),
        description=request.description,
        triggers=request.triggers,
        origin=request.origin,
        summary=request.summary,
        base_version_id=head.current_version_id,
        sources=[s.as_dependency() for s in request.sources],
        evidence=request.evidence,
        limits=request.limits,
        content_approvals=collect_content_approvals(
            {
                "SKILL.md": rendered,
                **{
                    f.relative_path: f.content.decode("utf-8", errors="replace")
                    for f in files
                },
            },
            approved_by_user_id=request.user_id,
            approved_matches=approvals,
        ),
    )
    if not head.auto_improve or suppression is not None:
        return await _file_proposal(request, head, draft, suppression)
    return await commit_and_write(request.user_id, head, draft, request.review)


async def _check_bundle(
    request: PublishRequest,
    rendered: str,
    files: list[SkillFile],
    content_approvals: list[SkillContentApproval] | None = None,
) -> PublishOutcome | None:
    bundle = {
        f.relative_path: f.content.decode("utf-8", errors="replace") for f in files
    }
    failure = check_metadata({str(i): f.relative_path for i, f in enumerate(files)})
    bundle["SKILL.md"] = rendered
    failure = failure or check_skill_bundle(
        bundle, approved_matches=content_approvals or []
    )
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
    request: PublishRequest, head: SkillHeadRecord, files: list[SkillFile]
) -> PublishOutcome | None:
    """Exact match → suppressed; uncertain overlap → must become a proposal."""
    fingerprint = behavior_fingerprint(
        request.skill_name, request.body, snapshot_files(files)
    )
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


async def _record_untracked_edit(
    request: PublishRequest, head: SkillHeadRecord
) -> PublishOutcome | None:
    """A workspace SKILL.md whose bytes no longer match the head is a human
    edit that bypassed version tracking: record it and report a conflict."""
    unsettled = await require_settled_head(request.user_id, head)
    if unsettled is not None:
        return unsettled
    if head.content_hash is None:
        return None
    raw = await read_user_skill_markdown(
        request.user_id, request.skill_name, expert_id=request.expert_id, strict=True
    )
    if raw is None:
        return None
    files = snapshot_files(
        await read_user_skill_files(
            request.user_id, request.skill_name, expert_id=request.expert_id
        )
    )
    package_hash = version_package_hash(raw, files)
    if (
        request.expected_package_hash is not None
        and package_hash != request.expected_package_hash
    ):
        return PublishOutcome(
            status="conflict", reason="the skill package changed after review"
        )
    tracked = (
        await skill_versions_db().get_version(request.user_id, head.current_version_id)
        if head.current_version_id
        else None
    )
    files_match = (
        tracked is None
        or tracked.files is None
        or version_package_hash(tracked.content, tracked.files) == package_hash
    )
    if content_hash(raw) == head.content_hash and files_match:
        return None
    current = parse_skill_markdown(raw, fallback_name=request.skill_name)
    if current is None:
        return None
    current = canonicalize_skill(current)
    invalid = _validate_skill_metadata(current)
    if invalid is not None:
        return invalid
    try:
        await store_user_skill(
            request.user_id,
            name=current.name,
            description=current.description,
            body=current.body,
            triggers=list(current.triggers),
            version=current.version,
            extra=current.extra,
            expert_id=request.expert_id,
            version_origin="edited",
            actor_user_id=request.user_id,
            summary="Edited outside version tracking",
            expected_head=ExpectedHead(version_id=head.current_version_id),
        )
    except SkillVersionConflictError:
        return PublishOutcome(
            status="conflict", reason="the skill changed during review"
        )
    except SkillContentBlockedError as exc:
        return PublishOutcome(
            status="blocked_content",
            reason=str(exc),
            pattern_class=exc.failure.pattern_class,
            blocked_step=exc.failure.step,
        )
    except VirusDetectedError:
        return PublishOutcome(
            status="blocked_content", reason="skill package rejected by malware scan"
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
        files=draft.files,
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
        content_approvals=draft.content_approvals,
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
    draft = draft.model_copy(
        update={
            "content_approvals": collect_content_approvals(
                {
                    "SKILL.md": draft.content,
                    **{
                        f.relative_path: f.content.decode("utf-8", errors="replace")
                        for f in draft.files or []
                    },
                },
                approved_by_user_id=user_id,
                approved_matches=draft.content_approvals,
            )
        }
    )
    invalid = _validate_version_content(
        draft.content, draft.files, draft.content_approvals
    )
    if invalid is not None:
        return invalid
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
    version = version.model_copy(
        update={
            "content_approvals": [
                a for a in version.content_approvals if a.approved_by_user_id == user_id
            ]
        }
    )
    invalid = _validate_version_content(
        version.content, version.files, version.content_approvals
    )
    if invalid is not None:
        await skill_publication_db().abandon_publication(
            user_id,
            version_id=version.id,
            review_id=review_id,
            reason=invalid.reason,
            disposition="skipped",
        )
        return invalid
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
            extra=parsed.extra,
            expert_id=version.expert_id,
            version_origin=None,
            expected_head=ExpectedHead(version_id=version.id),
            files=restore_files(version.files) if version.files is not None else None,
            content_approvals=version.content_approvals,
        )
    except (SkillContentBlockedError, VirusDetectedError):
        reason = "skill package rejected by content or malware scan"
        await skill_publication_db().abandon_publication(
            user_id,
            version_id=version.id,
            review_id=review_id,
            reason=reason,
            disposition="blocked_content",
        )
        return PublishOutcome(status="blocked_content", reason=reason)
    except SkillVersionConflictError:
        reason = "superseded by a newer version before its workspace write"
        await skill_publication_db().abandon_publication(
            user_id, version_id=version.id, review_id=review_id, reason=reason
        )
        return PublishOutcome(status="conflict", reason=reason)
    except SkillLimitError:
        # The limit was reached after the precheck; the version stays pending
        # and is written once the owner frees a slot.
        return PublishOutcome(status="paused", reason=SKILL_LIMIT_REASON)
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
        snapshot = await versions.get_version(user_id, pending.id)
        if snapshot is None or snapshot.state != "pending_write":
            continue
        outcome = await write_committed_version(user_id, snapshot, snapshot.review_id)
        if outcome.status == "applied":
            completed += 1
    return completed


async def reconcile_skill_for_read(
    user_id: str, expert_id: str | None, skill_name: str
) -> None:
    from backend.data.skill_learning import owner_key_for

    pending = await skill_publication_db().get_pending_publication_for_skill(
        user_id, owner_key_for(expert_id), skill_name
    )
    if pending is not None:
        await write_committed_version(user_id, pending, pending.review_id)
