"""Builders that turn learning records into viewer-safe API models.

Access is resolved per source: a conversation the viewer can no longer
open contributes no title, reference, or link — only the fact that a
source exists — and that never pauses the skill for anyone else.
"""

from __future__ import annotations

from backend.api.features.skill_learning.models import (
    DISPOSITION_LABELS,
    STATE_LABELS,
    LearningEvidenceLabel,
    LearningHistoryItem,
    LearningSourceLink,
    SkillUseSummary,
    SkillVersionSummary,
)
from backend.copilot import db as chat_db
from backend.copilot.learning.chat_source import CHAT_SOURCE_KIND
from backend.copilot.learning.retrieval import origin_label
from backend.data import skill_learning as learning_data
from backend.data.skill_reviews import LearningReviewRecord
from backend.data.skill_use import SkillUseEventRecord
from backend.data.skill_versions import SkillVersionRecord

MAX_SOURCE_LINKS = 10


def state_label(state: str) -> str:
    return STATE_LABELS.get(state, state.replace("_", " ").capitalize())


def disposition_label(disposition: str) -> str:
    return DISPOSITION_LABELS.get(
        disposition, disposition.replace("_", " ").capitalize()
    )


async def source_link(
    viewer_id: str, source_id: str, source_kind: str, source_ref: str, revision: str
) -> LearningSourceLink:
    record = await learning_data.get_source(viewer_id, source_id)
    hidden = LearningSourceLink(
        source_id=source_id,
        source_kind=source_kind,
        revision=revision,
        accessible=False,
    )
    if record is None or record.eligibility == "inaccessible":
        return hidden
    if source_kind != CHAT_SOURCE_KIND:
        return LearningSourceLink(
            source_id=source_id,
            source_kind=source_kind,
            revision=revision,
            accessible=True,
            source_ref=source_ref,
            excluded=record.eligibility == "excluded",
        )
    session = await chat_db.get_chat_session_metadata(source_ref)
    if session is None or session.user_id != viewer_id:
        return hidden
    url = f"/copilot?sessionId={session.session_id}"
    if session.expert_id:
        url += f"&expertId={session.expert_id}"
    return LearningSourceLink(
        source_id=source_id,
        source_kind=source_kind,
        revision=revision,
        accessible=True,
        source_ref=source_ref,
        title=session.title,
        url=url,
        excluded=record.eligibility == "excluded",
    )


async def version_summary(
    viewer_id: str,
    version: SkillVersionRecord,
    *,
    include_body: bool,
    events: list[SkillUseEventRecord] | None = None,
) -> SkillVersionSummary:
    links = [
        await source_link(
            viewer_id,
            str(dep.get("source_id", "")),
            str(dep.get("source_kind", "")),
            str(dep.get("source_ref", "")),
            str(dep.get("revision", "")),
        )
        for dep in version.sources[:MAX_SOURCE_LINKS]
    ]
    evidence = [
        LearningEvidenceLabel(
            kind=str(e.get("kind", "")),
            label=str(e.get("label", "")),
            ref=str(e.get("ref", "")),
        )
        for e in version.evidence
    ]
    return SkillVersionSummary(
        id=version.id,
        skill_name=version.skill_name,
        expert_id=version.expert_id,
        version=version.version,
        origin=version.origin,
        origin_label=origin_label(
            version.origin, viewer_is_actor=version.actor_user_id == viewer_id
        ),
        summary=version.summary,
        state=version.state,
        state_label=state_label(version.state),
        state_reason=version.state_reason,
        created_at=version.created_at,
        description=version.description,
        triggers=version.triggers,
        body=version.content if include_body and version.content else None,
        evidence=evidence,
        limits=version.limits,
        sources=links,
        use=use_summary(events or [], version.id),
        base_version_id=version.base_version_id,
        restored_from_version_id=version.restored_from_version_id,
        blocked_pattern_class=version.blocked_pattern_class,
        blocked_step=version.blocked_step,
    )


def use_summary(
    events: list[SkillUseEventRecord], version_id: str | None
) -> SkillUseSummary:
    """Counts for one exact version; events on other versions never count."""
    counts = SkillUseSummary(version_id=version_id)
    for event in events:
        if event.version_id != version_id:
            continue
        if event.kind == "loaded":
            counts.loads += 1
        elif event.kind == "check_passed":
            counts.checks_passed += 1
        elif event.kind == "check_failed":
            counts.checks_failed += 1
        elif event.kind == "succeeded":
            counts.reported_working += 1
        elif event.kind == "failed":
            counts.reported_failed += 1
        elif event.kind == "stopped_mid_use":
            counts.stopped_mid_use += 1
        else:
            counts.unknown += 1
    counts.reuse_label = _reuse_label(counts)
    return counts


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _reuse_label(counts: SkillUseSummary) -> str:
    if counts.checks_passed:
        return f"Passed checks in {_plural(counts.checks_passed, 'later use')}"
    if counts.checks_failed:
        return f"Failed checks in {_plural(counts.checks_failed, 'later use')}"
    if counts.reported_working:
        return (
            f"Reported as working by you ({_plural(counts.reported_working, 'report')})"
        )
    if counts.reported_failed:
        return (
            f"Reported as failed by you ({_plural(counts.reported_failed, 'report')})"
        )
    if counts.stopped_mid_use:
        return "Stopped mid-use"
    if counts.loads:
        return f"Loaded {_plural(counts.loads, 'time')} · Outcome unknown"
    return "Not yet reused"


async def history_item_from_version(
    viewer_id: str, version: SkillVersionRecord
) -> LearningHistoryItem:
    summary = await version_summary(viewer_id, version, include_body=False)
    return LearningHistoryItem(
        id=version.id,
        kind="version",
        skill_name=version.skill_name,
        expert_id=version.expert_id,
        created_at=version.created_at,
        summary=version.summary or version.description,
        origin=version.origin,
        origin_label=summary.origin_label,
        state=version.state,
        state_label=summary.state_label,
        version=version.version,
        version_id=version.id,
        source=summary.sources[0] if summary.sources else None,
    )


async def history_item_from_review(
    viewer_id: str, review: LearningReviewRecord
) -> LearningHistoryItem:
    record = await learning_data.get_source(viewer_id, review.source_id)
    link = (
        await source_link(
            viewer_id,
            review.source_id,
            record.source_kind,
            record.source_id,
            review.source_revision,
        )
        if record is not None
        else None
    )
    return LearningHistoryItem(
        id=review.id,
        kind="review",
        skill_name=review.skill_name,
        expert_id=review.expert_id,
        created_at=review.created_at,
        summary=review.reason or disposition_label(review.disposition),
        state=review.disposition,
        state_label=disposition_label(review.disposition),
        version_id=review.applied_version_id,
        source=link,
    )
