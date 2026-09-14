"""Owner actions on learned skills: edit now, restore a version, decide.

Every action ends in the same transactional commit as automated
publication (:mod:`publish`), so an owner's change can never race an
overnight change into a lost update.
"""

from __future__ import annotations

import logging
from typing import Literal

from backend.copilot.tools.skills import (
    ExpectedHead,
    ParsedSkill,
    SkillContentBlockedError,
    SkillVersionConflictError,
    SkillWriteLockError,
    canonicalize_skill,
    parse_skill_markdown,
    read_user_skill_with_body,
    render_skill_markdown,
    store_user_skill,
)
from backend.data.db_accessors import skill_use_db, skill_versions_db
from backend.data.skill_learning import owner_key_for
from backend.data.skill_publication import VersionDraft
from backend.data.skill_versions import SkillVersionRecord

from .contract import LearningScope, get_source_adapter
from .fingerprint import behavior_fingerprint, behavior_tokens, evidence_fingerprint
from .publish import (
    PublishOutcome,
    PublishRequest,
    SourceSnapshot,
    commit_and_write,
    require_fresh_eligibility,
)

logger = logging.getLogger(__name__)


async def apply_owner_edit(
    *,
    user_id: str,
    expert_id: str | None,
    skill_name: str,
    description: str,
    body: str,
    triggers: list[str],
    keep_auto_improve: bool,
    allowed_pattern_classes: list[str],
    expected_version_id: str | None,
) -> PublishOutcome:
    """An owner's correction takes effect immediately (same content checks).

    ``expected_version_id`` is the current version the editor saw when
    editing began (``None`` only for a skill with no tracked version). It
    is checked inside the registry write lock, so a change that landed
    while the owner was typing — overnight, or from another tab — turns
    this submission into a conflict with the draft kept, never a silent
    overwrite of that change.
    """
    current = await read_user_skill_with_body(user_id, skill_name, expert_id=expert_id)
    previous = await skill_versions_db().get_head(
        user_id, owner_key_for(expert_id), skill_name
    )
    try:
        await store_user_skill(
            user_id,
            name=skill_name,
            description=description,
            body=body,
            triggers=triggers,
            version=current.version if current else None,
            expert_id=expert_id,
            version_origin="edited",
            actor_user_id=user_id,
            summary="Edited by the owner",
            keep_auto_improve=keep_auto_improve,
            allowed_pattern_classes=allowed_pattern_classes,
            expected_head=ExpectedHead(version_id=expected_version_id),
        )
    except SkillContentBlockedError as exc:
        return PublishOutcome(
            status="blocked_content",
            reason=str(exc),
            pattern_class=exc.failure.pattern_class,
            blocked_step=exc.failure.step,
        )
    except SkillVersionConflictError:
        return PublishOutcome(
            status="conflict",
            reason="the skill changed since you started editing; "
            "review the current version, then apply your draft again",
        )
    except SkillWriteLockError as exc:
        return PublishOutcome(status="write_failed", reason=str(exc))
    if previous is not None and previous.current_version_id:
        replaced = await skill_versions_db().get_version(
            user_id, previous.current_version_id
        )
        if replaced is not None:
            await _suppress_replaced_behavior(
                user_id, expert_id, replaced, body, reason="edited by the owner"
            )
    resolved = await skill_versions_db().get_head(
        user_id, owner_key_for(expert_id), skill_name
    )
    version = (
        await skill_versions_db().get_version(user_id, resolved.current_version_id)
        if resolved and resolved.current_version_id
        else None
    )
    return PublishOutcome(status="applied", version=version, reason="edit applied")


async def restore_version(
    *,
    user_id: str,
    expert_id: str | None,
    skill_name: str,
    version_id: str,
    actor_user_id: str,
    reason: str = "Restored by the owner",
) -> PublishOutcome:
    """Restore prior content as a new version and suppress what it replaces.

    Restoration keeps the audit trail (a new version referring to the old
    one), affects future use only, and — as the first-release boundary —
    turns automatic improvements off for the skill so the undone change
    cannot reappear without an explicit decision.
    """
    versions = skill_versions_db()
    target = await versions.get_version(user_id, version_id)
    if (
        target is None
        or target.skill_name != skill_name
        or target.owner_key != owner_key_for(expert_id)
    ):
        return PublishOutcome(status="conflict", reason="version not found")
    if target.state in ("invalidated", "blocked_content") or not target.content:
        return PublishOutcome(
            status="stale_eligibility", reason="that version can no longer be used"
        )
    unavailable = await _unavailable_sources(user_id, expert_id, target)
    if unavailable:
        return PublishOutcome(status="stale_eligibility", reason=unavailable)
    head = await versions.get_head(user_id, owner_key_for(expert_id), skill_name)
    if head is None:
        return PublishOutcome(status="conflict", reason="skill has no version history")
    replaced = (
        await versions.get_version(user_id, head.current_version_id)
        if head.current_version_id
        else None
    )
    draft = VersionDraft(
        content=target.content,
        description=target.description,
        triggers=target.triggers,
        origin="restored",
        summary=f"{reason} (from v{target.version})",
        actor_user_id=actor_user_id,
        base_version_id=head.current_version_id,
        restored_from_version_id=target.id,
        sources=target.sources,
        evidence=target.evidence,
        limits=target.limits,
    )
    outcome = await commit_and_write(user_id, head, draft, None, auto_improve=False)
    if outcome.status == "applied" and replaced is not None:
        replaced_body = _body_of(replaced)
        await _suppress_replaced_behavior(
            user_id, expert_id, replaced, target.content, reason="undone by restore"
        )
        logger.info(
            "Restored %s to v%s for user %s (replaced v%s, %d chars)",
            skill_name,
            target.version,
            user_id[:12],
            replaced.version,
            len(replaced_body),
        )
    return outcome


async def decide_proposal(
    *,
    user_id: str,
    expert_id: str | None,
    skill_name: str,
    version_id: str,
    action: Literal["apply", "keep_current", "apply_edited"],
    edited_body: str | None,
    actor_user_id: str,
) -> PublishOutcome:
    """Resolve one open proposal. Applying re-checks eligibility first."""
    versions = skill_versions_db()
    proposal = await versions.get_version(user_id, version_id)
    if (
        proposal is None
        or proposal.skill_name != skill_name
        or proposal.owner_key != owner_key_for(expert_id)
    ):
        return PublishOutcome(status="conflict", reason="proposal not found")
    if proposal.state != "needs_decision":
        return PublishOutcome(status="conflict", reason="proposal is no longer open")
    head = await versions.get_head(user_id, owner_key_for(expert_id), skill_name)
    if head is None:
        return PublishOutcome(status="conflict", reason="skill has no version history")
    if action == "keep_current":
        await versions.set_version_state(
            user_id, proposal.id, "archived", "owner kept the current version"
        )
        await _suppress_replaced_behavior(
            user_id, expert_id, proposal, head_body="", reason="owner kept current"
        )
        return PublishOutcome(status="applied", version=proposal, reason="kept current")
    if head.current_version_id != proposal.base_version_id:
        await versions.set_version_state(
            user_id, proposal.id, "stale", "the skill changed before the decision"
        )
        return PublishOutcome(
            status="conflict", reason="the skill changed since this was proposed"
        )
    stale = await require_fresh_eligibility(
        PublishRequest(
            user_id=user_id,
            expert_id=expert_id,
            skill_name=skill_name,
            description=proposal.description,
            body=_body_of(proposal),
            summary=proposal.summary,
            origin=proposal.origin,
            sources=[
                SourceSnapshot.model_validate(_snapshot_of(s)) for s in proposal.sources
            ],
        )
    )
    if stale is not None:
        await versions.set_version_state(user_id, proposal.id, "stale", stale.reason)
        return stale
    body = (
        edited_body if action == "apply_edited" and edited_body else _body_of(proposal)
    )
    draft = VersionDraft(
        content=render_skill_markdown(
            canonicalize_skill(
                ParsedSkill(
                    name=skill_name,
                    description=proposal.description,
                    body=body,
                    triggers=tuple(proposal.triggers),
                )
            )
        ),
        description=proposal.description,
        triggers=proposal.triggers,
        origin="edited" if action == "apply_edited" else proposal.origin,
        summary=proposal.summary,
        actor_user_id=actor_user_id,
        base_version_id=head.current_version_id,
        sources=proposal.sources,
        evidence=proposal.evidence,
        limits=proposal.limits,
    )
    outcome = await commit_and_write(user_id, head, draft, None)
    if outcome.status == "applied":
        await versions.set_version_state(
            user_id,
            proposal.id,
            "archived",
            f"applied as v{outcome.version.version if outcome.version else '?'}",
        )
    return outcome


def _body_of(version: SkillVersionRecord) -> str:
    parsed = parse_skill_markdown(version.content, fallback_name=version.skill_name)
    return parsed.body if parsed else version.content


def _snapshot_of(dependency: dict[str, object]) -> dict[str, object]:
    epoch = dependency.get("epoch")
    return {
        "source_id": str(dependency.get("source_id", "")),
        "source_kind": str(dependency.get("source_kind", "")),
        "source_ref": str(dependency.get("source_ref", "")),
        "revision": str(dependency.get("revision", "")),
        "epoch": int(epoch) if isinstance(epoch, int) else None,
    }


async def _unavailable_sources(
    user_id: str, expert_id: str | None, version: SkillVersionRecord
) -> str | None:
    """Reason a version's contributing sources no longer allow its use."""
    scope = LearningScope(
        user_id=user_id, expert_id=expert_id, owner_key=owner_key_for(expert_id)
    )
    for dependency in version.sources:
        adapter = get_source_adapter(str(dependency.get("source_kind", "")))
        if adapter is None:
            return "a contributing source kind is no longer supported"
        eligibility = await adapter.revalidate(
            source_id=str(dependency.get("source_id", "")),
            revision=str(dependency.get("revision", "")),
            scope=scope,
            approval_event_id=None,
        )
        if eligibility.state.value in ("excluded", "inaccessible"):
            return f"a contributing source is {eligibility.state.value}"
    return None


async def _suppress_replaced_behavior(
    user_id: str,
    expert_id: str | None,
    replaced: SkillVersionRecord,
    head_body: str,
    *,
    reason: str,
) -> None:
    """Suppress an automated behaviour the owner just removed."""
    if replaced.origin not in ("saved_overnight", "requested") or not replaced.content:
        return
    replaced_body = _body_of(replaced)
    new_body = head_body
    if new_body and behavior_fingerprint(
        replaced.skill_name, replaced_body
    ) == behavior_fingerprint(replaced.skill_name, _body_of_text(new_body)):
        return
    await skill_use_db().add_suppression(
        user_id,
        expert_id=expert_id,
        skill_name=replaced.skill_name,
        behavior_fingerprint=behavior_fingerprint(replaced.skill_name, replaced_body),
        behavior_tokens=behavior_tokens(replaced_body),
        evidence_fingerprints=[
            evidence_fingerprint([str(e.get("ref", "")) for e in replaced.evidence])
        ],
        actor_user_id=user_id,
        reason=reason,
    )


def _body_of_text(text: str) -> str:
    parsed = parse_skill_markdown(text)
    return parsed.body if parsed else text
