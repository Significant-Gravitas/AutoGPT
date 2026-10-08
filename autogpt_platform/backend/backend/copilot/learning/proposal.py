"""Deterministic validation of a reviewer proposal and its publish request.

The reviewer model cannot talk its way past these checks: a valid slug,
the required sections, a concrete verification statement, citations of
evidence spans that exist and carry a checked outcome, and no duplicate
creation or phantom update.
"""

from __future__ import annotations

import re

from backend.copilot.tools.skills import (
    ParsedSkill,
    SkillPackage,
    validate_package,
    validate_skill_content,
)
from backend.data.skill_publication import ReviewStamp

from .contract import EvidenceBundle, SourceRevision
from .packages import ReviewedPackage
from .prompts import MAX_PROPOSAL_BODY_CHARS, LearningProposal
from .publish import PublishRequest, SourceSnapshot

_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_REQUIRED_SECTIONS = ("steps", "verification")


def validate_proposal(
    proposal: LearningProposal,
    bundle: EvidenceBundle,
    existing: list[ParsedSkill],
    packages: dict[str, ReviewedPackage] | None = None,
) -> str | None:
    """Deterministic checks the reviewer cannot talk its way past.

    Returns the reason the proposal is rejected, or ``None`` when it may be
    published. A ``skip`` is a valid, honest outcome and is reported with
    the model's reason.
    """
    if proposal.decision == "skip":
        return proposal.reason or "no reusable procedure"
    if not proposal.skill_name or not _SLUG_RE.match(proposal.skill_name):
        return "proposal has no valid skill name"
    if not proposal.description or not proposal.body:
        return "proposal is missing a description or body"
    try:
        validate_skill_content(proposal.description, proposal.body, proposal.triggers)
        validate_package(
            SkillPackage(
                skill_md=proposal.body,
                files=[f.as_skill_file() for f in proposal.files or []],
            )
        )
    except ValueError as exc:
        return str(exc)
    lowered = proposal.body.lower()
    missing = [s for s in _REQUIRED_SECTIONS if f"## {s}" not in lowered]
    if missing:
        return f"proposal lacks required sections: {', '.join(missing)}"
    if not proposal.verification.strip():
        return "proposal names no concrete verification"
    known = {span.ref for span in bundle.spans}
    cited = [ref for ref in proposal.supported_by if ref in known]
    if not cited:
        return "proposal cites no evidence spans that exist"
    verifying = {
        s.ref
        for s in bundle.source.outcome_signals
        if s.kind in ("tool_result", "user_confirmation", "accepted_artifact")
    }
    if not any(ref in verifying for ref in cited):
        return "proposal cites no span with a checked outcome"
    names = {s.name for s in existing}
    if proposal.decision == "update" and proposal.skill_name not in names:
        return "update names a skill that does not exist"
    if proposal.decision == "update":
        current = next(s for s in existing if s.name == proposal.skill_name)
        if not current.body or len(current.body) > MAX_PROPOSAL_BODY_CHARS:
            return "update requires the complete existing skill body"
    if proposal.decision == "create" and proposal.skill_name in names:
        return "create would duplicate an existing skill"
    return _validate_files(proposal, known, verifying, packages or {})


def _validate_files(
    proposal: LearningProposal,
    known: set[str],
    verifying: set[str],
    packages: dict[str, ReviewedPackage],
) -> str | None:
    if proposal.files is None:
        return None
    current = packages.get(proposal.skill_name or "")
    if proposal.decision == "update" and (current is None or not current.complete):
        return "file update requires the complete existing skill package"
    previous = {f.relative_path: f for f in current.files} if current else {}
    text = (proposal.body or "") + "\n" + "\n".join(f.content for f in proposal.files)
    for file in proposal.files:
        old = previous.get(file.relative_path)
        if (
            old
            and old.content == file.content
            and old.is_executable == file.is_executable
        ):
            continue
        if not (set(file.supported_by) & known & verifying):
            return "each changed file must cite evidence with a checked outcome"
        if file.relative_path not in text:
            return "each changed file must be referenced by the skill package"
    return None


def build_publish_request(
    user_id: str,
    source: SourceRevision,
    proposal: LearningProposal,
    bundle: EvidenceBundle,
    stamp: ReviewStamp,
    packages: dict[str, ReviewedPackage] | None = None,
) -> PublishRequest:
    assert proposal.skill_name and proposal.description and proposal.body
    signal_labels = {s.ref: s.label for s in source.outcome_signals}
    evidence = [
        {"kind": "span", "ref": ref, "label": signal_labels.get(ref, "cited evidence")}
        for ref in proposal.supported_by
        if ref in {span.ref for span in bundle.spans}
    ]
    evidence.append(
        {"kind": "outcome", "ref": "", "label": "Worked once in the source"}
    )
    if source.is_requested:
        evidence.append({"kind": "origin", "ref": "", "label": "Requested by user"})
    limits = list(proposal.limits)
    if bundle.omitted_refs:
        limits.append(f"{len(bundle.omitted_refs)} evidence item(s) were not reviewed")
    return PublishRequest(
        user_id=user_id,
        expert_id=source.scope.expert_id,
        skill_name=proposal.skill_name,
        description=proposal.description,
        triggers=proposal.triggers,
        body=proposal.body,
        files=(
            [f.as_skill_file() for f in proposal.files]
            if proposal.files is not None
            else None
        ),
        expected_package_hash=(
            packages[proposal.skill_name].package_hash
            if packages
            and proposal.decision == "update"
            and proposal.skill_name in packages
            else None
        ),
        summary=proposal.summary
        or f"{proposal.decision.capitalize()}d from a verified procedure",
        origin="requested" if source.is_requested else "saved_overnight",
        sources=[
            SourceSnapshot(
                source_id=source.source_id,
                source_kind=source.source_kind,
                source_ref=source.source_ref,
                revision=source.revision,
                epoch=source.epoch,
                approval=source.approval,
            )
        ],
        evidence=evidence,
        limits=limits,
        supported_refs=proposal.supported_by,
        review=stamp,
    )
