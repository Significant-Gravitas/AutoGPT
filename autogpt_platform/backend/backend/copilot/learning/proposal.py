"""Deterministic validation of a reviewer proposal and its publish request.

The reviewer model cannot talk its way past these checks: a valid slug,
the required sections, a concrete verification statement, citations of
evidence spans that exist and carry a checked outcome, and no duplicate
creation or phantom update.
"""

from __future__ import annotations

import re

from backend.copilot.tools.skills import ParsedSkill
from backend.data.skill_publication import ReviewStamp

from .contract import EvidenceBundle, SourceRevision
from .prompts import LearningProposal
from .publish import PublishRequest, SourceSnapshot

_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_REQUIRED_SECTIONS = ("steps", "verification")


def validate_proposal(
    proposal: LearningProposal,
    bundle: EvidenceBundle,
    existing: list[ParsedSkill],
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
    if proposal.decision == "create" and proposal.skill_name in names:
        return "create would duplicate an existing skill"
    return None


def build_publish_request(
    user_id: str,
    source: SourceRevision,
    proposal: LearningProposal,
    bundle: EvidenceBundle,
    stamp: ReviewStamp,
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
        triggers=proposal.triggers[:10],
        body=proposal.body,
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
