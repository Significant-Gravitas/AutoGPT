"""Request/response models for the user-facing skill-learning API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

STATE_LABELS: dict[str, str] = {
    "ready": "Ready to use",
    "pending_write": "Saving",
    "candidate": "Candidate",
    "needs_decision": "Needs your decision",
    "blocked_content": "Blocked by content check",
    "paused": "Paused",
    "archived": "Archived",
    "invalidated": "Archived (source unavailable)",
    "stale": "Archived (stale)",
}

DISPOSITION_LABELS: dict[str, str] = {
    "applied": "Applied",
    "applied_pending": "Saving",
    "needs_decision": "Needs your decision",
    "skipped": "Skipped",
    "no_novel_procedure": "Nothing new to learn",
    "deferred": "Deferred",
    "conflict": "Conflict",
    "blocked_content": "Blocked by content check",
    "provider_error": "Review failed",
    "budget_exhausted": "Budget exhausted",
    "stale_eligibility": "Source no longer eligible",
    "inaccessible_evidence": "Evidence not accessible",
    "suppressed": "Not reapplied (removed by you)",
    "paused": "Paused",
    "write_failed": "Save failed",
}


class LearningEvidenceLabel(BaseModel):
    kind: str
    label: str
    ref: str = ""


class LearningSourceLink(BaseModel):
    """A contributing source, with content only when the viewer may see it."""

    source_id: str
    source_kind: str
    revision: str
    accessible: bool
    source_ref: str | None = None
    title: str | None = None
    url: str | None = None
    excluded: bool = False


class SkillUseSummary(BaseModel):
    """Reuse evidence for exactly one version. Loads are loads; an owner's
    report is a report; only a recorded check can say "passed checks"."""

    version_id: str | None = None
    loads: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    reported_working: int = 0
    reported_failed: int = 0
    stopped_mid_use: int = 0
    unknown: int = 0
    reuse_label: str = "Not yet reused"


class SkillVersionSummary(BaseModel):
    id: str
    skill_name: str
    expert_id: str | None
    version: int
    origin: str
    origin_label: str
    summary: str
    state: str
    state_label: str
    state_reason: str
    created_at: datetime
    description: str
    triggers: list[str] = Field(default_factory=list)
    body: str | None = None
    evidence: list[LearningEvidenceLabel] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    sources: list[LearningSourceLink] = Field(default_factory=list)
    use: SkillUseSummary = Field(default_factory=SkillUseSummary)
    base_version_id: str | None = None
    restored_from_version_id: str | None = None
    blocked_pattern_class: str | None = None
    blocked_step: str | None = None


class SkillLearningPolicy(BaseModel):
    auto_improve: bool = True
    learning_paused: bool = False
    use_paused: bool = False


class SkillLearningDetail(BaseModel):
    skill_name: str
    expert_id: str | None
    state: str
    state_label: str
    current_version: SkillVersionSummary | None = None
    open_decision: SkillVersionSummary | None = None
    versions: list[SkillVersionSummary] = Field(default_factory=list)
    policy: SkillLearningPolicy
    use: SkillUseSummary


class LearningHistoryItem(BaseModel):
    id: str
    kind: Literal["version", "review"]
    skill_name: str | None
    expert_id: str | None
    created_at: datetime
    summary: str
    origin: str | None = None
    origin_label: str | None = None
    state: str
    state_label: str
    version: int | None = None
    version_id: str | None = None
    source: LearningSourceLink | None = None


class LearningHistoryResponse(BaseModel):
    expert_id: str | None
    items: list[LearningHistoryItem]


class OpenDecisionsResponse(BaseModel):
    items: list[SkillVersionSummary]


class LearningStatus(BaseModel):
    enabled: bool
    expert_id: str | None
    learning_paused: bool
    pending_sources: int
    oldest_pending_at: datetime | None
    last_review_at: datetime | None
    last_applied_at: datetime | None
    retrying_reviews: int
    cost_microdollars_30d: int
    source_kinds: list[str]


class LearningActionResult(BaseModel):
    status: str
    status_label: str
    reason: str = ""
    version: SkillVersionSummary | None = None
    pattern_class: str | None = None
    blocked_step: str | None = None


class SkillPolicyRequest(BaseModel):
    auto_improve: bool | None = None
    learning_paused: bool | None = None
    use_paused: bool | None = None


class EditSkillRequest(BaseModel):
    description: str = Field(min_length=1, max_length=1024)
    body: str = Field(min_length=1, max_length=20_000)
    triggers: list[str] = Field(default_factory=list, max_length=10)
    keep_auto_improve: bool = False
    allowed_pattern_classes: list[str] = Field(default_factory=list, max_length=10)
    expected_version_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "The current version's id when editing began; null only for a "
            "skill with no tracked version. A different current version at "
            "save time is a conflict and the edit is not applied."
        ),
    )


class RestoreVersionRequest(BaseModel):
    version_id: str = Field(min_length=1, max_length=128)


class DecisionRequest(BaseModel):
    action: Literal["apply", "keep_current", "apply_edited"]
    edited_body: str | None = Field(default=None, max_length=20_000)


class OutcomeReportRequest(BaseModel):
    version_id: str = Field(min_length=1, max_length=128)
    outcome: Literal["succeeded", "failed", "unknown"]
    detail: str = Field(default="", max_length=1000)


class ExpertLearningPolicyRequest(BaseModel):
    learning_paused: bool


class ExpertLearningPolicy(BaseModel):
    expert_id: str
    learning_paused_at: datetime | None


class SourceExclusionResult(BaseModel):
    source_id: str
    excluded: bool
    invalidated_version_ids: list[str] = Field(default_factory=list)
