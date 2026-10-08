"""Requests and outcomes shared by private skill publication paths."""

from typing import Literal

from pydantic import BaseModel, Field

from backend.copilot.tools.skills import SkillFile
from backend.data.skill_publication import ReviewStamp
from backend.data.skill_versions import SkillVersionRecord

from .contract import ApprovalCheckpoint

PublishStatus = Literal[
    "applied",
    "needs_decision",
    "conflict",
    "stale_eligibility",
    "blocked_content",
    "suppressed",
    "paused",
    "write_failed",
    "invalid_proposal",
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
    files: list[SkillFile] | None = None
    expected_package_hash: str | None = None
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
