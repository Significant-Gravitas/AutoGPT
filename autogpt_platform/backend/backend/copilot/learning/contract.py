"""Generic approval-aware source contract for skill learning.

A *source adapter* owns one kind of learning source (an ordinary chat
session, or an application work item with its own review lifecycle). The
learner never imports an adapter's schema and never infers eligibility from
text: it asks the adapter, which answers with a live :class:`Eligibility`
carrying the current state, revision, and epoch. Publication is conditional
on that fresh answer plus the expected skill version — a static approved
flag inside a queued payload is never sufficient.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class EligibilityState(str, Enum):
    ELIGIBLE = "eligible"
    WAITING_APPROVAL = "waiting_approval"
    EXCLUDED = "excluded"
    INACCESSIBLE = "inaccessible"
    PAUSED = "paused"
    STALE = "stale"


class LearningScope(BaseModel):
    """Who owns the learning and where the result lands."""

    user_id: str
    expert_id: str | None = None
    owner_key: str


class EvidenceRef(BaseModel):
    """A bounded pointer into the source — never the full archive."""

    kind: str
    ref: str
    label: str = ""


SignalKind = Literal[
    "tool_result",
    "tool_error",
    "user_confirmation",
    "accepted_artifact",
    "learn_request",
]

# Signals that can make a procedure eligible. A plan, a question, or an
# unsupported "done" claim never qualifies on its own.
VERIFYING_SIGNAL_KINDS: frozenset[str] = frozenset(
    {"tool_result", "user_confirmation", "accepted_artifact"}
)


class OutcomeSignal(BaseModel):
    kind: SignalKind
    ref: str
    label: str = ""


class ApprovalCheckpoint(BaseModel):
    """Who accepted which immutable revision, for approval-aware sources."""

    event_id: str
    actor_id: str | None = None
    approved_revision: str


class Eligibility(BaseModel):
    state: EligibilityState
    revision: str
    epoch: int
    reason: str = ""
    approval: ApprovalCheckpoint | None = None

    @property
    def usable(self) -> bool:
        return self.state == EligibilityState.ELIGIBLE


class SourceRevision(BaseModel):
    """One immutable revision of a source, as the learner sees it."""

    source_id: str
    source_kind: str
    source_ref: str
    scope: LearningScope
    revision: str
    epoch: int
    origin: Literal["ordinary", "requested"] = "ordinary"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    outcome_signals: list[OutcomeSignal] = Field(default_factory=list)
    approval: ApprovalCheckpoint | None = None

    @property
    def has_verifying_signal(self) -> bool:
        return any(s.kind in VERIFYING_SIGNAL_KINDS for s in self.outcome_signals)

    @property
    def is_requested(self) -> bool:
        return self.origin == "requested" or any(
            s.kind == "learn_request" for s in self.outcome_signals
        )


class EvidenceSpan(BaseModel):
    ref: str
    role: str
    text: str
    outcome: str | None = None


class EvidenceBundle(BaseModel):
    """The minimum useful evidence plus identifiers for what was omitted.

    ``spans`` never extend past the source revision under review, so a turn
    that lands between revalidation and the read cannot leak into an older
    revision. ``omitted_refs`` lists every captured reference the bundle
    could not include in full (page cap, context budget, or clipped text);
    the learner must not claim to have reviewed them.
    """

    source: SourceRevision
    spans: list[EvidenceSpan] = Field(default_factory=list)
    omitted_refs: list[str] = Field(default_factory=list)
    clipped_refs: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.spans

    @property
    def verification_complete(self) -> bool:
        """Every verifying signal's span is present and unclipped."""
        present = {span.ref for span in self.spans} - set(self.clipped_refs)
        return all(
            signal.ref in present
            for signal in self.source.outcome_signals
            if signal.kind in VERIFYING_SIGNAL_KINDS
        )


class SourceAdapter(Protocol):
    """Lifecycle rules stay here; the general learner only calls these."""

    kind: str
    requires_approval: bool

    async def revalidate(
        self,
        *,
        source_id: str,
        revision: str,
        scope: LearningScope,
        approval_event_id: str | None,
    ) -> Eligibility: ...

    async def load_evidence(
        self, source: SourceRevision, *, max_chars: int
    ) -> EvidenceBundle: ...


_ADAPTERS: dict[str, SourceAdapter] = {}


def register_source_adapter(adapter: SourceAdapter) -> None:
    _ADAPTERS[adapter.kind] = adapter


def get_source_adapter(kind: str) -> SourceAdapter | None:
    return _ADAPTERS.get(kind)


def list_source_kinds() -> list[str]:
    return sorted(_ADAPTERS)


def check_approval_precondition(
    adapter: SourceAdapter, eligibility: Eligibility, source: SourceRevision
) -> str | None:
    """Return a reason the source may not be used, or ``None`` when it may.

    Approval-aware kinds must present a checkpoint whose approved revision
    is exactly the revision under review; later unapproved work is never
    eligible, and an eligibility snapshot from an older epoch is stale.
    """
    if not eligibility.usable:
        return eligibility.reason or eligibility.state.value
    if eligibility.epoch != source.epoch:
        return "eligibility changed since the source was queued"
    if eligibility.revision != source.revision:
        return "a newer revision exists; re-gather before reviewing"
    if not adapter.requires_approval:
        return None
    checkpoint = eligibility.approval
    if checkpoint is None:
        return "approval checkpoint missing"
    if checkpoint.approved_revision != source.revision:
        return "approved revision does not match the revision under review"
    if source.approval is None or source.approval.event_id != checkpoint.event_id:
        return "approval event changed since the source was queued"
    return None
