"""Contract tests for the generic approval-aware source adapter.

A fake adapter with an approval lifecycle (no application schema) proves
the learner's rules: a static approved flag is not enough, publication
requires the fresh eligibility answer, revocation invalidates, and races
between exclusion and review are resolved in favour of the exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from .contract import (
    ApprovalCheckpoint,
    Eligibility,
    EligibilityState,
    EvidenceBundle,
    LearningScope,
    OutcomeSignal,
    SourceRevision,
    check_approval_precondition,
    get_source_adapter,
    list_source_kinds,
    register_source_adapter,
)

SCOPE = LearningScope(user_id="user-1", expert_id="expert-1", owner_key="expert-1")


@dataclass
class _ReviewedSource:
    revision: str
    epoch: int = 0
    approved_revision: str | None = None
    approval_event_id: str | None = None
    excluded: bool = False
    owner_key: str = "expert-1"


@dataclass
class FakeReviewedAdapter:
    """An approval-aware source kind that lives entirely in this test."""

    kind: str = "fake_reviewed_item"
    requires_approval: bool = True
    items: dict[str, _ReviewedSource] = field(default_factory=dict)

    async def revalidate(
        self,
        *,
        source_id: str,
        revision: str,
        scope: LearningScope,
        approval_event_id: str | None,
    ) -> Eligibility:
        item = self.items.get(source_id)
        if item is None or item.owner_key != scope.owner_key:
            return Eligibility(
                state=EligibilityState.INACCESSIBLE, revision=revision, epoch=-1
            )
        if item.excluded:
            return Eligibility(
                state=EligibilityState.EXCLUDED,
                revision=item.revision,
                epoch=item.epoch,
            )
        if item.approved_revision is None:
            return Eligibility(
                state=EligibilityState.WAITING_APPROVAL,
                revision=item.revision,
                epoch=item.epoch,
                reason="awaiting review",
            )
        return Eligibility(
            state=EligibilityState.ELIGIBLE,
            revision=item.revision,
            epoch=item.epoch,
            approval=ApprovalCheckpoint(
                event_id=item.approval_event_id or "",
                actor_id="reviewer-9",
                approved_revision=item.approved_revision,
            ),
        )

    async def load_evidence(
        self, source: SourceRevision, *, max_chars: int
    ) -> EvidenceBundle:
        return EvidenceBundle(source=source)


def _source(
    revision: str = "r1", epoch: int = 0, event: str | None = "ev-1"
) -> SourceRevision:
    return SourceRevision(
        source_id="item-1",
        source_kind="fake_reviewed_item",
        source_ref="item-1",
        scope=SCOPE,
        revision=revision,
        epoch=epoch,
        approval=(
            ApprovalCheckpoint(
                event_id=event, actor_id="reviewer-9", approved_revision=revision
            )
            if event
            else None
        ),
        outcome_signals=[OutcomeSignal(kind="accepted_artifact", ref="r", label="ok")],
    )


@pytest.fixture
def adapter() -> FakeReviewedAdapter:
    adapter = FakeReviewedAdapter()
    adapter.items["item-1"] = _ReviewedSource(
        revision="r1", approved_revision="r1", approval_event_id="ev-1"
    )
    register_source_adapter(adapter)
    return adapter


@pytest.mark.asyncio
async def test_registered_adapter_is_discoverable_by_kind(adapter: FakeReviewedAdapter):
    assert get_source_adapter("fake_reviewed_item") is adapter
    assert "fake_reviewed_item" in list_source_kinds()


@pytest.mark.asyncio
async def test_approved_revision_is_eligible(adapter: FakeReviewedAdapter):
    source = _source()
    eligibility = await adapter.revalidate(
        source_id="item-1", revision="r1", scope=SCOPE, approval_event_id="ev-1"
    )
    assert check_approval_precondition(adapter, eligibility, source) is None


@pytest.mark.asyncio
async def test_unapproved_revision_waits(adapter: FakeReviewedAdapter):
    adapter.items["item-1"].approved_revision = None
    eligibility = await adapter.revalidate(
        source_id="item-1", revision="r1", scope=SCOPE, approval_event_id=None
    )
    assert eligibility.state == EligibilityState.WAITING_APPROVAL
    assert check_approval_precondition(adapter, eligibility, _source(event=None))


@pytest.mark.asyncio
async def test_static_approved_flag_in_queued_payload_is_insufficient(
    adapter: FakeReviewedAdapter,
):
    """The queued snapshot says approved; the adapter now says a newer,
    unapproved revision exists. The stale snapshot must not publish."""
    adapter.items["item-1"].revision = "r2"
    adapter.items["item-1"].approved_revision = "r1"
    eligibility = await adapter.revalidate(
        source_id="item-1", revision="r1", scope=SCOPE, approval_event_id="ev-1"
    )
    problem = check_approval_precondition(adapter, eligibility, _source())
    assert problem == "a newer revision exists; re-gather before reviewing"


@pytest.mark.asyncio
async def test_only_the_approved_snapshot_is_eligible_after_revision(
    adapter: FakeReviewedAdapter,
):
    """Accepted, then revised: only r1 may be learned from, never r2."""
    adapter.items["item-1"].revision = "r2"
    eligibility = await adapter.revalidate(
        source_id="item-1", revision="r2", scope=SCOPE, approval_event_id="ev-1"
    )
    problem = check_approval_precondition(adapter, eligibility, _source(revision="r2"))
    assert problem == "approved revision does not match the revision under review"


@pytest.mark.asyncio
async def test_epoch_change_makes_a_queued_snapshot_stale(adapter: FakeReviewedAdapter):
    """Exclusion during a running review bumps the epoch; apply refuses."""
    adapter.items["item-1"].epoch = 1
    eligibility = await adapter.revalidate(
        source_id="item-1", revision="r1", scope=SCOPE, approval_event_id="ev-1"
    )
    problem = check_approval_precondition(adapter, eligibility, _source(epoch=0))
    assert problem == "eligibility changed since the source was queued"


@pytest.mark.asyncio
async def test_withdrawn_approval_event_is_refused(adapter: FakeReviewedAdapter):
    adapter.items["item-1"].approval_event_id = "ev-2"
    eligibility = await adapter.revalidate(
        source_id="item-1", revision="r1", scope=SCOPE, approval_event_id="ev-1"
    )
    problem = check_approval_precondition(adapter, eligibility, _source(event="ev-1"))
    assert problem == "approval event changed since the source was queued"


@pytest.mark.asyncio
async def test_excluded_and_foreign_scope_sources_are_refused(
    adapter: FakeReviewedAdapter,
):
    adapter.items["item-1"].excluded = True
    excluded = await adapter.revalidate(
        source_id="item-1", revision="r1", scope=SCOPE, approval_event_id="ev-1"
    )
    assert excluded.state == EligibilityState.EXCLUDED
    assert not excluded.usable
    adapter.items["item-1"].excluded = False
    other_scope = LearningScope(
        user_id="user-1", expert_id="expert-2", owner_key="expert-2"
    )
    foreign = await adapter.revalidate(
        source_id="item-1", revision="r1", scope=other_scope, approval_event_id="ev-1"
    )
    assert foreign.state == EligibilityState.INACCESSIBLE


def test_chat_wording_cannot_create_an_approval_checkpoint():
    """Acceptance signals are evidence labels only; approval lives on the
    adapter's checkpoint and the learner never infers it from text."""
    source = _source(event=None)
    assert source.has_verifying_signal
    assert source.approval is None
