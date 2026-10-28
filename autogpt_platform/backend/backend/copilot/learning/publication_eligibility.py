from backend.data.skill_learning import owner_key_for

from .contract import (
    Eligibility,
    EligibilityState,
    LearningScope,
    SourceRevision,
    check_approval_precondition,
    eligibility_snapshot_changed,
    get_source_adapter,
)
from .publication_models import (
    PublishOutcome,
    PublishRequest,
    PublishStatus,
    SourceSnapshot,
)


async def require_fresh_eligibility(
    request: PublishRequest, *, retry_changed_source: bool = False
) -> PublishOutcome | None:
    """Recheck pinned sources; automated reviews can retry changed evidence."""
    scope = LearningScope(
        user_id=request.user_id,
        expert_id=request.expert_id,
        owner_key=owner_key_for(request.expert_id),
    )
    for snapshot in request.sources:
        outcome = await _check_snapshot(snapshot, scope, retry_changed_source)
        if outcome is not None:
            return outcome
    return None


async def _check_snapshot(
    snapshot: SourceSnapshot, scope: LearningScope, retry_changed_source: bool
) -> PublishOutcome | None:
    adapter = get_source_adapter(snapshot.source_kind)
    if adapter is None:
        return PublishOutcome(status="stale_eligibility", reason="unknown source kind")
    eligibility = await adapter.revalidate(
        source_id=snapshot.source_id,
        revision=snapshot.revision,
        scope=scope,
        approval_event_id=snapshot.approval.event_id if snapshot.approval else None,
    )
    source = SourceRevision(
        source_id=snapshot.source_id,
        source_kind=snapshot.source_kind,
        source_ref=snapshot.source_ref,
        scope=scope,
        revision=snapshot.revision,
        epoch=snapshot.epoch if snapshot.epoch is not None else eligibility.epoch,
        approval=snapshot.approval,
    )
    problem = check_approval_precondition(adapter, eligibility, source)
    if problem is None:
        return None
    return PublishOutcome(
        status=_failure_status(eligibility, source, retry_changed_source),
        reason=problem,
    )


def _failure_status(
    eligibility: Eligibility, source: SourceRevision, retry_changed_source: bool
) -> PublishStatus:
    if eligibility.state in (
        EligibilityState.PAUSED,
        EligibilityState.WAITING_APPROVAL,
    ):
        return "paused"
    if retry_changed_source and (
        eligibility.state == EligibilityState.STALE
        or eligibility_snapshot_changed(eligibility, source)
    ):
        return "conflict"
    return "stale_eligibility"
