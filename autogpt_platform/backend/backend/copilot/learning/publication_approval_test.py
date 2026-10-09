import pytest

from backend.copilot.learning import owner_actions, publish
from backend.copilot.learning.contract import ApprovalCheckpoint
from backend.copilot.learning.contract_test import FakeReviewedAdapter, _ReviewedSource
from backend.copilot.learning.publication_models import PublishRequest, SourceSnapshot


def _saved_request():
    snapshot = SourceSnapshot(
        source_id="item-1",
        source_kind="fake_reviewed_item",
        source_ref="item-1",
        revision="r1",
        epoch=0,
        approval=ApprovalCheckpoint(
            event_id="ev-1", actor_id="reviewer-9", approved_revision="r1"
        ),
    )
    saved = SourceSnapshot.model_validate(
        owner_actions._snapshot_of(snapshot.as_dependency())
    )
    return PublishRequest(
        user_id="user-1",
        expert_id="expert-1",
        skill_name="checks",
        description="Run checks",
        body="Check saved inputs",
        summary="Verified",
        origin="saved_overnight",
        sources=[saved],
    )


@pytest.mark.asyncio
async def test_stored_approval_is_revalidated_before_owner_application(monkeypatch):
    item = _ReviewedSource(
        revision="r1", approved_revision="r1", approval_event_id="ev-1"
    )
    adapter = FakeReviewedAdapter(items={"item-1": item})
    monkeypatch.setattr(publish, "get_source_adapter", lambda _: adapter)
    request = _saved_request()
    assert await publish.require_fresh_eligibility(request) is None
    item.approval_event_id = "replacement-event"
    changed = await publish.require_fresh_eligibility(request)
    assert changed and changed.status == "stale_eligibility"
    item.approval_event_id = "ev-1"
    request.sources[0].approval = None
    missing = await publish.require_fresh_eligibility(request)
    assert missing and missing.status == "stale_eligibility"
