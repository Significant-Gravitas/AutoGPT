from datetime import datetime, timezone

import pytest

from . import (
    chat_source_test,
    nightly_test,
    owner_actions,
    publication_eligibility,
    publish,
)
from .chat_source import ChatSessionSourceAdapter, record_chat_turn
from .chat_source_test import _messages, _session
from .contract import register_source_adapter
from .contract_test import FakeReviewedAdapter, _ReviewedSource
from .nightly_test import _completion, _proposal, _run
from .publication_approval_test import _saved_request
from .publish_test import BODY_V2, EXPERT, NAME, USER, _content, _seed

chat_db = chat_source_test.chat_db
boundaries = nightly_test.boundaries


@pytest.mark.asyncio
async def test_pausing_during_review_preserves_the_revision_until_resume(
    fake_store, chat_db, boundaries
):
    register_source_adapter(ChatSessionSourceAdapter())
    source = await record_chat_turn(USER, _session(), _messages(3), "import the csv")
    _, _, expert = chat_db

    async def pause_after_review(*_args):
        expert.learning_paused_at = datetime.now(timezone.utc)
        return _completion(_proposal(supported_by=["msg:2"]))

    boundaries["review"].side_effect = pause_after_review
    paused = await _run()
    assert paused.dispositions == {"paused": 1}
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None
    boundaries["write"].assert_not_awaited()
    still_paused = await _run()
    assert still_paused.dispositions == {"deferred": 1}
    assert boundaries["review"].await_count == 1

    expert.learning_paused_at = None
    boundaries["review"].side_effect = None
    boundaries["review"].return_value = _completion(_proposal(supported_by=["msg:2"]))
    resumed = await _run()
    assert resumed.applied == 1
    assert boundaries["review"].await_count == 1
    boundaries["write"].assert_awaited_once()
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision


@pytest.mark.asyncio
async def test_owner_proposal_remains_open_while_expert_learning_is_paused(
    fake_store, chat_db, boundaries
):
    register_source_adapter(ChatSessionSourceAdapter())
    source = await record_chat_turn(USER, _session(), _messages(3), "import the csv")
    dependency = {
        "source_id": source.id,
        "source_kind": source.source_kind,
        "source_ref": source.source_id,
        "revision": source.revision,
        "epoch": source.epoch,
    }
    await _seed(fake_store, sources=[dependency])
    head = await fake_store.get_head(USER, EXPERT, NAME)
    proposal = await fake_store.create_version(
        USER,
        head=head,
        content=_content(BODY_V2 + "3. Log counts\n"),
        description="Import CSV",
        triggers=["csv"],
        origin="saved_overnight",
        state="needs_decision",
        base_version_id=head.current_version_id,
        sources=[dependency],
    )
    _, _, expert = chat_db
    expert.learning_paused_at = datetime.now(timezone.utc)
    paused = await _decide(proposal.id)
    assert paused.status == "paused"
    assert (await fake_store.get_version(USER, proposal.id)).state == "needs_decision"
    assert (
        await fake_store.get_head(USER, EXPERT, NAME)
    ).current_version_id == head.current_version_id
    boundaries["write"].assert_not_awaited()

    expert.learning_paused_at = None
    resumed = await _decide(proposal.id)
    assert resumed.status == "applied"
    assert (await fake_store.get_version(USER, proposal.id)).state == "archived"
    boundaries["write"].assert_awaited_once()


@pytest.mark.asyncio
async def test_waiting_for_source_approval_is_retryable(monkeypatch):
    item = _ReviewedSource(revision="r1", approval_event_id="ev-1")
    adapter = FakeReviewedAdapter(items={"item-1": item})
    monkeypatch.setattr(
        publication_eligibility, "get_source_adapter", lambda _: adapter
    )
    request = _saved_request()
    waiting = await publish.require_fresh_eligibility(request)
    assert waiting is not None and waiting.status == "paused"
    item.approved_revision = "r1"
    assert await publish.require_fresh_eligibility(request) is None


async def _decide(version_id: str):
    return await owner_actions.decide_proposal(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        version_id=version_id,
        action="apply",
        edited_body=None,
        actor_user_id=USER,
    )
