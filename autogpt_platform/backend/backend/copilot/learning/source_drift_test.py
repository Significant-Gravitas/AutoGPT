import pytest

from . import chat_source_test, nightly_test
from .chat_source import ChatSessionSourceAdapter, record_chat_turn
from .chat_source_test import USER, _messages, _session
from .contract import register_source_adapter
from .nightly_test import _completion, _proposal, _run

chat_db = chat_source_test.chat_db
boundaries = nightly_test.boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["revision", "epoch"])
async def test_source_changes_during_review_preserve_evidence_for_retry(
    fake_store, chat_db, boundaries, drift
):
    register_source_adapter(ChatSessionSourceAdapter())
    source = await record_chat_turn(USER, _session(), _messages(3), "import the csv")

    async def change_during_review(*_args):
        if drift == "revision":
            follow_up = [m for m in _messages(4) if m.sequence == 4]
            await record_chat_turn(USER, _session(), follow_up, "Now delete everything")
        else:
            await fake_store.set_source_eligibility(USER, source.id, "excluded")
            await fake_store.set_source_eligibility(USER, source.id, "eligible")
        return _completion(_proposal(supported_by=["msg:2"]))

    boundaries["review"].side_effect = change_during_review
    first = await _run(trigger="requested")
    pending = await fake_store.get_source(USER, source.id)
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, 1
    )
    assert first.dispositions == {"conflict": 1}
    assert ledger.disposition == "conflict" and ledger.cost_microdollars == 1000
    assert pending.processed_revision is None
    boundaries["write"].assert_not_awaited()

    boundaries["review"].side_effect = None
    boundaries["review"].return_value = _completion(_proposal(supported_by=["msg:2"]))
    retried = await _run()
    assert retried.applied == 1
    assert boundaries["review"].await_count == 2
    boundaries["write"].assert_awaited_once()
    reviewed_source = boundaries["review"].await_args.args[1].source
    assert reviewed_source.revision == pending.revision
    assert reviewed_source.epoch == pending.epoch
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == pending.revision


@pytest.mark.asyncio
async def test_source_excluded_during_review_still_settles_without_publication(
    fake_store, chat_db, boundaries
):
    register_source_adapter(ChatSessionSourceAdapter())
    source = await record_chat_turn(USER, _session(), _messages(3), "import the csv")

    async def exclude_during_review(*_args):
        await fake_store.set_source_eligibility(
            USER, source.id, "excluded", excluded_by_user_id=USER
        )
        return _completion(_proposal(supported_by=["msg:2"]))

    boundaries["review"].side_effect = exclude_during_review
    excluded = await _run()
    assert excluded.dispositions == {"stale_eligibility": 1}
    assert (
        await fake_store.get_source(USER, source.id)
    ).processed_revision == source.revision
    boundaries["write"].assert_not_awaited()
