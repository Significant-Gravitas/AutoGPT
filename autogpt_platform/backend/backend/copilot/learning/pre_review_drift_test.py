import pytest

from . import nightly_test
from .nightly_test import USER, _run, _source

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["epoch", "revision", "missing-approval"])
async def test_pre_review_drift_defers_but_invalid_approval_stays_terminal(
    fake_store, adapter, boundaries, monkeypatch, change
):
    source = await _source(fake_store)
    original = fake_store.mark_source_attempted

    async def mutate(user_id, source_id, revision):
        if change == "epoch":
            await fake_store.set_source_eligibility(user_id, source_id, "eligible")
        elif change == "revision":
            await fake_store.upsert_source_revision(
                user_id,
                expert_id=source.expert_id,
                source_kind=source.source_kind,
                source_id=source.source_id,
                revision="9",
                evidence_refs=[],
                outcome_signals=[],
            )
        else:
            adapter.requires_approval = True
        return await original(user_id, source_id, revision)

    monkeypatch.setattr(fake_store, "mark_source_attempted", mutate)
    result = await _run()
    boundaries["review"].assert_not_awaited()
    pending = await fake_store.get_source(USER, source.id)
    if change == "missing-approval":
        assert result.dispositions == {"stale_eligibility": 1}
        assert pending.processed_revision == source.revision
        return
    assert result.dispositions == {"deferred": 1}
    assert pending.processed_revision is None
    monkeypatch.setattr(fake_store, "mark_source_attempted", original)
    assert (await _run()).applied == 1
    reviewed = boundaries["review"].await_args.args[1].source
    assert reviewed.epoch == pending.epoch and reviewed.revision == pending.revision
