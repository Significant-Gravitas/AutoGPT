from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from backend.copilot.learning import nightly, nightly_test
from backend.copilot.learning.nightly_test import USER, _run, _source

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner, unavailable",
    [
        ("another-worker", False),
        (None, False),
        (None, True),
    ],
)
async def test_only_confirmed_lease_contention_is_reported_as_held(
    monkeypatch, owner, unavailable
):
    redis = AsyncMock()
    redis.set.return_value = False
    redis.get.return_value = owner
    if unavailable:
        redis.set.side_effect = ConnectionError("Redis unavailable")
    monkeypatch.setattr(nightly, "get_redis_async", AsyncMock(return_value=redis))
    monkeypatch.setattr(nightly, "is_feature_enabled", AsyncMock(return_value=True))
    run = AsyncMock()
    monkeypatch.setattr(nightly, "_run_under_lease", run)
    result = await nightly.run_skill_learning_pass("lease-test-user")
    run.assert_not_awaited()
    if owner:
        assert result.skip_reason == "lease_held" and not result.error
    else:
        assert not result.skipped and result.skip_reason != "lease_held"
        assert result.error == "RuntimeError: skill learning lease unavailable"


async def _reviewed_by_another_run(store, source):
    """The revision at its model-attempt limit, the last attempt still
    reviewing under another run's id."""
    for _ in range(nightly.MAX_MODEL_ATTEMPTS):
        claim = await store.reserve_model_review(
            USER,
            source_id=source.id,
            source_revision=source.revision,
            policy_version=nightly.POLICY_VERSION,
            run_id="another-run",
            max_attempts=nightly.MAX_MODEL_ATTEMPTS,
            expected_epoch=source.epoch,
        )
        assert claim is not None
    return await store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )


@pytest.mark.asyncio
async def test_a_review_another_run_may_still_be_making_is_left_alone(
    fake_store, adapter, boundaries
):
    """A run that lost its lease can still be inside its model call. Settling
    the revision here would overwrite its row and waste that paid review."""
    source = await _source(fake_store)
    live = await _reviewed_by_another_run(fake_store, source)

    result = await _run()

    assert result.dispositions == {"deferred": 1}
    boundaries["review"].assert_not_called()
    row = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert row == live


@pytest.mark.asyncio
async def test_a_review_left_by_a_run_that_died_is_settled(
    fake_store, adapter, boundaries
):
    """A reviewing row older than any model call can run belongs to a run
    that died; the attempt limit settles the revision as before."""
    source = await _source(fake_store)
    stale = await _reviewed_by_another_run(fake_store, source)
    key = fake_store._review_key(source.id, source.revision, nightly.POLICY_VERSION)
    fake_store.reviews[key] = stale.model_copy(
        update={
            "updated_at": stale.updated_at
            - timedelta(seconds=nightly.LIVE_REVIEW_SECONDS + 1)
        }
    )

    result = await _run()

    assert result.dispositions == {"skipped": 1}
    boundaries["review"].assert_not_called()
    row = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert row.disposition == "skipped" and "attempt" in row.reason
