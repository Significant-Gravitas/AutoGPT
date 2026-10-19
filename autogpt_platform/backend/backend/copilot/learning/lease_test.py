from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from backend.copilot.learning import nightly, nightly_test
from backend.copilot.learning.nightly_test import USER, _run, _source
from backend.copilot.learning.reviewer import REVIEW_TIMEOUT_SECONDS
from backend.data.skill_reviews import LIVE_REVIEW_SECONDS

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


async def _reviewed_by_another_run(
    store, source, attempts: int = nightly.MAX_MODEL_ATTEMPTS
):
    """The revision after ``attempts`` model attempts, the last one still
    reviewing under another run's id."""
    for _ in range(attempts):
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


def _left_by_a_run_that_died(store, review):
    """Age the claim past any model call it could still be waiting on."""
    key = store._review_key(
        review.source_id, review.source_revision, review.policy_version
    )
    store.reviews[key] = review.model_copy(
        update={
            "updated_at": review.updated_at - timedelta(seconds=LIVE_REVIEW_SECONDS + 1)
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts", range(1, nightly.MAX_MODEL_ATTEMPTS + 1))
async def test_a_review_another_run_may_still_be_making_is_left_alone(
    fake_store, adapter, boundaries, attempts
):
    """A run that lost its lease can still be inside its model call. Taking
    over or settling the revision here would overwrite its row and waste
    that paid review."""
    source = await _source(fake_store)
    live = await _reviewed_by_another_run(fake_store, source, attempts)

    result = await _run()

    assert result.dispositions == {"deferred": 1}
    boundaries["review"].assert_not_called()
    row = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert row == live


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts", range(1, nightly.MAX_MODEL_ATTEMPTS))
async def test_a_review_left_by_a_run_that_died_is_taken_over(
    fake_store, adapter, boundaries, attempts
):
    """Below the attempt limit, the revision is reviewed again."""
    source = await _source(fake_store)
    stale = await _reviewed_by_another_run(fake_store, source, attempts)
    _left_by_a_run_that_died(fake_store, stale)

    result = await _run()

    assert result.dispositions == {"applied": 1}
    boundaries["review"].assert_awaited_once()
    row = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert (row.run_id, row.model_attempts) == (result.run_id, attempts + 1)


@pytest.mark.asyncio
async def test_a_review_left_by_a_run_that_died_is_settled(
    fake_store, adapter, boundaries
):
    """At the attempt limit, the revision settles as before."""
    source = await _source(fake_store)
    stale = await _reviewed_by_another_run(fake_store, source)
    _left_by_a_run_that_died(fake_store, stale)

    result = await _run()

    assert result.dispositions == {"skipped": 1}
    boundaries["review"].assert_not_called()
    row = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert row.disposition == "skipped" and "attempt" in row.reason


def test_a_claim_stays_live_for_longer_than_a_model_review_can_run():
    assert LIVE_REVIEW_SECONDS >= 2 * REVIEW_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_an_exit_before_the_reservation_leaves_a_live_review_alone(
    fake_store, adapter, boundaries
):
    """The pass can stop on a source before reserving it, here because the
    budget ran out after the pass began; recording why must not overwrite
    another run's review either."""
    source = await _source(fake_store)
    live = await _reviewed_by_another_run(fake_store, source, 1)
    boundaries["budget"].side_effect = [(True, None), (False, "insufficient_credits")]

    result = await _run()

    assert result.dispositions == {"deferred": 1}
    boundaries["review"].assert_not_called()
    row = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert row == live
