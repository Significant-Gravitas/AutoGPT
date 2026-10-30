import asyncio
from uuid import uuid4

import pytest

from backend.data import skill_reviews as reviews
from backend.data.skill_learning_test import _cleanup, _create_user, _source


@pytest.mark.asyncio(loop_scope="session")
async def test_model_attempt_reservation_is_atomic_and_owned(server):
    user, foreign = str(uuid4()), str(uuid4())
    await _create_user(user)
    await _create_user(foreign)
    try:
        source = await _source(user)
        params = dict(
            source_id=source.id,
            source_revision=source.revision,
            policy_version=1,
            max_attempts=3,
            live_seconds=0,
            expected_epoch=source.epoch,
        )
        assert (
            await reviews.reserve_model_review(foreign, run_id="foreign", **params)
            is None
        )
        results = await asyncio.gather(
            *(
                reviews.reserve_model_review(user, run_id=f"attempt-{i}", **params)
                for i in range(5)
            )
        )
        assert sum(row is not None for row in results) == 3
        row = await reviews.get_review_for_revision(user, source.id, source.revision, 1)
        assert row.model_attempts == 3 and row.attempts == 0
        assert not await reviews.checkpoint_review_usage(
            foreign,
            row.id,
            row.run_id,
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            cost_microdollars=1000,
        )
        assert not await reviews.set_review_replay(
            user, row.id, "stale-run", {"version": 1}
        )
        assert await reviews.checkpoint_review_usage(
            user,
            row.id,
            row.run_id,
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            cost_microdollars=1000,
        )
        assert await reviews.set_review_replay(user, row.id, row.run_id, {"version": 1})
        assert (
            await reviews.get_review_for_revision(
                foreign, source.id, source.revision, 1
            )
            is None
        )
        settled = await reviews.upsert_review(
            user,
            source_id=source.id,
            source_revision=source.revision,
            policy_version=1,
            run_id="settled",
            disposition="skipped",
            reason="attempt limit reached",
            live_seconds=0,
        )
        assert settled.cost_microdollars == 1000 and settled.input_tokens == 10
        assert settled.model_attempts == 3 and settled.replay is None
        assert not await reviews.set_review_replay(
            user, row.id, "settled", {"version": 1}
        )
    finally:
        await _cleanup(user, foreign)


@pytest.mark.asyncio(loop_scope="session")
async def test_checkpoint_replay_is_bounded_and_does_not_create_another_paid_attempt(
    server,
):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        params = dict(
            source_id=source.id,
            source_revision=source.revision,
            policy_version=1,
            max_attempts=3,
            live_seconds=0,
            expected_epoch=source.epoch,
        )
        first = await reviews.reserve_model_review(user, run_id="paid", **params)
        assert first is not None
        with pytest.raises(ValueError, match="storage budget"):
            await reviews.set_review_replay(
                user, first.id, first.run_id, {"oversized": "x" * 524_288}
            )
        assert await reviews.set_review_replay(
            user, first.id, first.run_id, {"version": 1}
        )
        assert (
            await reviews.reserve_model_review(user, run_id="duplicate", **params)
            is None
        )
        assert await reviews.set_review_replay(user, first.id, first.run_id, None)
        assert (
            await reviews.reserve_model_review(user, run_id="new", **params)
        ).model_attempts == 2
    finally:
        await _cleanup(user)
