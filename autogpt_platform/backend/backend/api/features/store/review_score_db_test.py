"""The database refuses a store review score outside 1-5 (#15304)."""

import uuid

import pytest

from backend.data.db import execute_raw_with_schema, transaction


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("score", [0, 6, 1_000_000])
async def test_db_rejects_out_of_range_review_score(server, score: int):
    # Postgres checks a CHECK on the row before the foreign keys' after-trigger,
    # so dangling ids are fine here: the row must fail on score.
    with pytest.raises(Exception, match="StoreListingReview_score_range"):
        await _insert_review(score)


@pytest.mark.asyncio(loop_scope="session")
async def test_db_in_range_score_passes_the_score_check(server):
    # In range, the same insert gets past the CHECK and only trips the FK.
    with pytest.raises(Exception) as exc:
        await _insert_review(3)
    assert "StoreListingReview_score_range" not in str(exc.value)


async def _insert_review(score: int) -> None:
    # A failed statement outside a transaction leaves the shared connection
    # aborted and breaks every later test in the session.
    async with transaction() as tx:
        await execute_raw_with_schema(
            _INSERT,
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            score,
            client=tx,
        )


_INSERT = (
    'INSERT INTO {schema_prefix}"StoreListingReview" '
    '("id", "storeListingVersionId", "reviewByUserId", "score") '
    "VALUES ($1, $2, $3, $4)"
)
