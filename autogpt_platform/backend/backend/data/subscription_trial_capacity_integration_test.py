"""The seat cap against real Postgres: the SQL, the lock and the race.

Run only against a disposable database with TRIAL_TEST_DATABASE=1.
"""

import asyncio
import os
from datetime import UTC, datetime
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import SubscriptionTrial, User

from backend.data import db
from backend.data import subscription_trial as trials
from backend.data.subscription_trial import TrialState, reserve_subscription_trial
from backend.data.subscription_trial_capacity import (
    TrialCapacityReached,
    count_trial_seats,
    trial_seat_available,
)
from backend.data.subscription_trial_config import AcceptedTrialOffer

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires explicitly selected disposable trial database",
)


@pytest_asyncio.fixture
async def new_user():
    target = urlparse(db.DATABASE_URL)
    local = (target.hostname, target.port, target.path) == (
        "127.0.0.1",
        15432,
        "/trial_test",
    )
    ci = os.environ.get("GITHUB_ACTIONS") == "true" and (
        target.hostname,
        target.port,
        target.path,
    ) == ("localhost", 5432, "/postgres")
    assert (
        local or ci
    ), "Trial integration tests require an approved disposable database"
    owns_connection = not db.is_connected()
    await db.connect()
    created: list[str] = []

    async def make() -> str:
        user_id = str(uuid4())
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com"}
        )
        created.append(user_id)
        return user_id

    yield make
    await User.prisma().delete_many(where={"id": {"in": created}})
    if owns_connection:
        await db.disconnect()


def offer(max_active_trials: int | None) -> AcceptedTrialOffer:
    return AcceptedTrialOffer(
        version="capacity-integration-v1",
        new_users_from=datetime(2026, 9, 1, tzinfo=UTC),
        duration_days=7,
        tier="PRO",
        billing_cycle="monthly",
        daily_cost_limit=250_000,
        weekly_cost_limit=1_000_000,
        total_cost_limit=1_000_000,
        onboarding_credit_amount=300,
        price_id="price_integration",
        unit_amount=2000,
        currency="usd",
        max_active_trials=max_active_trials,
    )


async def reserve(user_id: str, terms: AcceptedTrialOffer) -> TrialState:
    return await reserve_subscription_trial(
        user_id,
        terms,
        "cus_integration",
        "https://example.com/success",
        "https://example.com/cancel",
        {},
    )


@pytest.mark.asyncio
async def test_two_enrolments_racing_for_the_last_seat_admit_exactly_one(new_user):
    first, second = await new_user(), await new_user()
    # Whatever else this database holds, exactly one seat is left.
    capped = offer(await count_trial_seats() + 1)

    # Each enrolment, having read the count, waits for the other to read it
    # too before inserting. Without the capacity lock both read the same free
    # seat and both insert; with it, the second cannot count until the first
    # has committed, so it waits out the pause and then finds the trial full.
    counted = 0
    both_counted = asyncio.Event()
    real_seat_available = trials.trial_seat_available

    async def count_then_pause(*args, **kwargs):
        nonlocal counted
        free = await real_seat_available(*args, **kwargs)
        counted += 1
        if counted == 2:
            both_counted.set()
        try:
            await asyncio.wait_for(both_counted.wait(), timeout=2)
        except TimeoutError:
            pass
        return free

    with patch.object(trials, "trial_seat_available", count_then_pause):
        results = await asyncio.gather(
            reserve(first, capped), reserve(second, capped), return_exceptions=True
        )

    winners = [r for r in results if isinstance(r, TrialState)]
    refused = [r for r in results if isinstance(r, TrialCapacityReached)]
    assert (len(winners), len(refused)) == (1, 1), results
    assert await count_trial_seats() == capped.max_active_trials
    assert (
        await SubscriptionTrial.prisma().count(
            where={"userId": {"in": [first, second]}}
        )
        == 1
    )


@pytest.mark.asyncio
async def test_a_full_trial_keeps_its_holder_and_frees_an_abandoned_checkout(
    new_user,
):
    held = await reserve(await new_user(), offer(None))
    full = offer(await count_trial_seats())

    assert await trial_seat_available(full) is False
    # the holder is not crowded out by its own seat
    assert await trial_seat_available(full, trial_id=held.id) is True
    assert await count_trial_seats(exclude_trial_id=held.id) == (
        full.max_active_trials - 1
    )
    # pausing enrolment (cap 0) refuses newcomers but keeps the holder
    assert await trial_seat_available(offer(0)) is False
    assert await trial_seat_available(offer(0), trial_id=held.id) is True

    # A checkout left open past the reservation window stops holding a seat.
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"SubscriptionTrial" '
        'SET "updatedAt" = NOW() - interval \'31 minutes\' WHERE "id" = $1',
        held.id,
    )
    assert await count_trial_seats() == full.max_active_trials - 1
    assert await trial_seat_available(full) is True


@pytest.mark.asyncio
async def test_the_stored_offer_leaves_the_cap_out(new_user):
    """Rows must stay readable by code that predates the cap, so a revert of
    the cap does not strand every trial enrolled while it was deployed."""
    held = await reserve(await new_user(), offer(500))
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": held.id})
    assert isinstance(row.offer, dict)
    assert "max_active_trials" not in row.offer
    assert held.offer.token == offer(500).token
