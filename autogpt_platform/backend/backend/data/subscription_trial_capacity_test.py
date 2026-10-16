from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import subscription_trial_capacity as capacity
from backend.data.subscription_trial_config import TrialOffer


def offer(max_active_trials: int | None) -> TrialOffer:
    return TrialOffer.model_validate(
        {
            "version": "capacity-v1",
            "new_users_from": datetime.now(UTC) - timedelta(days=1),
            "duration_days": 7,
            "tier": "PRO",
            "billing_cycle": "monthly",
            "daily_cost_limit": 250_000,
            "weekly_cost_limit": 1_000_000,
            "total_cost_limit": 1_000_000,
            "onboarding_credit_amount": 300,
            "max_active_trials": max_active_trials,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cap,taken,expected",
    [
        (10, 9, True),
        (10, 10, False),
        (10, 11, False),  # a lowered cap refuses new seats, never revokes
        (1, 0, True),
        (1, 1, False),
    ],
)
async def test_seat_is_free_only_below_the_cap(cap, taken, expected):
    with patch.object(
        capacity, "count_trial_seats", AsyncMock(return_value=taken)
    ), patch.object(capacity, "_holds_seat", AsyncMock(return_value=False)):
        assert await capacity.trial_seat_available(offer(cap)) is expected


@pytest.mark.asyncio
async def test_uncapped_offer_never_counts_seats():
    """An uncapped trial must not pay for a COUNT on every status check."""
    with patch.object(capacity, "count_trial_seats", AsyncMock()) as count:
        assert await capacity.trial_seat_available(offer(None)) is True
    count.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_cap_closes_enrolment_without_a_query():
    with patch.object(
        capacity, "count_trial_seats", AsyncMock()
    ) as count, patch.object(capacity, "_holds_seat", AsyncMock(return_value=False)):
        assert await capacity.trial_seat_available(offer(0)) is False
        assert (
            await capacity.trial_seat_available(offer(0), trial_id="expired-1") is False
        )
    count.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_cap_keeps_a_held_seat():
    """Pausing enrolment must not strand someone already at the card screen."""
    with patch.object(
        capacity, "count_trial_seats", AsyncMock()
    ) as count, patch.object(
        capacity, "_holds_seat", AsyncMock(return_value=True)
    ) as holds:
        assert await capacity.trial_seat_available(offer(0), trial_id="trial-1") is True
    count.assert_not_awaited()
    assert holds.await_args.args[0] == "trial-1"


@pytest.mark.asyncio
async def test_a_held_seat_is_not_revoked_by_latecomers():
    """Someone mid-checkout keeps their seat when the rest fill up."""
    with patch.object(
        capacity, "count_trial_seats", AsyncMock(return_value=10)
    ) as count, patch.object(capacity, "_holds_seat", AsyncMock(return_value=True)):
        assert (
            await capacity.trial_seat_available(offer(10), trial_id="trial-1") is True
        )
    # counted the *others*, so the holder does not crowd itself out
    assert count.await_args.kwargs["exclude_trial_id"] == "trial-1"


@pytest.mark.asyncio
async def test_full_trial_refuses_an_enrolment_holding_no_seat():
    with patch.object(
        capacity, "count_trial_seats", AsyncMock(return_value=10)
    ), patch.object(capacity, "_holds_seat", AsyncMock(return_value=False)):
        assert (
            await capacity.trial_seat_available(offer(10), trial_id="expired-1")
            is False
        )


@pytest.mark.asyncio
async def test_seat_query_counts_running_trials_and_open_checkouts():
    """The cap is meaningless if it ignores everyone at the card screen."""
    with patch.object(
        capacity, "query_raw_with_schema", AsyncMock(return_value=[])
    ) as query:
        assert await capacity.count_trial_seats() == 0
    sql = query.await_args.args[0]
    assert "'trialing'" in sql and '"cardVerifiedAt" IS NOT NULL' in sql
    assert "'checkout_pending'" in sql and '"updatedAt"' in sql
    # the reservation window is passed as a bound parameter, not interpolated
    assert query.await_args.args[2] == (
        f"{capacity.CHECKOUT_RESERVATION.total_seconds()} seconds"
    )


@pytest.mark.asyncio
async def test_capacity_lock_is_one_global_key():
    """Two enrolments must contend, or both read the same last free seat."""
    tx = AsyncMock()
    with patch.object(
        capacity, "query_raw_with_schema", AsyncMock(return_value=[])
    ) as query:
        await capacity.lock_trial_capacity(tx)
    assert "pg_advisory_xact_lock" in query.await_args.args[0]
    assert query.await_args.args[1] == capacity._CAPACITY_LOCK


@pytest.mark.asyncio
async def test_renewal_without_a_cap_takes_no_lock():
    with patch.object(capacity, "transaction") as tx:
        assert await capacity.renew_trial_seat(offer(None), "trial-1") is True
    tx.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("free", [True, False])
async def test_renewal_checks_and_restarts_the_hold_under_the_lock(free):
    """A returner is re-admitted atomically, or not at all."""
    tx = AsyncMock()
    calls: list[str] = []

    async def lock(client):
        assert client is tx
        calls.append("lock")

    async def seat(capped, *, trial_id, client):
        assert (trial_id, client) == ("trial-1", tx)
        calls.append("check")
        return free

    async def renew(sql, trial_id, *, client):
        assert "checkout_pending" in sql and '"updatedAt" = NOW()' in sql
        assert (trial_id, client) == ("trial-1", tx)
        calls.append("renew")
        return 1

    transaction = MagicMock()
    transaction.return_value.__aenter__.return_value = tx
    with (
        patch.object(capacity, "transaction", transaction),
        patch.object(capacity, "lock_trial_capacity", lock),
        patch.object(capacity, "trial_seat_available", seat),
        patch.object(capacity, "execute_raw_with_schema", renew),
    ):
        assert await capacity.renew_trial_seat(offer(5), "trial-1") is free
    assert calls == (["lock", "check", "renew"] if free else ["lock", "check"])
