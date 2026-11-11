"""Reading a billing period off a Stripe subscription, and the on-demand
re-read that only ever writes the clock."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import usage_clock, usage_clock_backfill
from backend.data.usage_clock import (
    BillingPeriod,
    billing_period,
    refresh_billing_clock,
    save_billing_clock,
)
from backend.data.usage_clock_backfill import backfill_billing_clocks

START = 1_793_824_256
END = START + 30 * 86_400


def _subscription(**fields: Any) -> dict:
    return {
        "id": "sub_1",
        "customer": "cus_1",
        "status": "active",
        "current_period_start": START,
        "current_period_end": END,
        "billing_cycle_anchor": START - 86_400,
        "created": START - 90 * 86_400,
        "metadata": {"user_id": "user-1"},
        **fields,
    }


def test_the_period_is_read_from_the_subscription():
    assert billing_period(_subscription()) == BillingPeriod(
        subscription_id="sub_1",
        start=START,
        end=END,
        anchor=START - 86_400,
        created=START - 90 * 86_400,
    )


def test_a_subscription_without_its_creation_time_is_dated_from_its_period():
    period = billing_period(_subscription(created=None))

    assert period is not None and period.created == START


def test_a_newer_stripe_api_keeps_the_period_on_the_item():
    """From API version 2025-03-31 the period is on each subscription item."""
    newer = _subscription(
        current_period_start=None,
        current_period_end=None,
        items={"data": [{"current_period_start": START, "current_period_end": END}]},
    )

    period = billing_period(newer)

    assert period is not None and (period.start, period.end) == (START, END)


@pytest.mark.parametrize(
    "broken",
    [
        {"id": None},
        {"current_period_start": None},
        {"current_period_end": None},
        {"current_period_start": float(START)},
        {"current_period_start": str(START)},
        {"current_period_start": True},
        {"current_period_start": START * 1000, "current_period_end": END * 1000},
        {"current_period_start": 5},
        {"current_period_end": START},  # ends when it starts
        {"current_period_end": START - 10},  # ends before it starts
        {"current_period_end": START + 60},  # a minute long
    ],
)
def test_a_period_that_is_not_sane_seconds_is_no_period(broken: dict):
    assert billing_period(_subscription(**broken)) is None


def test_an_unusable_anchor_is_left_out_not_fatal():
    period = billing_period(_subscription(billing_cycle_anchor="soon"))

    assert period is not None and period.anchor is None


@pytest.mark.asyncio
async def test_no_period_writes_nothing(monkeypatch: pytest.MonkeyPatch):
    execute = AsyncMock()
    monkeypatch.setattr(usage_clock, "execute_raw_with_schema", execute)

    assert await save_billing_clock("user-1", _subscription(id=None)) is False

    execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("rows,saved", [(1, True), (0, False)])
async def test_saving_reports_whether_the_row_was_written(
    monkeypatch: pytest.MonkeyPatch, rows: int, saved: bool
):
    execute = AsyncMock(return_value=rows)
    monkeypatch.setattr(usage_clock, "execute_raw_with_schema", execute)

    assert await save_billing_clock("user-1", _subscription()) is saved

    statement, *values = execute.await_args.args
    assert values == [
        "user-1",
        "sub_1",
        START,
        END,
        START - 86_400,
        START - 90 * 86_400,
    ]
    # Only forwards: an earlier period never replaces a later one.
    assert 'c."periodStart" <= EXCLUDED."periodStart"' in statement


class Stripe:
    """The user row and Stripe's subscription lists for a re-read."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.user: MagicMock | None = MagicMock(
            stripeCustomerId="cus_1", subscriptionTier=SubscriptionTier.PRO
        )
        self.by_status: dict[str, list[dict]] = {"active": [], "trialing": []}
        self.save = AsyncMock(return_value=True)
        self.list = AsyncMock(side_effect=self._list)
        monkeypatch.setattr(usage_clock, "save_billing_clock", self.save)
        monkeypatch.setattr(stripe.Subscription, "list_async", self.list)
        self.user_table = MagicMock(
            find_unique=AsyncMock(side_effect=lambda **_: self.user)
        )
        monkeypatch.setattr(usage_clock.User, "prisma", lambda: self.user_table)

    async def _list(self, **kwargs: Any) -> MagicMock:
        assert kwargs["customer"] == "cus_1"
        return MagicMock(data=self.by_status[kwargs["status"]])


@pytest.fixture
def stripe_account(monkeypatch: pytest.MonkeyPatch) -> Stripe:
    return Stripe(monkeypatch)


@pytest.mark.asyncio
async def test_a_re_read_copies_the_live_subscriptions_period(stripe_account: Stripe):
    stripe_account.by_status["active"] = [_subscription()]

    assert await refresh_billing_clock("user-1") == "saved"

    stripe_account.save.assert_awaited_once_with("user-1", _subscription())


@pytest.mark.asyncio
async def test_a_re_read_that_finds_nothing_newer_says_so(stripe_account: Stripe):
    stripe_account.by_status["active"] = [_subscription()]
    stripe_account.save.return_value = False

    assert await refresh_billing_clock("user-1") == "unchanged"


@pytest.mark.asyncio
async def test_no_live_subscription_is_an_answer_not_an_error(stripe_account: Stripe):
    assert await refresh_billing_clock("user-1") == "no_subscription"

    stripe_account.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "other",
    [
        _subscription(metadata={"user_id": "someone-else"}),
        _subscription(customer="cus_other"),
        _subscription(metadata={"user_id": "user-1", "trial_enrollment_id": "t1"}),
    ],
)
async def test_a_subscription_that_is_not_this_users_plan_is_never_copied(
    stripe_account: Stripe, other: dict
):
    stripe_account.by_status["trialing"] = [other]

    assert await refresh_billing_clock("user-1") == "no_subscription"

    stripe_account.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user",
    [
        None,
        MagicMock(stripeCustomerId=None, subscriptionTier=SubscriptionTier.PRO),
        MagicMock(
            stripeCustomerId="cus_1", subscriptionTier=SubscriptionTier.ENTERPRISE
        ),
        MagicMock(stripeCustomerId="cus_1", subscriptionTier=SubscriptionTier.TRIAL),
    ],
)
async def test_an_account_stripe_does_not_bill_asks_stripe_nothing(
    stripe_account: Stripe, user: MagicMock | None
):
    stripe_account.user = user

    assert await refresh_billing_clock("user-1") == "not_stripe_billed"

    stripe_account.list.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_stripe_failure_is_raised_so_it_is_not_taken_for_no_subscription(
    stripe_account: Stripe,
):
    stripe_account.list.side_effect = stripe.APIConnectionError("down")

    with pytest.raises(stripe.APIConnectionError):
        await refresh_billing_clock("user-1")


@pytest.mark.asyncio
async def test_a_re_read_never_writes_a_tier(stripe_account: Stripe):
    """It cannot downgrade, upgrade or revert anyone: tiers change only
    through the webhooks and the sweep."""
    stripe_account.by_status["active"] = [_subscription()]

    await refresh_billing_clock("user-1")

    # The user row is read once and never written.
    assert [call[0] for call in stripe_account.user_table.method_calls] == [
        "find_unique"
    ]


class Listing:
    """Every live subscription in Stripe, and the users they belong to."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.subscriptions = [
            _subscription(id="sub_a", customer="cus_a", metadata={}),
            _subscription(id="sub_b", customer="cus_b", metadata={}),
            _subscription(id="sub_nobody", customer="cus_gone", metadata={}),
            _subscription(id="sub_other", customer="cus_a", metadata={"user_id": "x"}),
            _subscription(
                id="sub_trial", customer="cus_b", metadata={"trial_enrollment_id": "t"}
            ),
            _subscription(id="sub_broken", customer="cus_a", current_period_end=None),
        ]
        self.save = AsyncMock(return_value=True)
        self.users = AsyncMock(
            return_value=[
                {"id": "user-a", "customer": "cus_a"},
                {"id": "user-b", "customer": "cus_b"},
            ]
        )
        monkeypatch.setattr(usage_clock_backfill, "save_billing_clock", self.save)
        monkeypatch.setattr(usage_clock_backfill, "query_raw_with_schema", self.users)
        monkeypatch.setattr(
            stripe.Subscription, "list_async", AsyncMock(side_effect=self._list)
        )

    async def _list(self, **kwargs: Any) -> MagicMock:
        found = self.subscriptions if kwargs["status"] == "active" else []
        return MagicMock(data=found, has_more=False)


@pytest.fixture
def listing(monkeypatch: pytest.MonkeyPatch) -> Listing:
    return Listing(monkeypatch)


@pytest.mark.asyncio
async def test_the_backfill_copies_each_owned_plan_and_skips_the_rest(listing: Listing):
    listing.save.side_effect = [True, False]

    summary = await backfill_billing_clocks()

    assert [call.args[0] for call in listing.save.await_args_list] == [
        "user-a",
        "user-b",
    ]
    assert (summary.subscriptions, summary.saved, summary.unchanged) == (6, 1, 1)
    assert (summary.skipped, summary.failed) == (4, 0)
    # Admin-managed accounts are left out, as in the tier sweep.
    assert "<> 'ENTERPRISE'" in listing.users.await_args.args[0]


@pytest.mark.asyncio
async def test_a_dry_run_counts_without_writing(listing: Listing):
    summary = await backfill_billing_clocks(dry_run=True)

    assert summary.saved == 2
    listing.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_failed_save_does_not_stop_the_rest(listing: Listing):
    listing.save.side_effect = [RuntimeError("database"), True]

    summary = await backfill_billing_clocks()

    assert (summary.saved, summary.failed) == (1, 1)
