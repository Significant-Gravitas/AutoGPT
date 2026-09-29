from datetime import UTC, datetime, timedelta
from itertools import permutations

import pytest
from prisma.enums import SubscriptionTier

from backend.data.posthog_lifecycle import (
    LifecycleSnapshot,
    LifecycleUser,
    StripeSubscriptionFacts,
    compute_lifecycle_snapshot,
)
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.data.subscription_trial_rejection import TrialRejectionReason

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
SIGNUP = NOW - timedelta(days=90)
TRIAL_START = NOW - timedelta(days=30)


def user(
    tier: SubscriptionTier = SubscriptionTier.NO_TIER, customer: str | None = "cus_1"
) -> LifecycleUser:
    return LifecycleUser(
        id="user-1",
        created_at=SIGNUP,
        stripe_customer_id=customer,
        subscription_tier=tier,
    )


def trial(**overrides) -> TrialState:
    """A trial that started 30 days ago and is still running."""
    values = {
        "id": "trial-1",
        "user_id": "user-1",
        "customer_id": "cus_1",
        "offer": AcceptedTrialOffer(
            version="v1",
            new_users_from=SIGNUP,
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=1,
            weekly_cost_limit=1,
            total_cost_limit=1,
            onboarding_credit_amount=0,
            price_id="price_pro",
            unit_amount=5000,
            currency="usd",
        ),
        "checkout_session_id": "cs_1",
        "subscription_id": "sub_trial",
        "checkout_attempt": 0,
        "success_url": "https://example.com/ok",
        "cancel_url": "https://example.com/no",
        "checkout_metadata": {},
        "status": "trialing",
        "card_verified_at": TRIAL_START,
        "started_at": TRIAL_START,
        "ends_at": NOW + timedelta(days=3),
        "consumed_at": TRIAL_START,
        "converted_at": None,
        "cancel_at_period_end": False,
        "cost_microdollars": 0,
    }
    return TrialState(**{**values, **overrides})


def sub(
    sub_id: str = "sub_paid",
    status: str = "active",
    *,
    start: datetime = NOW - timedelta(days=10),
    cancel_at_period_end: bool = False,
    canceled_at: datetime | None = None,
    ended_at: datetime | None = None,
    enrollment: str | None = None,
) -> StripeSubscriptionFacts:
    return StripeSubscriptionFacts.model_validate(
        {
            "id": sub_id,
            "customer": "cus_1",
            "status": status,
            "cancel_at_period_end": cancel_at_period_end,
            "start_date": int(start.timestamp()),
            "created": int(start.timestamp()),
            "canceled_at": int(canceled_at.timestamp()) if canceled_at else None,
            "ended_at": int(ended_at.timestamp()) if ended_at else None,
            "metadata": {"trial_enrollment_id": enrollment} if enrollment else {},
        }
    )


def snapshot(
    *,
    u: LifecycleUser | None = None,
    t: TrialState | None = None,
    subs: list[StripeSubscriptionFacts] | None = None,
) -> LifecycleSnapshot:
    return compute_lifecycle_snapshot(
        user=u or user(), trial=t, subscriptions=subs or [], now=NOW
    )


def test_signed_up_only():
    result = snapshot(u=user(customer=None))
    assert result == LifecycleSnapshot(subscription_status="signed", signup_at=SIGNUP)


def test_customer_without_subscriptions_is_signed():
    # A Stripe customer is created for credit top-ups too.
    assert snapshot().subscription_status == "signed"


def test_unfinished_trial_checkout_is_signed():
    result = snapshot(
        t=trial(status="checkout_pending", consumed_at=None, card_verified_at=None)
    )
    assert result.subscription_status == "signed"
    assert result.trial_started_at is None


def test_rejected_trial_is_signed():
    result = snapshot(
        t=trial(
            status="canceled",
            rejection_reason=TrialRejectionReason.INTRO_OFFER_ALREADY_USED,
        )
    )
    assert result.subscription_status == "signed"
    assert result.trial_started_at is None


def test_in_trial():
    result = snapshot(
        t=trial(), subs=[sub("sub_trial", "trialing", enrollment="trial-1")]
    )
    assert result == LifecycleSnapshot(
        subscription_status="in_trial", signup_at=SIGNUP, trial_started_at=TRIAL_START
    )


def test_trial_with_lapsed_card_is_still_in_trial():
    assert snapshot(t=trial(card_verified_at=None)).subscription_status == "in_trial"


def test_trial_set_to_cancel():
    result = snapshot(t=trial(cancel_at_period_end=True))
    assert result.subscription_status == "trial_canceled"
    assert result.trial_started_at == TRIAL_START


def test_trial_canceled_immediately():
    result = snapshot(
        t=trial(status="canceled", ends_at=NOW - timedelta(days=1)),
        subs=[sub("sub_trial", "canceled", enrollment="trial-1")],
    )
    assert result.subscription_status == "trial_canceled"


def test_trial_that_ran_out_on_the_clock():
    # The webhook for the trial end hasn't landed; the clock alone ends it.
    result = snapshot(t=trial(ends_at=NOW - timedelta(minutes=1)))
    assert result.subscription_status == "trial_canceled"


def test_failed_conversion_is_not_a_paid_subscription():
    result = snapshot(
        t=trial(status="past_due", ends_at=NOW - timedelta(days=1)),
        subs=[sub("sub_trial", "past_due", enrollment="trial-1")],
    )
    assert result.subscription_status == "trial_canceled"
    assert result.subscription_started_at is None


def test_converted_trial_starts_its_subscription_at_conversion():
    converted = NOW - timedelta(days=20)
    result = snapshot(
        u=user(SubscriptionTier.PRO),
        t=trial(status="active", converted_at=converted),
        subs=[sub("sub_trial", "active", start=TRIAL_START, enrollment="trial-1")],
    )
    assert result == LifecycleSnapshot(
        subscription_status="subscribed",
        signup_at=SIGNUP,
        trial_started_at=TRIAL_START,
        subscription_started_at=converted,
    )


def test_converted_trial_then_canceled_and_ended():
    converted = NOW - timedelta(days=20)
    canceled = NOW - timedelta(days=5)
    ended = NOW - timedelta(days=1)
    t = trial(status="canceled", converted_at=converted)
    canceling = sub(
        "sub_trial",
        "active",
        start=TRIAL_START,
        enrollment="trial-1",
        cancel_at_period_end=True,
        canceled_at=canceled,
    )
    result = snapshot(t=t, subs=[canceling])
    assert result.subscription_status == "subscription_canceled"
    assert result.subscription_canceled_at == canceled
    assert result.subscription_ended_at is None

    gone = sub(
        "sub_trial",
        "canceled",
        start=TRIAL_START,
        enrollment="trial-1",
        canceled_at=canceled,
        ended_at=ended,
    )
    result = snapshot(t=t, subs=[gone])
    assert result.subscription_status == "subscription_ended"
    assert result.subscription_started_at == converted
    assert result.subscription_canceled_at == canceled
    assert result.subscription_ended_at == ended


def test_converted_trial_missing_from_stripe_still_counts_as_ended():
    converted = NOW - timedelta(days=20)
    result = snapshot(t=trial(status="canceled", converted_at=converted))
    assert result.subscription_status == "subscription_ended"
    assert result.subscription_started_at == converted


def test_subscription_from_another_trial_attempt_is_not_paid():
    stale = sub("sub_old_attempt", "active", enrollment="trial-1")
    result = snapshot(t=trial(ends_at=NOW - timedelta(days=1)), subs=[stale])
    assert result.subscription_status == "trial_canceled"


def test_subscribed():
    start = NOW - timedelta(days=10)
    result = snapshot(u=user(SubscriptionTier.PRO), subs=[sub(start=start)])
    assert result == LifecycleSnapshot(
        subscription_status="subscribed",
        signup_at=SIGNUP,
        subscription_started_at=start,
    )


@pytest.mark.parametrize("status", ["past_due", "unpaid"])
def test_a_failed_renewal_is_payment_failed(status: str):
    # Lost access (the backend drops them to NO_TIER), but didn't choose to leave.
    start = NOW - timedelta(days=40)
    result = snapshot(subs=[sub(status=status, start=start)])
    assert result == LifecycleSnapshot(
        subscription_status="payment_failed",
        signup_at=SIGNUP,
        subscription_started_at=start,
    )


def test_a_canceled_subscription_after_a_failed_payment_has_ended():
    # Stripe cancels after its retries run out; the newest state wins.
    ended = NOW - timedelta(days=1)
    result = snapshot(subs=[sub(status="canceled", canceled_at=ended, ended_at=ended)])
    assert result.subscription_status == "subscription_ended"


def test_an_active_subscription_wins_over_an_old_past_due_one():
    result = snapshot(subs=[sub("sub_old", "past_due"), sub("sub_new", "active")])
    assert result.subscription_status == "subscribed"


def test_subscription_set_to_cancel():
    canceled = NOW - timedelta(days=2)
    result = snapshot(
        subs=[sub(cancel_at_period_end=True, canceled_at=canceled)],
    )
    assert result.subscription_status == "subscription_canceled"
    assert result.subscription_canceled_at == canceled
    assert result.subscription_ended_at is None


def test_subscription_ended():
    canceled = NOW - timedelta(days=5)
    ended = NOW - timedelta(days=1)
    result = snapshot(
        subs=[sub(status="canceled", canceled_at=canceled, ended_at=ended)]
    )
    assert result.subscription_status == "subscription_ended"
    assert result.subscription_canceled_at == canceled
    assert result.subscription_ended_at == ended


def test_never_paid_subscriptions_are_ignored():
    subs = [sub("a", "incomplete"), sub("b", "incomplete_expired")]
    assert snapshot(subs=subs).subscription_status == "signed"


def test_resubscribe_after_ended_clears_the_old_dates():
    old = sub(
        "sub_old",
        "canceled",
        start=NOW - timedelta(days=200),
        canceled_at=NOW - timedelta(days=100),
        ended_at=NOW - timedelta(days=90),
    )
    new_start = NOW - timedelta(days=3)
    new = sub("sub_new", "active", start=new_start)
    result = snapshot(u=user(SubscriptionTier.PRO), subs=[old, new])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == new_start
    _, unset = result.person_update()
    assert "subscription_canceled_at" in unset
    assert "subscription_ended_at" in unset


def test_latest_ended_subscription_supplies_the_dates():
    first = sub(
        "sub_1",
        "canceled",
        start=NOW - timedelta(days=300),
        ended_at=NOW - timedelta(days=200),
    )
    latest_start = NOW - timedelta(days=100)
    latest_end = NOW - timedelta(days=10)
    latest = sub("sub_2", "canceled", start=latest_start, ended_at=latest_end)
    for order in permutations([first, latest]):
        result = snapshot(subs=list(order))
        assert result.subscription_started_at == latest_start
        assert result.subscription_ended_at == latest_end


@pytest.mark.parametrize("order", list(permutations(range(2))))
def test_out_of_order_delete_of_old_subscription(order):
    """A late ``deleted`` for the old sub after ``created`` for the new one.

    The mapping reads every subscription as it is now, so which event came
    last (and the order Stripe lists them in) can't change the answer.
    """
    old = sub(
        "sub_old",
        "canceled",
        start=NOW - timedelta(days=60),
        canceled_at=NOW - timedelta(days=1),
        ended_at=NOW - timedelta(days=1),
    )
    new = sub("sub_new", "active", start=NOW - timedelta(days=2))
    subs = [[old, new][i] for i in order]
    result = snapshot(u=user(SubscriptionTier.PRO), subs=subs)
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == new.started_at
    assert result.subscription_canceled_at is None


def test_two_live_subscriptions_keep_the_one_not_being_canceled():
    leaving = sub("sub_a", start=NOW - timedelta(days=1), cancel_at_period_end=True)
    staying = sub("sub_b", start=NOW - timedelta(days=30))
    result = snapshot(subs=[leaving, staying])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == staying.started_at


def test_paid_subscription_wins_over_a_running_trial():
    result = snapshot(t=trial(), subs=[sub()])
    assert result.subscription_status == "subscribed"
    assert result.trial_started_at == TRIAL_START


@pytest.mark.parametrize("customer", [None, "cus_1"])
def test_enterprise_is_subscribed(customer):
    ended = sub(status="canceled", ended_at=NOW - timedelta(days=1))
    result = snapshot(u=user(SubscriptionTier.ENTERPRISE, customer), subs=[ended])
    assert result == LifecycleSnapshot(
        subscription_status="subscribed", signup_at=SIGNUP
    )


def test_paid_tier_without_stripe_customer_is_a_manual_grant():
    result = snapshot(u=user(SubscriptionTier.BUSINESS, customer=None))
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at is None


def test_stale_paid_tier_with_stripe_customer_follows_stripe():
    # Stripe owns this tier (the reconcile sweep will downgrade it), so a
    # missing subscription is not a manual grant.
    assert snapshot(u=user(SubscriptionTier.PRO)).subscription_status == "signed"


def test_person_update_sets_utc_iso_dates_and_unsets_the_rest():
    local = datetime(2026, 9, 1, 14, 30, tzinfo=UTC) + timedelta(microseconds=5)
    result = LifecycleSnapshot(
        subscription_status="in_trial",
        signup_at=local,
        trial_started_at=datetime(2026, 9, 2, 8, 0),
    )
    to_set, to_unset = result.person_update()
    assert to_set == {
        "subscription_status": "in_trial",
        "signup_at": "2026-09-01T14:30:00+00:00",
        "trial_started_at": "2026-09-02T08:00:00+00:00",
    }
    assert to_unset == [
        "subscription_started_at",
        "subscription_canceled_at",
        "subscription_ended_at",
    ]
