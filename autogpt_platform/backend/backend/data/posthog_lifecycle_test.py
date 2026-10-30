from datetime import UTC, datetime, timedelta
from itertools import permutations

import pytest

from backend.data.credit import REPLACED_PLAN_CANCELLATION_COMMENT
from backend.data.posthog_lifecycle import (
    LifecycleSnapshot,
    LifecycleUser,
    compute_lifecycle_snapshot,
)
from backend.data.stripe_subscription_facts import StripeSubscriptionFacts
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.data.subscription_trial_rejection import TrialRejectionReason

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
SIGNUP = NOW - timedelta(days=90)
TRIAL_START = NOW - timedelta(days=30)


def user(customer: str | None = "cus_1") -> LifecycleUser:
    return LifecycleUser(id="user-1", created_at=SIGNUP, stripe_customer_id=customer)


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
    trial_end: datetime | None = None,
    **stripe_fields,
) -> StripeSubscriptionFacts:
    """``stripe_fields`` are more of Stripe's own, as a list call returns
    them with ``data.discounts`` and ``data.customer`` expanded."""
    return StripeSubscriptionFacts.model_validate(
        {
            "items": {"data": [{"price": {"unit_amount": 5000}, "quantity": 1}]},
            "id": sub_id,
            "customer": "cus_1",
            "status": status,
            "cancel_at_period_end": cancel_at_period_end,
            "start_date": int(start.timestamp()),
            "created": int(start.timestamp()),
            "canceled_at": int(canceled_at.timestamp()) if canceled_at else None,
            "ended_at": int(ended_at.timestamp()) if ended_at else None,
            "trial_start": int(start.timestamp()) if trial_end else None,
            "trial_end": int(trial_end.timestamp()) if trial_end else None,
            "metadata": {"trial_enrollment_id": enrollment} if enrollment else {},
            **stripe_fields,
        }
    )


def coupon(
    *, percent_off: float | None = None, amount_off: int | None = None, **discount
) -> dict:
    """A Stripe discount object, as ``data.discounts`` expands it."""
    return {
        "object": "discount",
        "coupon": {"percent_off": percent_off, "amount_off": amount_off},
        "end": None,
        **discount,
    }


def by_request(**details) -> dict:
    """Canceled through the API with no comment, as when the subscriber asks."""
    return {"cancellation_details": {"reason": "cancellation_requested", **details}}


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
        subs=[sub("sub_trial", "canceled", enrollment="trial-1", **by_request())],
    )
    assert result.subscription_status == "trial_canceled"


def test_a_trial_set_to_cancel_that_ran_out_stays_trial_canceled():
    result = snapshot(
        t=trial(
            status="canceled",
            ends_at=NOW - timedelta(days=1),
            cancel_at_period_end=True,
        ),
        subs=[sub("sub_trial", "canceled", enrollment="trial-1")],
    )
    assert result.subscription_status == "trial_canceled"


def test_trial_that_ran_out_on_the_clock_has_expired():
    # The webhook for the trial end hasn't landed; the clock alone ends it.
    # Nobody canceled it, so it ran out rather than being canceled.
    result = snapshot(t=trial(ends_at=NOW - timedelta(minutes=1)))
    assert result.subscription_status == "trial_expired"
    assert result.trial_started_at == TRIAL_START


@pytest.mark.parametrize("status", ["past_due", "unpaid"])
def test_a_failed_first_charge_after_the_trial_is_payment_failed(status: str):
    # Stripe is still retrying, as with a failed renewal; the backend sends
    # payment_failed for this state too. It was never paid, so no start date.
    result = snapshot(
        t=trial(status=status, ends_at=NOW - timedelta(days=1)),
        subs=[sub("sub_trial", status, enrollment="trial-1")],
    )
    assert result.subscription_status == "payment_failed"
    assert result.subscription_started_at is None


@pytest.mark.parametrize(
    "details",
    [
        # Stripe gave up retrying the first charge.
        {"reason": "payment_failed"},
        # Our own cancel after the charge failed carries a comment.
        {"reason": "cancellation_requested", "comment": "autogpt:payment_failed"},
    ],
)
def test_a_trial_whose_first_charge_never_cleared_has_expired(details: dict):
    result = snapshot(
        t=trial(status="canceled", ends_at=NOW - timedelta(days=1)),
        subs=[
            sub(
                "sub_trial",
                "canceled",
                enrollment="trial-1",
                cancellation_details=details,
            )
        ],
    )
    assert result.subscription_status == "trial_expired"
    assert result.subscription_started_at is None


@pytest.mark.parametrize(
    "comment, expected",
    [
        # Written by the customer in the billing portal's cancel flow.
        ("Too expensive for me right now", "trial_canceled"),
        ("autogpt_trial:intro_offer_already_used", "trial_expired"),
        ("autogpt:replaced_by_another_plan", "trial_expired"),
    ],
)
def test_only_our_own_cancel_comments_mean_we_canceled(comment, expected):
    result = snapshot(
        t=trial(status="canceled", ends_at=NOW - timedelta(days=1)),
        subs=[
            sub(
                "sub_trial",
                "canceled",
                enrollment="trial-1",
                **by_request(comment=comment),
            )
        ],
    )
    assert result.subscription_status == expected


def test_every_comment_we_cancel_with_is_recognised_as_ours():
    # Plans cancelled for non-payment before #15345 still carry
    # ``autogpt:payment_failed``.
    ours = ["autogpt:payment_failed", REPLACED_PLAN_CANCELLATION_COMMENT] + [
        reason.stripe_comment for reason in TrialRejectionReason
    ]
    for comment in ours:
        facts = sub(status="canceled", **by_request(comment=comment))
        assert not facts.cancel_requested, comment


def test_trial_awaiting_its_first_charge_is_still_in_trial():
    """Stripe makes the subscription ``active`` at trial end and charges the
    draft invoice about an hour later; until then it hasn't converted."""
    t = trial(status="active", ends_at=NOW - timedelta(minutes=10))
    live = sub("sub_trial", "active", start=TRIAL_START, enrollment="trial-1")
    result = snapshot(t=t, subs=[live])
    assert result == LifecycleSnapshot(
        subscription_status="in_trial", signup_at=SIGNUP, trial_started_at=TRIAL_START
    )

    converted = NOW - timedelta(minutes=1)
    paid = trial(
        status="active", ends_at=NOW - timedelta(minutes=10), converted_at=converted
    )
    result = snapshot(t=paid, subs=[live])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == converted


def test_converted_trial_starts_its_subscription_at_conversion():
    converted = NOW - timedelta(days=20)
    result = snapshot(
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
    assert result.subscription_status == "trial_expired"


def test_subscribed():
    start = NOW - timedelta(days=10)
    result = snapshot(subs=[sub(start=start)])
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


def test_resubscribe_after_a_gap_keeps_the_first_paid_start():
    first_start = NOW - timedelta(days=200)
    old = sub(
        "sub_old",
        "canceled",
        start=first_start,
        canceled_at=NOW - timedelta(days=100),
        ended_at=NOW - timedelta(days=90),
    )
    new = sub("sub_new", "active", start=NOW - timedelta(days=3))
    result = snapshot(subs=[old, new])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == first_start
    _, unset = result.person_update()
    assert "subscription_canceled_at" in unset
    assert "subscription_ended_at" in unset


def test_latest_ended_subscription_supplies_the_end_and_the_first_the_start():
    first_start = NOW - timedelta(days=300)
    first = sub(
        "sub_1", "canceled", start=first_start, ended_at=NOW - timedelta(days=200)
    )
    latest_end = NOW - timedelta(days=10)
    latest = sub(
        "sub_2", "canceled", start=NOW - timedelta(days=100), ended_at=latest_end
    )
    for order in permutations([first, latest]):
        result = snapshot(subs=list(order))
        assert result.subscription_started_at == first_start
        assert result.subscription_ended_at == latest_end


def test_a_converted_trial_then_a_resubscribe_keeps_the_conversion():
    converted = NOW - timedelta(days=60)
    t = trial(status="canceled", converted_at=converted)
    ended = sub(
        "sub_trial",
        "canceled",
        start=TRIAL_START - timedelta(days=40),
        enrollment="trial-1",
        ended_at=NOW - timedelta(days=20),
    )
    again = sub("sub_new", "active", start=NOW - timedelta(days=5))
    result = snapshot(t=t, subs=[ended, again])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == converted


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
    result = snapshot(subs=subs)
    assert result.subscription_status == "subscribed"
    # The first paid start stays, whichever subscription is the current one.
    assert result.subscription_started_at == old.started_at
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


def test_failed_renewal_after_an_upgrade_beats_the_older_canceled_sub():
    """The old sub is canceled after the new one starts (stale cleanup), so
    end dates would pick the old one; the newer start wins."""
    new_start = NOW - timedelta(days=31)
    old = sub(
        "old",
        "canceled",
        start=NOW - timedelta(days=60),
        canceled_at=NOW - timedelta(days=30),
        ended_at=NOW - timedelta(days=30),
    )
    new = sub("new", "past_due", start=new_start)
    for order in permutations([old, new]):
        result = snapshot(subs=list(order))
        assert result.subscription_status == "payment_failed"
        assert result.subscription_started_at == old.started_at
        assert result.subscription_ended_at is None


def test_a_newer_ended_sub_beats_an_older_failed_one():
    # A's renewal failed and stayed past_due; the user then took out B, which
    # ended. B is where they are now.
    failed = sub("a", "past_due", start=NOW - timedelta(days=90))
    ended_at = NOW - timedelta(days=2)
    newer = sub(
        "b",
        "canceled",
        start=NOW - timedelta(days=40),
        canceled_at=ended_at,
        ended_at=ended_at,
    )
    for order in permutations([failed, newer]):
        result = snapshot(subs=list(order))
        assert result.subscription_status == "subscription_ended"
        assert result.subscription_ended_at == ended_at


def test_a_start_date_tie_goes_to_the_sub_that_ended_last():
    start = NOW - timedelta(days=40)
    early = sub("a", "canceled", start=start, ended_at=NOW - timedelta(days=20))
    late_end = NOW - timedelta(days=2)
    late = sub("b", "canceled", start=start, ended_at=late_end)
    for order in permutations([early, late]):
        assert snapshot(subs=list(order)).subscription_ended_at == late_end


def test_a_start_date_tie_between_ended_and_failed_is_payment_failed():
    start = NOW - timedelta(days=40)
    ended = sub("a", "canceled", start=start, ended_at=NOW - timedelta(days=2))
    failed = sub("b", "past_due", start=start)
    for order in permutations([ended, failed]):
        assert snapshot(subs=list(order)).subscription_status == "payment_failed"


def test_a_converted_native_trial_keeps_its_trial_start():
    start = NOW - timedelta(days=20)
    converted = sub(
        "sub_native", "active", start=start, trial_end=start + timedelta(days=7)
    )
    result = snapshot(subs=[converted])
    assert result.subscription_status == "subscribed"
    assert result.trial_started_at == start


def test_stripe_native_trial_is_in_trial():
    start = NOW - timedelta(days=2)
    native = sub(
        "sub_native", "trialing", start=start, trial_end=NOW + timedelta(days=5)
    )
    result = snapshot(subs=[native])
    assert result == LifecycleSnapshot(
        subscription_status="in_trial", signup_at=SIGNUP, trial_started_at=start
    )


def test_stripe_native_trial_set_to_cancel():
    native = sub(
        "sub_native",
        "trialing",
        cancel_at_period_end=True,
        trial_end=NOW + timedelta(days=5),
    )
    assert snapshot(subs=[native]).subscription_status == "trial_canceled"


def test_stripe_native_trial_past_its_end_is_not_in_trial():
    native = sub("sub_native", "trialing", trial_end=NOW - timedelta(hours=1))
    assert snapshot(subs=[native]).subscription_status == "signed"


def test_stripe_native_trial_canceled_before_its_first_charge():
    start = NOW - timedelta(days=10)
    native = sub(
        "sub_native",
        "canceled",
        start=start,
        trial_end=NOW - timedelta(days=3),
        canceled_at=NOW - timedelta(days=5),
        ended_at=NOW - timedelta(days=5),
        **by_request(),
    )
    result = snapshot(subs=[native])
    assert result == LifecycleSnapshot(
        subscription_status="trial_canceled",
        signup_at=SIGNUP,
        trial_started_at=start,
    )


def test_stripe_native_trial_whose_first_charge_never_cleared_has_expired():
    start = NOW - timedelta(days=10)
    native = sub(
        "sub_native",
        "canceled",
        start=start,
        trial_end=NOW - timedelta(days=3),
        canceled_at=NOW - timedelta(days=3),
        ended_at=NOW - timedelta(days=3),
        cancellation_details={"reason": "payment_failed"},
    )
    result = snapshot(subs=[native])
    assert result.subscription_status == "trial_expired"
    assert result.trial_started_at == start


def test_stripe_native_trial_that_was_paid_then_canceled_has_ended():
    ended = NOW - timedelta(days=1)
    native = sub(
        "sub_native",
        "canceled",
        start=NOW - timedelta(days=40),
        trial_end=NOW - timedelta(days=26),
        canceled_at=ended,
        ended_at=ended,
    )
    assert snapshot(subs=[native]).subscription_status == "subscription_ended"


def test_our_trial_subscription_is_not_a_native_trial():
    ours = sub(
        "sub_trial", "trialing", enrollment="trial-1", trial_end=NOW + timedelta(days=1)
    )
    assert snapshot(subs=[ours]).subscription_status == "signed"


# ---- only paying subscribers are subscribed (SECRT-2795) ---------------------


@pytest.mark.parametrize("customer", [None, "cus_1"])
def test_a_tier_without_a_paid_subscription_is_not_subscribed(customer):
    """The tier is not read at all: an admin-granted ENTERPRISE, BUSINESS or
    any other plan with no paid Stripe subscription is ``signed``. The tier
    column also defaulted to PRO for signups from 2026-03-26 to 2026-05-01,
    so a paid tier alone never meant a payment."""
    assert "subscription_tier" not in LifecycleUser.model_fields
    assert snapshot(u=user(customer)).subscription_status == "signed"


def test_an_enterprise_user_who_pays_through_stripe_is_subscribed():
    start = NOW - timedelta(days=40)
    result = snapshot(subs=[sub(start=start)])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == start


@pytest.mark.parametrize(
    "discounts",
    [
        [coupon(percent_off=100)],
        [coupon(amount_off=5000)],
        [coupon(amount_off=9900)],
        # Two discounts that together take the whole price.
        [coupon(percent_off=50), coupon(amount_off=2500)],
    ],
)
def test_a_fully_discounted_subscription_is_not_subscribed(discounts):
    result = snapshot(subs=[sub(discounts=discounts)])
    assert result == LifecycleSnapshot(subscription_status="signed", signup_at=SIGNUP)


def test_a_free_price_is_not_paid():
    free = sub(items={"data": [{"price": {"unit_amount": 0}, "quantity": 1}]})
    assert snapshot(subs=[free]).subscription_status == "signed"


def test_a_customer_wide_full_discount_is_not_paid():
    # A discount on the customer applies to a subscription without its own.
    free = sub(customer={"id": "cus_1", "discount": coupon(percent_off=100)})
    assert free.customer == "cus_1"
    assert snapshot(subs=[free]).subscription_status == "signed"


def test_a_subscription_discount_overrides_the_customers():
    partial = sub(
        customer={"id": "cus_1", "discount": coupon(percent_off=100)},
        discounts=[coupon(percent_off=20)],
    )
    assert snapshot(subs=[partial]).subscription_status == "subscribed"


@pytest.mark.parametrize("discount", [coupon(percent_off=50), coupon(amount_off=4999)])
def test_a_partial_discount_is_still_paid(discount):
    result = snapshot(subs=[sub(discounts=[discount])])
    assert result.subscription_status == "subscribed"


def test_a_full_discount_that_has_ended_is_paid_again():
    # A repeating coupon (N months): once it ends, the subscription is paid.
    ended = coupon(percent_off=100, end=int((NOW - timedelta(days=1)).timestamp()))
    running = coupon(percent_off=100, end=int((NOW + timedelta(days=1)).timestamp()))
    assert snapshot(subs=[sub(discounts=[ended])]).subscription_status == ("subscribed")
    assert snapshot(subs=[sub(discounts=[running])]).subscription_status == "signed"


def test_stacked_discounts_apply_in_their_listed_order():
    # $25 off then 50% off a $50 price leaves $12.50, as Stripe charges it;
    # 50% off then $25 off leaves nothing.
    amount_first = [coupon(amount_off=2500), coupon(percent_off=50)]
    percent_first = [coupon(percent_off=50), coupon(amount_off=2500)]
    assert snapshot(subs=[sub(discounts=amount_first)]).subscription_status == (
        "subscribed"
    )
    assert snapshot(subs=[sub(discounts=percent_first)]).subscription_status == (
        "signed"
    )


def test_a_customer_coupon_given_later_keeps_an_old_paid_subscription():
    # Paid, ended, and only then a 100% coupon on the customer: the old
    # subscription was paid, so the person has ended, not merely signed up.
    start = NOW - timedelta(days=100)
    ended_at = NOW - timedelta(days=50)
    later = coupon(percent_off=100, start=int((NOW - timedelta(days=5)).timestamp()))
    old = sub(
        status="canceled",
        start=start,
        canceled_at=ended_at,
        ended_at=ended_at,
        customer={"id": "cus_1", "discount": later},
    )
    result = snapshot(subs=[old])
    assert result.subscription_status == "subscription_ended"
    assert result.subscription_started_at == start


def first_invoice(total: int, discounted: int = 0, reason="subscription_create"):
    """A subscription's ``latest_invoice``, as ``data.latest_invoice``
    expands it."""
    return {
        "billing_reason": reason,
        "total": total,
        "total_discount_amounts": (
            [{"amount": discounted, "discount": "di_1"}] if discounted else []
        ),
    }


@pytest.mark.parametrize(
    "invoice, expected",
    [
        # A 100% coupon for the first period: Stripe removed it from the
        # subscription after the first invoice, which it brought to zero.
        (first_invoice(0, discounted=5000), "signed"),
        (first_invoice(2500, discounted=2500), "subscribed"),
        # $0 without a discount (a metered price with no usage yet, a later
        # billing anchor): still paid.
        (first_invoice(0), "subscribed"),
        # A renewal that a coupon made free is read from the coupon instead.
        (first_invoice(0, discounted=5000, reason="subscription_cycle"), "subscribed"),
        # Unexpanded: only the invoice ID.
        ("in_1", "subscribed"),
    ],
)
def test_a_first_invoice_a_discount_took_all_of_is_not_paid(invoice, expected):
    result = snapshot(subs=[sub(latest_invoice=invoice)])
    assert result.subscription_status == expected


def test_a_first_free_period_canceled_before_any_charge_never_paid():
    gone = sub(
        status="canceled",
        canceled_at=NOW - timedelta(days=3),
        ended_at=NOW - timedelta(days=1),
        latest_invoice=first_invoice(0, discounted=5000),
    )
    assert snapshot(subs=[gone]) == LifecycleSnapshot(
        subscription_status="signed", signup_at=SIGNUP
    )


def test_an_unexpanded_discount_is_not_read_as_a_coupon():
    # A list call without the expansion returns discount IDs only.
    result = snapshot(subs=[sub(discounts=["di_1"])])
    assert result.subscription_status == "subscribed"


def test_a_fully_discounted_subscription_neither_cancels_nor_ends():
    free = [coupon(percent_off=100)]
    canceling = sub(
        cancel_at_period_end=True, canceled_at=NOW - timedelta(days=1), discounts=free
    )
    gone = sub(
        status="canceled",
        canceled_at=NOW - timedelta(days=3),
        ended_at=NOW - timedelta(days=1),
        discounts=free,
    )
    assert snapshot(subs=[canceling]).subscription_status == "signed"
    assert snapshot(subs=[gone]).subscription_status == "signed"


@pytest.mark.parametrize("old_status", ["canceled", "past_due"])
def test_a_former_payer_on_a_free_subscription_is_signed_not_churned(old_status):
    """Comped after paying: not paying now, so not subscribed, but not
    churned or failing either. The first paid start stays."""
    start = NOW - timedelta(days=100)
    old = sub(
        "sub_old",
        old_status,
        start=start,
        canceled_at=NOW - timedelta(days=30) if old_status == "canceled" else None,
        ended_at=NOW - timedelta(days=30) if old_status == "canceled" else None,
    )
    comped = sub(
        "sub_comped",
        start=NOW - timedelta(days=20),
        discounts=[coupon(percent_off=100)],
    )
    assert snapshot(subs=[old, comped]) == LifecycleSnapshot(
        subscription_status="signed",
        signup_at=SIGNUP,
        subscription_started_at=start,
    )


def test_a_subscription_that_paid_before_its_coupon_keeps_its_paid_start():
    """One subscription, paid for a month, then given a 100% coupon: they're
    comped now, not a never-paid signup, and they first paid at its start."""
    start = NOW - timedelta(days=60)
    comp = coupon(percent_off=100, start=int((NOW - timedelta(days=30)).timestamp()))
    result = snapshot(subs=[sub(start=start, discounts=[comp])])
    assert result == LifecycleSnapshot(
        subscription_status="signed", signup_at=SIGNUP, subscription_started_at=start
    )


def test_a_converted_trial_given_a_coupon_later_keeps_its_conversion():
    converted = NOW - timedelta(days=40)
    t = trial(status="active", ends_at=converted, converted_at=converted)
    comp = coupon(percent_off=100, start=int((NOW - timedelta(days=10)).timestamp()))
    later = sub("sub_trial", start=TRIAL_START, enrollment="trial-1", discounts=[comp])
    result = snapshot(t=t, subs=[later])
    assert result.subscription_status == "signed"
    assert result.subscription_started_at == converted


def test_a_subscription_ending_with_its_repeating_coupon_was_never_paid():
    # Three free months, set to cancel: Stripe ends it the moment the coupon
    # ends.
    start = NOW - timedelta(days=90)
    ended = NOW - timedelta(days=1)
    free = coupon(
        percent_off=100,
        start=int(start.timestamp()),
        end=int(ended.timestamp()),
    )
    gone = sub(
        status="canceled",
        start=start,
        canceled_at=NOW - timedelta(days=20),
        ended_at=ended,
        discounts=[free],
    )
    assert snapshot(subs=[gone]) == LifecycleSnapshot(
        subscription_status="signed", signup_at=SIGNUP
    )


def test_a_free_subscription_does_not_supply_the_first_paid_start():
    free = sub(
        "sub_free",
        "canceled",
        start=NOW - timedelta(days=100),
        ended_at=NOW - timedelta(days=50),
        discounts=[coupon(percent_off=100)],
    )
    paid_start = NOW - timedelta(days=10)
    result = snapshot(subs=[free, sub("sub_paid", start=paid_start)])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == paid_start


def test_a_trial_converted_onto_a_free_subscription_is_not_subscribed():
    t = trial(
        status="active",
        ends_at=NOW - timedelta(days=20),
        converted_at=NOW - timedelta(days=20),
    )
    free = sub(
        "sub_trial",
        "active",
        start=TRIAL_START,
        enrollment="trial-1",
        discounts=[coupon(percent_off=100)],
    )
    result = snapshot(t=t, subs=[free])
    assert result.subscription_status == "trial_expired"
    assert result.subscription_started_at is None


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
