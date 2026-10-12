"""The lifecycle status of a trial canceled at period end: trial canceled
while it runs out, in trial again once resumed, and subscribed as soon as
another plan is bought or the trial converts early."""

from datetime import timedelta

import pytest
from prisma.enums import SubscriptionTier

from backend.data.posthog_lifecycle import LifecycleSnapshot, StripeSubscriptionFacts
from backend.data.posthog_lifecycle_test import (
    NOW,
    SIGNUP,
    TRIAL_START,
    snapshot,
    sub,
    trial,
    user,
)

TRIAL_END = NOW + timedelta(days=5)


def _trial_sub(status: str = "trialing", **extra) -> StripeSubscriptionFacts:
    extra.setdefault("trial_end", TRIAL_END)
    return sub("sub_trial", status, start=TRIAL_START, enrollment="trial-1", **extra)


def test_cancel_pending_trial_is_trial_canceled_without_subscription_dates():
    pending = _trial_sub(cancel_at_period_end=True, canceled_at=NOW)
    result = snapshot(
        t=trial(cancel_at_period_end=True, ends_at=TRIAL_END), subs=[pending]
    )
    assert result == LifecycleSnapshot(
        subscription_status="trial_canceled",
        signup_at=SIGNUP,
        trial_started_at=TRIAL_START,
    )


def test_cancel_then_resume_goes_back_to_in_trial():
    steps = [(True, "trial_canceled"), (False, "in_trial"), (True, "trial_canceled")]
    for pending, expected in steps:
        result = snapshot(
            t=trial(cancel_at_period_end=pending, ends_at=TRIAL_END),
            subs=[_trial_sub(cancel_at_period_end=pending)],
        )
        assert result.subscription_status == expected


@pytest.mark.parametrize("deleted", [False, True], ids=["awaiting-delete", "deleted"])
def test_cancel_pending_trial_that_reached_its_end_is_trial_canceled(deleted: bool):
    """Stripe ends it at trial_end with no invoice; it never became a paid
    subscription, so it is not subscription_ended."""
    end = NOW - timedelta(minutes=5)
    status = "canceled" if deleted else "trialing"
    ended = _trial_sub(
        status,
        trial_end=end,
        cancel_at_period_end=True,
        canceled_at=NOW - timedelta(days=2),
        ended_at=end if deleted else None,
    )
    t = trial(status=status, cancel_at_period_end=True, ends_at=end)
    assert snapshot(t=t, subs=[ended]) == LifecycleSnapshot(
        subscription_status="trial_canceled",
        signup_at=SIGNUP,
        trial_started_at=TRIAL_START,
    )


def test_buying_another_plan_while_cancel_pending_is_subscribed():
    """The new plan is live before the trial subscription is canceled, and
    stays the answer after."""
    max_sub = sub("sub_max", start=NOW)
    before = (
        trial(cancel_at_period_end=True, ends_at=TRIAL_END),
        _trial_sub(cancel_at_period_end=True),
    )
    after = (
        trial(status="canceled", cancel_at_period_end=True, ends_at=NOW),
        _trial_sub("canceled", canceled_at=NOW, ended_at=NOW),
    )
    for t, trial_sub in (before, after):
        result = snapshot(u=user(SubscriptionTier.MAX), t=t, subs=[trial_sub, max_sub])
        assert result == LifecycleSnapshot(
            subscription_status="subscribed",
            signup_at=SIGNUP,
            trial_started_at=TRIAL_START,
            subscription_started_at=max_sub.started_at,
        )


def test_converting_early_while_cancel_pending_is_subscribed_from_conversion():
    """Subscribe now ends the trial early and clears the cancellation in the
    same update, so it reads as a trial awaiting its charge, then subscribed."""
    early_end = NOW - timedelta(seconds=1)
    live = _trial_sub("active", trial_end=early_end)
    charging = trial(status="active", ends_at=early_end)
    assert snapshot(t=charging, subs=[live]).subscription_status == "in_trial"
    converted = trial(status="active", ends_at=early_end, converted_at=NOW)
    result = snapshot(u=user(SubscriptionTier.PRO), t=converted, subs=[live])
    assert result.subscription_status == "subscribed"
    assert result.subscription_started_at == NOW
