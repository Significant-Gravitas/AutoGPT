"""A trial Stripe has just ended converts even when Stripe's clock runs a
little ahead of ours, as long as its post-trial invoice is paid."""

from datetime import UTC, datetime

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_checkout_test as checkout_test
from backend.data.subscription_trial_stripe import (
    STRIPE_CLOCK_SKEW_SECONDS,
    SubscriptionSnapshot,
    trial_subscription_tier,
)

trial = checkout_test.trial


@pytest.mark.parametrize(
    "end_delta,invoice_status,reason,created_delta,expected",
    [
        (60, "paid", "subscription_update", 0, SubscriptionTier.PRO),
        (
            STRIPE_CLOCK_SKEW_SECONDS,
            "paid",
            "subscription_update",
            0,
            SubscriptionTier.PRO,
        ),
        (
            STRIPE_CLOCK_SKEW_SECONDS + 1,
            "paid",
            "subscription_update",
            0,
            SubscriptionTier.NO_TIER,
        ),
        (60, "open", "subscription_update", 0, SubscriptionTier.NO_TIER),
        (60, "paid", "subscription_create", 0, SubscriptionTier.NO_TIER),
        (60, "paid", "subscription_update", -1, SubscriptionTier.NO_TIER),
    ],
    ids=[
        "ended-a-minute-ahead",
        "ended-at-the-skew-limit",
        "beyond-the-skew-limit",
        "unpaid",
        "first-invoice",
        "invoice-before-end",
    ],
)
def test_trial_stripe_just_ended_converts_despite_clock_skew(
    trial, end_delta, invoice_status, reason, created_delta, expected
):
    """Ending a trial "now" stamps Stripe's clock, which can run ahead of ours:
    status active already means Stripe ended the trial, and the paid post-trial
    invoice stays the real guard."""
    now = datetime(2026, 9, 10, tzinfo=UTC)
    end = int(now.timestamp()) + end_delta
    subscription = SubscriptionSnapshot.model_validate(
        {
            "id": "sub_1",
            "customer": "cus_1",
            "status": "active",
            "trial_end": end,
            "latest_invoice": {
                "id": "in_1",
                "status": invoice_status,
                "billing_reason": reason,
                "created": end + created_delta,
            },
        }
    )
    assert trial_subscription_tier(trial, subscription, now) == expected
