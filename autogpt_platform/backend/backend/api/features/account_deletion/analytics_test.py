"""The plan and account-age buckets the account deletion events carry."""

from datetime import datetime, timedelta, timezone

import pytest
from prisma.enums import SubscriptionTier

from backend.api.features.account_deletion.analytics import plan_state, tenure_bucket

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
SECOND = timedelta(seconds=1)


@pytest.mark.parametrize(
    "tier, expected",
    [
        (SubscriptionTier.PRO, "pro"),
        (SubscriptionTier.MAX, "max"),
        (SubscriptionTier.TRIAL, "trial"),
        (SubscriptionTier.NO_TIER, "none"),
        (SubscriptionTier.BASIC, "managed"),
        (SubscriptionTier.BUSINESS, "managed"),
        (SubscriptionTier.ENTERPRISE, "managed"),
    ],
)
def test_plan_state(tier: SubscriptionTier, expected: str):
    assert plan_state(tier) == expected


def test_every_tier_has_a_plan_state():
    """A new tier must be mapped, or deleting an account on it would fail."""
    assert {plan_state(tier) for tier in SubscriptionTier} <= {
        "pro",
        "max",
        "trial",
        "none",
        "managed",
    }


@pytest.mark.parametrize(
    "age, expected",
    [
        (timedelta(0), "<7d"),
        (timedelta(days=7) - SECOND, "<7d"),
        (timedelta(days=7), "7-30d"),
        (timedelta(days=30) - SECOND, "7-30d"),
        (timedelta(days=30), "30-90d"),
        (timedelta(days=90) - SECOND, "30-90d"),
        (timedelta(days=90), "90d+"),
        (timedelta(days=3650), "90d+"),
    ],
)
def test_tenure_bucket(age: timedelta, expected: str):
    assert tenure_bucket(NOW - age, now=NOW) == expected


def test_tenure_is_measured_up_to_now_by_default():
    created_at = datetime.now(timezone.utc) - timedelta(days=8)

    assert tenure_bucket(created_at) == "7-30d"
