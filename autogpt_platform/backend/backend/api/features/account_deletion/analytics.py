"""The account facts the account deletion events carry, computed on the server.

The events are anonymous, so they carry buckets rather than the tier or the
sign-up date, and the client never supplies them: the deletion service and the
feedback emitter both read them from the User row.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

from prisma.enums import SubscriptionTier

PlanState = Literal["pro", "max", "trial", "none", "managed"]
TenureBucket = Literal["<7d", "7-30d", "30-90d", "90d+"]

_PLAN_STATES: dict[SubscriptionTier, PlanState] = {
    SubscriptionTier.PRO: "pro",
    SubscriptionTier.MAX: "max",
    SubscriptionTier.TRIAL: "trial",
    SubscriptionTier.NO_TIER: "none",
    # Set by the team, not bought in the app.
    SubscriptionTier.BASIC: "managed",
    SubscriptionTier.BUSINESS: "managed",
    SubscriptionTier.ENTERPRISE: "managed",
}

# Each bucket's upper bound, youngest first; anything older is "90d+".
_TENURE_BUCKETS: tuple[tuple[timedelta, TenureBucket], ...] = (
    (timedelta(days=7), "<7d"),
    (timedelta(days=30), "7-30d"),
    (timedelta(days=90), "30-90d"),
)


def plan_state(tier: SubscriptionTier) -> PlanState:
    return _PLAN_STATES[tier]


def tenure_bucket(created_at: datetime, *, now: datetime | None = None) -> TenureBucket:
    """How old the account is, as one of four buckets."""
    age = (now or datetime.now(timezone.utc)) - created_at
    for upper_bound, bucket in _TENURE_BUCKETS:
        if age < upper_bound:
            return bucket
    return "90d+"
