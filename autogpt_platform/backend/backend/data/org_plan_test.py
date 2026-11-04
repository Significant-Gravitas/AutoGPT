"""When a team organization's plan counts, and how many seats it gives."""

from datetime import datetime, timedelta, timezone

import pytest
from prisma.models import OrganizationSubscription

from backend.copilot.rate_limit import SubscriptionTier
from backend.data.org_plan import OrgPlan, OrgPlanSnapshot, plan_from_row

NOW = datetime.now(timezone.utc)


def _plan(**changes) -> OrgPlan:
    fields = {
        "tier": SubscriptionTier.BUSINESS,
        "seats": 5,
        "status": "active",
        "ends_at": None,
        "note": None,
        "updated_at": NOW,
    }
    return OrgPlan(**(fields | changes))


@pytest.mark.parametrize(
    "plan, counts",
    [
        (_plan(), True),
        (_plan(ends_at=NOW + timedelta(days=1)), True),
        (_plan(ends_at=NOW - timedelta(seconds=1)), False),
        (_plan(status="canceled"), False),
        (_plan(tier=SubscriptionTier.TRIAL), False),
        (_plan(tier=SubscriptionTier.NO_TIER), False),
    ],
    ids=["open-ended", "ends-later", "ended", "canceled", "trial", "no-tier"],
)
def test_a_plan_counts_while_active_and_before_its_end(plan: OrgPlan, counts: bool):
    assert plan.counts(NOW) is counts


@pytest.mark.parametrize(
    "plan, seats",
    [(_plan(seats=5), 5), (_plan(seats=5, status="canceled"), 1), (None, 1)],
    ids=["plan", "plan-that-doesnt-count", "no-plan"],
)
def test_without_a_plan_that_counts_the_owner_holds_the_only_seat(plan, seats):
    snapshot = OrgPlanSnapshot(org_id="o", exists=True, is_live=True, plan=plan)
    assert snapshot.seat_limit(NOW) == seats


def test_a_row_without_a_known_tier_is_no_plan():
    row = OrganizationSubscription.model_construct(
        organizationId="o",
        planTier="PLATINUM",
        seats=3,
        status="active",
        endsAt=None,
        note=None,
        updatedAt=NOW,
    )
    assert plan_from_row(row) is None
