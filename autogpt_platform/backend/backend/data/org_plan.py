"""A team organization's plan, as the entitlement check reads it.

Sales invoices a customer outside the product, then an AutoGPT admin sets the
organization's plan (``org_plan_admin.py``). The plan is the organization's
``OrganizationSubscription`` row: ``planTier``, ``status``, and the typed
``seats``, ``endsAt`` and ``note`` columns. It counts only while ``status`` is
``active`` and ``endsAt`` has not passed. ``seats`` caps the organization's
active members plus its pending invitations (``org_seats.py``).

Every process reads the same snapshot: the shared cache first, then the
database or the DatabaseManager. An admin's write drops the snapshot at once;
otherwise a change reaches every process within ``PLAN_CACHE_SECONDS``.
"""

import logging
from datetime import datetime, timezone
from typing import Literal

from prisma import Prisma
from prisma.models import Organization, OrganizationSubscription, OrgMember
from pydantic import BaseModel

from backend.copilot.rate_limit import SubscriptionTier
from backend.data.db import prisma
from backend.data.db_accessors import org_plan_db
from backend.util.cache import cached

logger = logging.getLogger(__name__)

PLAN_CACHE_SECONDS = 60
ACTIVE_STATUS = "active"
PlanStatus = Literal["active", "canceled"]
# The tiers a plan can give; NO_TIER and TRIAL are personal states only.
PLAN_TIERS = (
    SubscriptionTier.BASIC,
    SubscriptionTier.PRO,
    SubscriptionTier.MAX,
    SubscriptionTier.BUSINESS,
    SubscriptionTier.ENTERPRISE,
)


class OrgPlan(BaseModel):
    tier: SubscriptionTier
    seats: int
    status: str
    ends_at: datetime | None
    note: str | None
    updated_at: datetime

    def counts(self, now: datetime | None = None) -> bool:
        """Whether the plan entitles the organization's members right now."""
        if self.status != ACTIVE_STATUS or self.tier not in PLAN_TIERS:
            return False
        return self.ends_at is None or self.ends_at > (
            now or datetime.now(timezone.utc)
        )


class OrgPlanSnapshot(BaseModel):
    """What the entitlement check needs to know about one organization."""

    org_id: str
    exists: bool
    is_personal: bool = False
    # Neither archived nor deleted.
    is_live: bool = False
    plan: OrgPlan | None = None

    def counting_plan(self, now: datetime | None = None) -> OrgPlan | None:
        return self.plan if self.plan is not None and self.plan.counts(now) else None

    def seat_limit(self, now: datetime | None = None) -> int:
        """Members plus pending invitations allowed: the plan's seats, or the
        owner alone while no plan counts."""
        plan = self.counting_plan(now)
        return max(1, plan.seats) if plan is not None else 1


@cached(maxsize=4096, ttl_seconds=PLAN_CACHE_SECONDS, shared_cache=True)
async def get_org_plan_snapshot(org_id: str) -> OrgPlanSnapshot:
    """The organization's kind and plan, shared across processes."""
    return await org_plan_db().query_org_plan_snapshot(org_id)


def forget_org_plan(org_id: str) -> None:
    """Drop the shared snapshot after the plan or the organization changed."""
    try:
        get_org_plan_snapshot.cache_delete(org_id)
    except Exception:
        logger.warning(f"Could not drop the cached plan of org {org_id}", exc_info=True)


async def query_org_plan_snapshot(org_id: str) -> OrgPlanSnapshot:
    """The uncached read behind :func:`get_org_plan_snapshot`."""
    return await read_org_plan_snapshot(prisma, org_id)


async def read_org_plan_snapshot(client: Prisma, org_id: str) -> OrgPlanSnapshot:
    """The organization's plan read through *client*, e.g. a transaction's."""
    org = await Organization.prisma(client).find_unique(
        where={"id": org_id}, include={"Subscription": True}
    )
    if org is None:
        return OrgPlanSnapshot(org_id=org_id, exists=False)
    return OrgPlanSnapshot(
        org_id=org_id,
        exists=True,
        is_personal=org.isPersonal,
        is_live=org.archivedAt is None and org.deletedAt is None,
        plan=plan_from_row(org.Subscription),
    )


async def query_active_membership(org_id: str, user_id: str) -> bool:
    """Whether *user_id* is an active member of *org_id*, read live."""
    member = await OrgMember.prisma().find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )
    return member is not None and member.status == "ACTIVE"


def plan_from_row(row: OrganizationSubscription | None) -> OrgPlan | None:
    """The plan a subscription row describes, or None when it names no tier."""
    if row is None or not row.planTier:
        return None
    try:
        tier = SubscriptionTier(row.planTier)
    except ValueError:
        logger.warning(
            f"Org {row.organizationId} has an unknown plan tier {row.planTier!r}"
        )
        return None
    return OrgPlan(
        tier=tier,
        seats=row.seats,
        status=row.status,
        ends_at=row.endsAt,
        note=row.note,
        updated_at=row.updatedAt,
    )
