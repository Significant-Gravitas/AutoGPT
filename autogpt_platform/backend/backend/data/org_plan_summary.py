"""What the admin page shows about a team organization's plan: the plan, the
seats in use and the wallet balance."""

from datetime import datetime, timezone

from prisma.models import Organization, OrgBalance, OrgInvitation, OrgMember
from pydantic import BaseModel

from backend.data.org_plan import OrgPlan, plan_from_row
from backend.util.exceptions import NotFoundError


class OrgPlanView(OrgPlan):
    # Active and not past its end date: what the entitlement check reads.
    in_effect: bool


class OrgPlanSummary(BaseModel):
    org_id: str
    name: str
    slug: str
    owner_user_id: str | None
    owner_email: str | None
    plan: OrgPlanView | None
    members: int
    pending_invitations: int
    # Members plus pending invitations, which the plan's seats cap.
    seats_in_use: int
    # The organization's own wallet, in credits.
    balance: int


async def org_plan_summary(org_id: str) -> OrgPlanSummary:
    org = await Organization.prisma().find_unique(
        where={"id": org_id}, include={"Subscription": True}
    )
    if org is None or org.deletedAt is not None:
        raise NotFoundError(f"Organization #{org_id} not found")
    owner = await OrgMember.prisma().find_first(
        where={"orgId": org_id, "isOwner": True, "status": "ACTIVE"},
        include={"User": True},
    )
    members = await OrgMember.prisma().count(
        where={"orgId": org_id, "status": "ACTIVE"}
    )
    pending = await OrgInvitation.prisma().count(
        where={
            "orgId": org_id,
            "acceptedAt": None,
            "revokedAt": None,
            "expiresAt": {"gt": datetime.now(timezone.utc)},
        }
    )
    wallet = await OrgBalance.prisma().find_unique(where={"orgId": org_id})
    plan = plan_from_row(org.Subscription)
    return OrgPlanSummary(
        org_id=org.id,
        name=org.name,
        slug=org.slug,
        owner_user_id=owner.userId if owner else None,
        owner_email=owner.User.email if owner and owner.User else None,
        plan=(
            OrgPlanView(**plan.model_dump(), in_effect=plan.counts()) if plan else None
        ),
        members=members,
        pending_invitations=pending,
        seats_in_use=members + pending,
        balance=wallet.balance if wallet else 0,
    )
