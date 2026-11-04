"""Seats: a team organization's active members plus its pending invitations
may not exceed its plan's seats (``backend.data.org_plan``).

Without a plan that counts, the owner holds the only seat, so nobody can be
invited. Checked when an invitation is created and when one is accepted,
under the organization's membership lock, so two at once can't both take the
last seat. With ``ENABLE_PLATFORM_PAYMENT`` off for the installation (a
self-hosted one), there are no seats to count.
"""

from datetime import datetime, timezone

from fastapi import HTTPException
from prisma import Prisma

from backend.data.org_plan import OrgPlanSnapshot, read_org_plan_snapshot
from backend.util.platform_payment import is_platform_payment_enabled

from .db import lock_org_membership

NO_PLAN_REFUSAL = (
    "This organization has no active plan, so there are no seats to invite "
    "people to. Ask AutoGPT to set up a team plan."
)


async def assert_free_seat(
    client: Prisma, org_id: str, actor_user_id: str, *, accepting: bool
) -> None:
    """Refuse (409) when taking one more seat would pass the plan's seats.

    An invitation takes a seat when it is created and keeps it until it is
    accepted, declined, revoked or expires. Accepting one moves its seat to the
    new member, so the count must already fit with it included.
    """
    if not is_platform_payment_enabled():
        return
    await lock_org_membership(client, org_id, actor_user_id)
    snapshot = await read_org_plan_snapshot(client, org_id)
    in_use = await seats_in_use(client, org_id)
    wanted = in_use if accepting else in_use + 1
    if wanted <= snapshot.seat_limit():
        return
    raise HTTPException(status_code=409, detail=_refusal(snapshot))


async def seats_in_use(client: Prisma, org_id: str) -> int:
    """Active members plus pending invitations: unexpired, and not accepted,
    declined or revoked."""
    members = await client.orgmember.count(where={"orgId": org_id, "status": "ACTIVE"})
    pending = await client.orginvitation.count(
        where={
            "orgId": org_id,
            "acceptedAt": None,
            "revokedAt": None,
            "expiresAt": {"gt": datetime.now(timezone.utc)},
        }
    )
    return members + pending


def _refusal(snapshot: OrgPlanSnapshot) -> str:
    plan = snapshot.counting_plan()
    if plan is None:
        return NO_PLAN_REFUSAL
    return (
        f"All {plan.seats} seats on this organization's plan are taken by "
        "members and pending invitations. Ask AutoGPT to add seats, or revoke "
        "a pending invitation."
    )
