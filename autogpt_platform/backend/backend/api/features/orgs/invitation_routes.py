"""Invitation API routes for organization membership."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import uuid4

from autogpt_libs.auth import get_user_id, requires_org_permission, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, HTTPException, Query, Security
from prisma import Prisma
from prisma.errors import UniqueViolationError
from prisma.models import OrgInvitation
from prisma.types import OrgInvitationWhereInput

from backend.api.live_auth import requires_live_org_permission
from backend.data.db import execute_raw_with_schema, prisma, transaction
from backend.data.tenancy import lock_live_org_permission_scope
from backend.util.exceptions import NotFoundError

from .db import lock_org_membership
from .model import (
    CreateInvitationRequest,
    InvitationCreateResponse,
    InvitationResponse,
    UserInvitationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

INVITATION_TTL_DAYS = 7


def _verify_org_path(ctx: RequestContext, org_id: str) -> None:
    """Ensure the authenticated user's active org matches the path parameter."""
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")


async def _get_org_invitation(
    org_id: str, invitation_id: str, client: Prisma | None = None
) -> OrgInvitation:
    """Load an invitation, enforcing that it belongs to `org_id`.

    Cross-org lookups are reported as "not found" rather than "forbidden" so
    the endpoint does not leak the existence of other orgs' invitations.
    """
    db = client or prisma
    invitation = await db.orginvitation.find_unique(where={"id": invitation_id})
    if invitation is None or invitation.orgId != org_id:
        raise NotFoundError(f"Invitation {invitation_id} not found")
    return invitation


def _reject_if_not_pending(invitation: OrgInvitation) -> None:
    """Reject invitations that are no longer actionable (expiry is allowed)."""
    if invitation.acceptedAt is not None:
        raise HTTPException(400, detail="Invitation already accepted")
    if invitation.revokedAt is not None:
        raise HTTPException(400, detail="Invitation was revoked")


async def _lock_invitation_admin(
    client: Prisma,
    org_id: str,
    actor_user_id: str,
    target_user_id: str | None = None,
) -> None:
    if (
        await lock_live_org_permission_scope(
            client,
            actor_user_id,
            org_id,
            OrgAction.MANAGE_MEMBERS,
            [target_user_id] if target_user_id is not None else None,
        )
        is None
    ):
        raise HTTPException(403, detail="Organization admin access was revoked")


# --- Org-scoped invitation endpoints (under /api/orgs/{org_id}/invitations) ---

org_router = APIRouter()


@org_router.post(
    "",
    summary="Create invitation",
    tags=["orgs", "invitations"],
)
async def create_invitation(
    org_id: str,
    request: CreateInvitationRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> InvitationCreateResponse:
    _verify_org_path(ctx, org_id)
    return await _create_invitation_locked(org_id, request, ctx.user_id)


async def _create_invitation_locked(
    org_id: str, request: CreateInvitationRequest, user_id: str
) -> InvitationCreateResponse:

    email = request.email.strip().lower()
    preliminary_target = await prisma.user.find_first(
        where={"email": {"equals": email, "mode": "insensitive"}}
    )
    try:
        async with transaction() as tx:
            await _lock_invitation_admin(
                tx,
                org_id,
                user_id,
                preliminary_target.id if preliminary_target is not None else None,
            )
            target_user = await tx.user.find_first(
                where={"email": {"equals": email, "mode": "insensitive"}}
            )
            if (target_user.id if target_user is not None else None) != (
                preliminary_target.id if preliminary_target is not None else None
            ):
                raise HTTPException(409, detail="Invitation target changed; retry")
            if target_user is not None:
                membership = await tx.orgmember.find_unique(
                    where={
                        "orgId_userId": {
                            "orgId": org_id,
                            "userId": target_user.id,
                        }
                    }
                )
                if membership is not None and membership.status == "ACTIVE":
                    raise HTTPException(409, detail="This user is already a member")

            pending = await tx.orginvitation.find_first(
                where={
                    "orgId": org_id,
                    "email": {"equals": email, "mode": "insensitive"},
                    "acceptedAt": None,
                    "revokedAt": None,
                }
            )
            if pending is not None:
                raise HTTPException(409, detail="A pending invitation already exists")

            if request.team_ids:
                teams = await tx.team.find_many(where={"id": {"in": request.team_ids}})
                valid_ids = {team.id for team in teams if team.orgId == org_id}
                invalid = [team for team in request.team_ids if team not in valid_ids]
                if invalid:
                    raise HTTPException(
                        400,
                        detail=f"Teams not found in this organization: {invalid}",
                    )

            invitation = await tx.orginvitation.create(
                data={
                    "orgId": org_id,
                    "email": email,
                    "targetUserId": (
                        target_user.id if target_user is not None else None
                    ),
                    "isAdmin": request.is_admin,
                    "isBillingManager": request.is_billing_manager,
                    "expiresAt": datetime.now(timezone.utc)
                    + timedelta(days=INVITATION_TTL_DAYS),
                    "invitedByUserId": user_id,
                    "teamIds": request.team_ids,
                }
            )
    except UniqueViolationError as error:
        raise HTTPException(
            409, detail="A pending invitation already exists"
        ) from error

    # TODO: Send email via Postmark with invitation link
    # link = f"{frontend_base_url}/org/invite/{invitation.token}"

    return InvitationCreateResponse.from_db(invitation)


@org_router.get(
    "",
    summary="List pending invitations",
    tags=["orgs", "invitations"],
)
async def list_invitations(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        requires_live_org_permission(OrgAction.MANAGE_MEMBERS),
    ],
    include_expired: Annotated[
        bool,
        Query(
            description=(
                "Also return invitations whose expiresAt has passed. These are "
                "still resendable, so admins need them to recover a lapsed invite."
            )
        ),
    ] = False,
) -> list[InvitationResponse]:
    _verify_org_path(ctx, org_id)
    where: OrgInvitationWhereInput = {
        "orgId": org_id,
        "acceptedAt": None,
        "revokedAt": None,
    }
    if not include_expired:
        where["expiresAt"] = {"gt": datetime.now(timezone.utc)}
    invitations = await prisma.orginvitation.find_many(
        where=where,
        order={"createdAt": "desc"},
    )
    return [InvitationResponse.from_db(inv) for inv in invitations]


@org_router.delete(
    "/{invitation_id}",
    summary="Revoke invitation",
    tags=["orgs", "invitations"],
    status_code=204,
)
async def revoke_invitation(
    org_id: str,
    invitation_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> None:
    _verify_org_path(ctx, org_id)
    preliminary = await _get_org_invitation(org_id, invitation_id)
    async with transaction() as tx:
        await _lock_invitation_admin(tx, org_id, ctx.user_id, preliminary.targetUserId)
        await _get_org_invitation(org_id, invitation_id, tx)
        changed = await tx.orginvitation.update_many(
            where={
                "id": invitation_id,
                "acceptedAt": None,
                "revokedAt": None,
            },
            data={"revokedAt": datetime.now(timezone.utc)},
        )
        if changed != 1:
            raise HTTPException(409, detail="Invitation is no longer available")


@org_router.post(
    "/{invitation_id}/resend",
    summary="Resend invitation",
    tags=["orgs", "invitations"],
    responses={
        400: {"description": "Invitation was already accepted or revoked"},
        403: {"description": "Not a member of this organization"},
        404: {"description": "Invitation not found in this organization"},
    },
)
async def resend_invitation(
    org_id: str,
    invitation_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> InvitationCreateResponse:
    """Refresh a pending (possibly expired) invitation: new token, new TTL.

    Rotating the token invalidates any previously sent link, so a resend
    also acts as a soft revoke of the old email.
    """
    _verify_org_path(ctx, org_id)
    preliminary = await _get_org_invitation(org_id, invitation_id)
    async with transaction() as tx:
        await _lock_invitation_admin(tx, org_id, ctx.user_id, preliminary.targetUserId)
        invitation = await _get_org_invitation(org_id, invitation_id, tx)
        _reject_if_not_pending(invitation)

        updated_count = await tx.orginvitation.update_many(
            where={
                "id": invitation_id,
                "acceptedAt": None,
                "revokedAt": None,
            },
            data={
                "token": str(uuid4()),
                "teamIds": await _surviving_team_ids(invitation, tx),
                "expiresAt": datetime.now(timezone.utc)
                + timedelta(days=INVITATION_TTL_DAYS),
            },
        )
        if updated_count == 0:
            current = await _get_org_invitation(org_id, invitation_id, tx)
            _reject_if_not_pending(current)
            raise HTTPException(400, detail="Invitation changed concurrently; retry")

        refreshed = await tx.orginvitation.find_unique(where={"id": invitation_id})
        if refreshed is None:
            raise NotFoundError(f"Invitation {invitation_id} not found")

    # TODO: Send email via Postmark with invitation link (same gap as create).
    # Rate-limit resends (min-interval / per-invite cap) as part of that work —
    # unbounded resend is only an email-bombing vector once delivery is wired up.
    return InvitationCreateResponse.from_db(refreshed)


async def _surviving_team_ids(
    invitation: OrgInvitation, client: Prisma | None = None
) -> list[str]:
    """Drop team IDs that no longer exist in the org, logging what was dropped.

    `create_invitation` validates team IDs up front, but a team can be deleted
    between create and resend. Accept only logs-and-skips such teams, so a
    resent invite would silently promise access it can no longer grant.
    Pruning here keeps the stored invitation honest about what it confers.
    """
    if not invitation.teamIds:
        return []

    db = client or prisma
    teams = await db.team.find_many(
        where={"id": {"in": invitation.teamIds}, "archivedAt": None}
    )
    valid_ids = {t.id for t in teams if t.orgId == invitation.orgId}
    surviving = [tid for tid in invitation.teamIds if tid in valid_ids]
    dropped = [tid for tid in invitation.teamIds if tid not in valid_ids]
    if dropped:
        logger.warning(
            f"Invitation resend: dropping teams {dropped} from invitation "
            f"{invitation.id} in org {invitation.orgId} (deleted since invite)"
        )
    return surviving


# --- Token-based endpoints (under /api/invitations) ---


@router.post(
    "/{token}/accept",
    summary="Accept invitation",
    tags=["invitations"],
    dependencies=[Security(requires_user)],
)
async def accept_invitation(
    token: str,
    user_id: Annotated[str, Security(get_user_id)],
) -> dict:
    invitation = await prisma.orginvitation.find_unique(where={"token": token})
    if invitation is None:
        raise NotFoundError("Invitation not found")
    if invitation.acceptedAt is not None:
        raise HTTPException(400, detail="Invitation already accepted")
    if invitation.revokedAt is not None:
        raise HTTPException(400, detail="Invitation has been revoked")
    if invitation.expiresAt < datetime.now(timezone.utc):
        raise HTTPException(400, detail="Invitation has expired")

    # Verify the accepting user's email matches the invitation
    accepting_user = await prisma.user.find_unique(where={"id": user_id})
    if accepting_user is None:
        raise HTTPException(401, detail="User not found")
    if accepting_user.email.lower() != invitation.email.lower():
        raise HTTPException(
            403,
            detail="This invitation was sent to a different email address",
        )

    accepted_at = datetime.now(timezone.utc)
    already_member = False
    async with transaction() as tx:
        await lock_org_membership(tx, invitation.orgId, user_id)
        org = await tx.organization.find_first(
            where={"id": invitation.orgId, "deletedAt": None}
        )
        if org is None:
            raise NotFoundError("Invitation not found")

        active_member = await tx.orgmember.find_unique(
            where={
                "orgId_userId": {
                    "orgId": invitation.orgId,
                    "userId": user_id,
                }
            }
        )
        if active_member is not None and active_member.status == "ACTIVE":
            revoked = await tx.orginvitation.update_many(
                where={
                    "id": invitation.id,
                    "token": token,
                    "acceptedAt": None,
                    "revokedAt": None,
                },
                data={"revokedAt": accepted_at, "targetUserId": user_id},
            )
            if revoked != 1:
                raise HTTPException(409, detail="Invitation is no longer available")
            already_member = True

        if already_member:
            teams = []
        else:
            teams = await tx.team.find_many(
                where={
                    "orgId": invitation.orgId,
                    "archivedAt": None,
                    "OR": [
                        {"isDefault": True},
                        *(
                            [{"id": {"in": invitation.teamIds}}]
                            if invitation.teamIds
                            else []
                        ),
                    ],
                }
            )
        active_team_ids = {team.id for team in teams}
        for team_id in sorted(active_team_ids):
            await execute_raw_with_schema(
                'UPDATE {schema_prefix}"Team" SET "updatedAt" = "updatedAt" '
                'WHERE "id" = $1 AND "orgId" = $2',
                team_id,
                invitation.orgId,
                client=tx,
            )
        continue_accept = not already_member
        if continue_accept and any(
            team_id not in active_team_ids for team_id in invitation.teamIds
        ):
            raise HTTPException(409, detail="An invited workspace is no longer active")

        claimed = (
            await tx.orginvitation.update_many(
                where={
                    "id": invitation.id,
                    "token": token,
                    "acceptedAt": None,
                    "revokedAt": None,
                    "expiresAt": {"gt": accepted_at},
                },
                data={"acceptedAt": accepted_at, "targetUserId": user_id},
            )
            if continue_accept
            else 1
        )
        if claimed != 1:
            raise HTTPException(409, detail="Invitation is no longer available")

        if continue_accept:
            await tx.orgmember.upsert(
                where={
                    "orgId_userId": {
                        "orgId": invitation.orgId,
                        "userId": user_id,
                    }
                },
                data={
                    "create": {
                        "orgId": invitation.orgId,
                        "userId": user_id,
                        "isAdmin": invitation.isAdmin,
                        "isBillingManager": invitation.isBillingManager,
                        "status": "ACTIVE",
                        "invitedByUserId": invitation.invitedByUserId,
                    },
                    "update": {
                        "isAdmin": invitation.isAdmin,
                        "isBillingManager": invitation.isBillingManager,
                        "status": "ACTIVE",
                        "invitedByUserId": invitation.invitedByUserId,
                    },
                },
            )
        for team_id in active_team_ids if continue_accept else []:
            await tx.teammember.upsert(
                where={"teamId_userId": {"teamId": team_id, "userId": user_id}},
                data={
                    "create": {
                        "teamId": team_id,
                        "userId": user_id,
                        "isAdmin": invitation.isAdmin,
                        "isBillingManager": invitation.isBillingManager,
                        "status": "ACTIVE",
                        "invitedByUserId": invitation.invitedByUserId,
                    },
                    "update": {
                        "isAdmin": invitation.isAdmin,
                        "isBillingManager": invitation.isBillingManager,
                        "status": "ACTIVE",
                        "invitedByUserId": invitation.invitedByUserId,
                    },
                },
            )

    if already_member:
        raise HTTPException(409, detail="This user is already a member")
    return {"orgId": invitation.orgId, "message": "Invitation accepted"}


@router.post(
    "/{token}/decline",
    summary="Decline invitation",
    tags=["invitations"],
    dependencies=[Security(requires_user)],
    status_code=204,
)
async def decline_invitation(
    token: str,
    user_id: Annotated[str, Security(get_user_id)],
) -> None:
    invitation = await prisma.orginvitation.find_unique(where={"token": token})
    if invitation is None:
        raise NotFoundError("Invitation not found")

    # State checks — same as accept_invitation
    if invitation.acceptedAt is not None:
        raise HTTPException(400, detail="Invitation already accepted")
    if invitation.revokedAt is not None:
        raise HTTPException(400, detail="Invitation already revoked")
    if invitation.expiresAt < datetime.now(timezone.utc):
        raise HTTPException(400, detail="Invitation has expired")

    # Verify the declining user's email matches the invitation
    declining_user = await prisma.user.find_unique(where={"id": user_id})
    if declining_user is None:
        raise HTTPException(401, detail="User not found")
    if declining_user.email.lower() != invitation.email.lower():
        raise HTTPException(
            403, detail="This invitation was sent to a different email address"
        )

    changed = await prisma.orginvitation.update_many(
        where={
            "id": invitation.id,
            "token": token,
            "acceptedAt": None,
            "revokedAt": None,
            "expiresAt": {"gt": datetime.now(timezone.utc)},
            "Org": {"is": {"deletedAt": None}},
        },
        data={"revokedAt": datetime.now(timezone.utc)},
    )
    if changed != 1:
        raise HTTPException(409, detail="Invitation is no longer available")


@router.get(
    "/pending",
    summary="List pending invitations for current user",
    tags=["invitations"],
    dependencies=[Security(requires_user)],
)
async def list_pending_for_user(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[UserInvitationResponse]:
    # Get user's email
    user = await prisma.user.find_unique(where={"id": user_id})
    if user is None:
        return []

    invitations = await prisma.orginvitation.find_many(
        where={
            "email": user.email,
            "acceptedAt": None,
            "revokedAt": None,
            "expiresAt": {"gt": datetime.now(timezone.utc)},
        },
        include={"Org": True},
        order={"createdAt": "desc"},
    )
    return [UserInvitationResponse.from_db(inv) for inv in invitations]
