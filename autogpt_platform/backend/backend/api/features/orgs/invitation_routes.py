"""Invitation API routes for organization membership."""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_org_permission, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, HTTPException, Security
from prisma.errors import UniqueViolationError

from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

from . import db as org_db
from .model import CreateInvitationRequest, InvitationCreateResponse, InvitationResponse

router = APIRouter()

INVITATION_TTL_DAYS = 7


def _verify_org_path(ctx: RequestContext, org_id: str) -> None:
    """Ensure the authenticated user's active org matches the path parameter."""
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")


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
    expires_at = datetime.now(timezone.utc) + timedelta(days=INVITATION_TTL_DAYS)

    invitation = await prisma.orginvitation.create(
        data={
            "orgId": org_id,
            "email": request.email,
            "isAdmin": request.is_admin,
            "isBillingManager": request.is_billing_manager,
            "expiresAt": expires_at,
            "invitedByUserId": ctx.user_id,
            "teamIds": request.team_ids,
        }
    )

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
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> list[InvitationResponse]:
    _verify_org_path(ctx, org_id)
    invitations = await prisma.orginvitation.find_many(
        where={
            "orgId": org_id,
            "acceptedAt": None,
            "revokedAt": None,
            "expiresAt": {"gt": datetime.now(timezone.utc)},
        },
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
    invitation = await prisma.orginvitation.find_unique(where={"id": invitation_id})
    if invitation is None or invitation.orgId != org_id:
        raise NotFoundError(f"Invitation {invitation_id} not found")

    await prisma.orginvitation.update(
        where={"id": invitation_id},
        data={"revokedAt": datetime.now(timezone.utc)},
    )


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

    # Add user to org (idempotent — handles race condition from concurrent accepts)
    try:
        await org_db.add_org_member(
            org_id=invitation.orgId,
            user_id=user_id,
            is_admin=invitation.isAdmin,
            is_billing_manager=invitation.isBillingManager,
            invited_by=invitation.invitedByUserId,
        )
    except UniqueViolationError:
        # User is already a member — treat as success (idempotent)
        pass

    # Add to specified workspaces
    for ws_id in invitation.teamIds:
        try:
            from . import team_db as team_db

            await team_db.add_team_member(
                ws_id=ws_id,
                user_id=user_id,
                org_id=invitation.orgId,
                invited_by=invitation.invitedByUserId,
            )
        except Exception:
            # Non-fatal -- workspace may have been deleted
            pass

    # Mark invitation as accepted
    await prisma.orginvitation.update(
        where={"id": invitation.id},
        data={"acceptedAt": datetime.now(timezone.utc), "targetUserId": user_id},
    )

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

    await prisma.orginvitation.update(
        where={"id": invitation.id},
        data={"revokedAt": datetime.now(timezone.utc)},
    )


@router.get(
    "/pending",
    summary="List pending invitations for current user",
    tags=["invitations"],
    dependencies=[Security(requires_user)],
)
async def list_pending_for_user(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[InvitationResponse]:
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
        order={"createdAt": "desc"},
    )
    return [InvitationResponse.from_db(inv) for inv in invitations]
