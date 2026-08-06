"""Invitation API routes for organization membership."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import uuid4

from autogpt_libs.auth import get_user_id, requires_org_permission, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, HTTPException, Query, Security
from prisma.errors import UniqueViolationError
from prisma.models import OrgInvitation
from prisma.types import OrgInvitationWhereInput

from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

from . import db as org_db
from .model import CreateInvitationRequest, InvitationCreateResponse, InvitationResponse

logger = logging.getLogger(__name__)

router = APIRouter()

INVITATION_TTL_DAYS = 7


def _verify_org_path(ctx: RequestContext, org_id: str) -> None:
    """Ensure the authenticated user's active org matches the path parameter."""
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")


async def _get_org_invitation(org_id: str, invitation_id: str) -> OrgInvitation:
    """Load an invitation, enforcing that it belongs to `org_id`.

    Cross-org lookups are reported as "not found" rather than "forbidden" so
    the endpoint does not leak the existence of other orgs' invitations.
    """
    invitation = await prisma.orginvitation.find_unique(where={"id": invitation_id})
    if invitation is None or invitation.orgId != org_id:
        raise NotFoundError(f"Invitation {invitation_id} not found")
    return invitation


def _reject_if_not_pending(invitation: OrgInvitation) -> None:
    """Reject invitations that are no longer actionable (expiry is allowed)."""
    if invitation.acceptedAt is not None:
        raise HTTPException(400, detail="Invitation already accepted")
    if invitation.revokedAt is not None:
        raise HTTPException(400, detail="Invitation was revoked")


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

    # Reject team IDs outside this org at create time. The accept path's
    # add_team_member re-validates (and silently skips failures), so
    # without this check a poisoned invitation would fail silently at
    # accept instead of loudly at create.
    if request.team_ids:
        teams = await prisma.team.find_many(where={"id": {"in": request.team_ids}})
        valid_ids = {t.id for t in teams if t.orgId == org_id}
        invalid = [t for t in request.team_ids if t not in valid_ids]
        if invalid:
            raise HTTPException(
                400,
                detail=f"Teams not found in this organization: {invalid}",
            )

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
    await _get_org_invitation(org_id, invitation_id)

    await prisma.orginvitation.update(
        where={"id": invitation_id},
        data={"revokedAt": datetime.now(timezone.utc)},
    )


@org_router.post(
    "/{invitation_id}/resend",
    summary="Resend invitation",
    tags=["orgs", "invitations"],
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
    invitation = await _get_org_invitation(org_id, invitation_id)
    _reject_if_not_pending(invitation)

    new_token = str(uuid4())
    # Compare-and-swap: re-assert acceptedAt/revokedAt in the WHERE clause so a
    # concurrent accept/revoke landing between the read above and this write
    # cannot be overwritten with a fresh token. `update()` only accepts a unique
    # WHERE, so this uses `update_many()` (which returns the affected count).
    updated_count = await prisma.orginvitation.update_many(
        where={"id": invitation_id, "acceptedAt": None, "revokedAt": None},
        data={
            "token": new_token,
            "tokenHash": None,
            "teamIds": await _surviving_team_ids(invitation),
            "expiresAt": datetime.now(timezone.utc)
            + timedelta(days=INVITATION_TTL_DAYS),
        },
    )
    if updated_count == 0:
        # Lost the race. Re-read to report the same error the pre-check would
        # have: 404 if it was deleted, 400 if it was accepted/revoked.
        current = await _get_org_invitation(org_id, invitation_id)
        _reject_if_not_pending(current)
        raise HTTPException(400, detail="Invitation changed concurrently; retry")

    # Read back by the token we just minted so the response is provably the row
    # this request wrote. `update_many` returns a count, not the record.
    refreshed = await prisma.orginvitation.find_unique(where={"token": new_token})
    if refreshed is None:
        raise NotFoundError(f"Invitation {invitation_id} not found")

    # TODO: Send email via Postmark with invitation link (same gap as create).
    # Rate-limit resends (min-interval / per-invite cap) as part of that work —
    # unbounded resend is only an email-bombing vector once delivery is wired up.
    return InvitationCreateResponse.from_db(refreshed)


async def _surviving_team_ids(invitation: OrgInvitation) -> list[str]:
    """Drop team IDs that no longer exist in the org, logging what was dropped.

    `create_invitation` validates team IDs up front, but a team can be deleted
    between create and resend. Accept only logs-and-skips such teams, so a
    resent invite would silently promise access it can no longer grant.
    Pruning here keeps the stored invitation honest about what it confers.
    """
    if not invitation.teamIds:
        return []

    teams = await prisma.team.find_many(where={"id": {"in": invitation.teamIds}})
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

    # Add to specified workspaces. Failures are non-fatal (a team may have
    # been deleted between invite and accept) but must not be silent — the
    # user ends up an org member without the team access the invite
    # promised, which support needs to be able to trace.
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
            logger.warning(
                "Invitation accept: failed to add user %s to team %s in "
                "org %s (team deleted?); continuing with org membership",
                user_id,
                ws_id,
                invitation.orgId,
                exc_info=True,
            )

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
