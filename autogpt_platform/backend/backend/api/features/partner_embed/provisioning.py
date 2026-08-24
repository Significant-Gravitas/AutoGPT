"""JIT provisioning for partner shadow users and customer tenants."""

import uuid
from dataclasses import dataclass

from prisma import Prisma

from backend.api.features.partner_embed.models import (
    ProvisionPartnerIdentityRequest,
    ProvisionPartnerIdentityResponse,
    ShadowIdentityIDs,
)
from backend.data.db import transaction
from backend.util.json import SafeJson

_PARTNER_IDENTITY_NAMESPACE = uuid.UUID("1d9f79de-2bdf-4be7-a6e4-371704b39bad")


@dataclass(frozen=True)
class _ProvisioningContext:
    request: ProvisionPartnerIdentityRequest
    ids: ShadowIdentityIDs
    slug: str
    synthetic_email: str
    user_metadata: SafeJson
    org_settings: SafeJson


def derive_shadow_identity_ids(
    request: ProvisionPartnerIdentityRequest,
) -> ShadowIdentityIDs:
    """Derive stable internal IDs without using mutable email addresses."""
    user_id = uuid.uuid5(
        _PARTNER_IDENTITY_NAMESPACE,
        f"{request.partner_id}:user:{request.external_subject}",
    )
    organization_id = uuid.uuid5(
        _PARTNER_IDENTITY_NAMESPACE,
        f"{request.partner_id}:account:{request.external_account_id}",
    )
    team_id = uuid.uuid5(
        _PARTNER_IDENTITY_NAMESPACE,
        f"{organization_id}:default-team",
    )
    return ShadowIdentityIDs(
        user_id=str(user_id),
        organization_id=str(organization_id),
        team_id=str(team_id),
    )


async def provision_partner_identity(
    request: ProvisionPartnerIdentityRequest,
) -> ProvisionPartnerIdentityResponse:
    """Create or refresh the platform rows needed by an embedded principal."""
    context = _build_context(request)
    async with transaction() as tx:
        await _upsert_user(tx, context)
        is_first_member = await _upsert_organization(tx, context)
        await _upsert_team(tx, context)
        await _upsert_org_membership(tx, context, is_first_member)
        await _upsert_team_membership(tx, context, is_first_member)
        await _upsert_org_profile(tx, context)
        await _upsert_seat_assignment(tx, context)
        await _upsert_org_balance(tx, context)

    return ProvisionPartnerIdentityResponse(
        user_id=context.ids.user_id,
        organization_id=context.ids.organization_id,
        team_id=context.ids.team_id,
    )


def _build_context(request: ProvisionPartnerIdentityRequest) -> _ProvisioningContext:
    ids = derive_shadow_identity_ids(request)
    return _ProvisioningContext(
        request=request,
        ids=ids,
        slug=f"embed-{ids.organization_id.replace('-', '')[:20]}",
        synthetic_email=f"embed+{ids.user_id}@partners.autogpt.local",
        user_metadata=SafeJson(
            {
                "partner_embed": {
                    "partner_id": request.partner_id,
                    "external_subject": request.external_subject,
                    "external_email": str(request.email),
                }
            }
        ),
        org_settings=SafeJson(
            {
                "partner_embed": {
                    "partner_id": request.partner_id,
                    "external_account_id": request.external_account_id,
                }
            }
        ),
    )


async def _upsert_user(tx: Prisma, context: _ProvisioningContext) -> None:
    await tx.user.upsert(
        where={"id": context.ids.user_id},
        data={
            "create": {
                "id": context.ids.user_id,
                "email": context.synthetic_email,
                "name": context.request.display_name,
                "metadata": context.user_metadata,
            },
            "update": {
                "name": context.request.display_name,
                "metadata": context.user_metadata,
            },
        },
    )


async def _upsert_organization(
    tx: Prisma,
    context: _ProvisioningContext,
) -> bool:
    existing = await tx.organization.find_unique(
        where={"id": context.ids.organization_id}
    )
    await tx.organization.upsert(
        where={"id": context.ids.organization_id},
        data={
            "create": {
                "id": context.ids.organization_id,
                "name": context.request.account_name,
                "slug": context.slug,
                "isPersonal": False,
                "bootstrapUserId": context.ids.user_id,
                "settings": context.org_settings,
            },
            "update": {
                "name": context.request.account_name,
                "settings": context.org_settings,
                "deletedAt": None,
            },
        },
    )
    return existing is None


async def _upsert_team(tx: Prisma, context: _ProvisioningContext) -> None:
    await tx.team.upsert(
        where={"id": context.ids.team_id},
        data={
            "create": {
                "id": context.ids.team_id,
                "name": "Default",
                "orgId": context.ids.organization_id,
                "isDefault": True,
                "joinPolicy": "OPEN",
                "createdByUserId": context.ids.user_id,
            },
            "update": {"archivedAt": None},
        },
    )


async def _upsert_org_membership(
    tx: Prisma,
    context: _ProvisioningContext,
    is_first_member: bool,
) -> None:
    is_admin = is_first_member or context.request.is_admin
    await tx.orgmember.upsert(
        where={
            "orgId_userId": {
                "orgId": context.ids.organization_id,
                "userId": context.ids.user_id,
            }
        },
        data={
            "create": {
                "orgId": context.ids.organization_id,
                "userId": context.ids.user_id,
                "isOwner": is_first_member,
                "isAdmin": is_admin,
                "status": "ACTIVE",
            },
            "update": {"isAdmin": is_admin, "status": "ACTIVE"},
        },
    )


async def _upsert_team_membership(
    tx: Prisma,
    context: _ProvisioningContext,
    is_first_member: bool,
) -> None:
    is_admin = is_first_member or context.request.is_admin
    await tx.teammember.upsert(
        where={
            "teamId_userId": {
                "teamId": context.ids.team_id,
                "userId": context.ids.user_id,
            }
        },
        data={
            "create": {
                "teamId": context.ids.team_id,
                "userId": context.ids.user_id,
                "isAdmin": is_admin,
                "status": "ACTIVE",
            },
            "update": {"isAdmin": is_admin, "status": "ACTIVE"},
        },
    )


async def _upsert_org_profile(tx: Prisma, context: _ProvisioningContext) -> None:
    await tx.organizationprofile.upsert(
        where={"organizationId": context.ids.organization_id},
        data={
            "create": {
                "organizationId": context.ids.organization_id,
                "username": context.slug,
                "displayName": context.request.account_name,
            },
            "update": {"displayName": context.request.account_name},
        },
    )


async def _upsert_seat_assignment(
    tx: Prisma,
    context: _ProvisioningContext,
) -> None:
    await tx.organizationseatassignment.upsert(
        where={
            "organizationId_userId": {
                "organizationId": context.ids.organization_id,
                "userId": context.ids.user_id,
            }
        },
        data={
            "create": {
                "organizationId": context.ids.organization_id,
                "userId": context.ids.user_id,
                "seatType": "FREE",
                "status": "ACTIVE",
                "assignedByUserId": context.ids.user_id,
            },
            "update": {"status": "ACTIVE"},
        },
    )


async def _upsert_org_balance(tx: Prisma, context: _ProvisioningContext) -> None:
    await tx.orgbalance.upsert(
        where={"orgId": context.ids.organization_id},
        data={
            "create": {"orgId": context.ids.organization_id, "balance": 0},
            "update": {},
        },
    )
