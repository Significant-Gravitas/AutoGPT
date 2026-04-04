"""Database operations for organization management."""

import logging
from datetime import datetime, timezone

import prisma.errors

from backend.data.db import prisma
from backend.data.org_migration import _resolve_unique_slug, _sanitize_slug
from backend.util.exceptions import NotFoundError

from .model import OrgAliasResponse, OrgMemberResponse, OrgResponse, UpdateOrgData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


async def get_user_default_org_workspace(
    user_id: str,
) -> tuple[str | None, str | None]:
    """Get the user's personal org ID and its default workspace ID.

    Returns (organization_id, workspace_id). Either may be None if
    the user has no org (e.g., migration hasn't run yet).
    """
    member = await prisma.orgmember.find_first(
        where={
            "userId": user_id,
            "isOwner": True,
            "Org": {"isPersonal": True, "deletedAt": None},
        },
    )
    if member is None:
        logger.warning(
            f"User {user_id} has no personal org — account may be in inconsistent state"
        )
        return None, None

    org_id = member.orgId
    workspace = await prisma.orgworkspace.find_first(
        where={"orgId": org_id, "isDefault": True}
    )
    ws_id = workspace.id if workspace else None
    return org_id, ws_id


async def _create_personal_org_for_user(
    user_id: str,
    slug_base: str,
    display_name: str,
) -> OrgResponse:
    """Create a new personal org with all required records.

    Used by both initial org creation (migration) and conversion (spawning
    a new personal org when the old one becomes a team org).
    """
    slug = await _resolve_unique_slug(slug_base)

    org = await prisma.organization.create(
        data={
            "name": display_name,
            "slug": slug,
            "isPersonal": True,
            "bootstrapUserId": user_id,
            "settings": "{}",
        }
    )

    await prisma.orgmember.create(
        data={
            "orgId": org.id,
            "userId": user_id,
            "isOwner": True,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    workspace = await prisma.orgworkspace.create(
        data={
            "name": "Default",
            "orgId": org.id,
            "isDefault": True,
            "joinPolicy": "OPEN",
            "createdByUserId": user_id,
        }
    )

    await prisma.orgworkspacemember.create(
        data={
            "workspaceId": workspace.id,
            "userId": user_id,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    await prisma.organizationprofile.create(
        data={
            "organizationId": org.id,
            "username": slug,
            "displayName": display_name,
        }
    )

    await prisma.organizationseatassignment.create(
        data={
            "organizationId": org.id,
            "userId": user_id,
            "seatType": "FREE",
            "status": "ACTIVE",
            "assignedByUserId": user_id,
        }
    )

    # Create zero-balance row so credit operations don't need upsert
    await prisma.orgbalance.create(data={"orgId": org.id, "balance": 0})

    return OrgResponse.from_db(org, member_count=1)


# ---------------------------------------------------------------------------
# Org CRUD
# ---------------------------------------------------------------------------


async def create_org(
    name: str,
    slug: str,
    user_id: str,
    description: str | None = None,
) -> OrgResponse:
    """Create a team organization and make the user the owner.

    Raises:
        ValueError: If the slug is already taken by another org or alias.
    """
    existing_org = await prisma.organization.find_unique(where={"slug": slug})
    if existing_org:
        raise ValueError(f"Slug '{slug}' is already in use")
    existing_alias = await prisma.organizationalias.find_unique(
        where={"aliasSlug": slug}
    )
    if existing_alias:
        raise ValueError(f"Slug '{slug}' is already in use as an alias")

    org = await prisma.organization.create(
        data={
            "name": name,
            "slug": slug,
            "description": description,
            "isPersonal": False,
            "bootstrapUserId": user_id,
            "settings": "{}",
        }
    )

    await prisma.orgmember.create(
        data={
            "orgId": org.id,
            "userId": user_id,
            "isOwner": True,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    workspace = await prisma.orgworkspace.create(
        data={
            "name": "Default",
            "orgId": org.id,
            "isDefault": True,
            "joinPolicy": "OPEN",
            "createdByUserId": user_id,
        }
    )

    await prisma.orgworkspacemember.create(
        data={
            "workspaceId": workspace.id,
            "userId": user_id,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    await prisma.organizationprofile.create(
        data={
            "organizationId": org.id,
            "username": slug,
            "displayName": name,
        }
    )

    await prisma.organizationseatassignment.create(
        data={
            "organizationId": org.id,
            "userId": user_id,
            "seatType": "FREE",
            "status": "ACTIVE",
            "assignedByUserId": user_id,
        }
    )

    # Create zero-balance row so credit operations don't need upsert
    await prisma.orgbalance.create(data={"orgId": org.id, "balance": 0})

    return OrgResponse.from_db(org, member_count=1)


async def list_user_orgs(user_id: str) -> list[OrgResponse]:
    """List all non-deleted organizations the user belongs to."""
    memberships = await prisma.orgmember.find_many(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "Org": {"deletedAt": None},
        },
        include={"Org": True},
    )
    results = []
    for m in memberships:
        org = m.Org
        if org is None:
            continue
        results.append(OrgResponse.from_db(org))
    return results


async def get_org(org_id: str) -> OrgResponse:
    """Get organization details."""
    org = await prisma.organization.find_unique(where={"id": org_id})
    if org is None or org.deletedAt is not None:
        raise NotFoundError(f"Organization {org_id} not found")
    return OrgResponse.from_db(org)


async def update_org(org_id: str, data: UpdateOrgData) -> OrgResponse:
    """Update organization fields. Creates a RENAME alias if slug changes.

    Only accepts the structured UpdateOrgData model — no arbitrary dict keys.
    """
    update_dict: dict = {}
    if data.name is not None:
        update_dict["name"] = data.name
    if data.description is not None:
        update_dict["description"] = data.description
    if data.avatar_url is not None:
        update_dict["avatarUrl"] = data.avatar_url

    if data.slug is not None:
        existing = await prisma.organization.find_unique(where={"slug": data.slug})
        if existing and existing.id != org_id:
            raise ValueError(f"Slug '{data.slug}' is already in use")
        existing_alias = await prisma.organizationalias.find_unique(
            where={"aliasSlug": data.slug}
        )
        if existing_alias:
            raise ValueError(f"Slug '{data.slug}' is already in use as an alias")

        old_org = await prisma.organization.find_unique(where={"id": org_id})
        if old_org and old_org.slug != data.slug:
            await prisma.organizationalias.create(
                data={
                    "organizationId": org_id,
                    "aliasSlug": old_org.slug,
                    "aliasType": "RENAME",
                }
            )
        update_dict["slug"] = data.slug

    if not update_dict:
        return await get_org(org_id)

    await prisma.organization.update(where={"id": org_id}, data=update_dict)
    return await get_org(org_id)


async def delete_org(org_id: str) -> None:
    """Soft-delete an organization. Cannot delete personal orgs.

    Sets deletedAt instead of hard-deleting to preserve financial records.
    """
    org = await prisma.organization.find_unique(where={"id": org_id})
    if org is None:
        raise NotFoundError(f"Organization {org_id} not found")
    if org.isPersonal:
        raise ValueError("Cannot delete a personal organization. Convert it first.")
    if org.deletedAt is not None:
        raise ValueError("Organization is already deleted")

    await prisma.organization.update(
        where={"id": org_id},
        data={"deletedAt": datetime.now(timezone.utc)},
    )


async def convert_personal_org(org_id: str, user_id: str) -> OrgResponse:
    """Convert a personal org to a team org.

    Creates a new personal org for the user so they always have one.
    Existing resources (agents, credits, store listings) stay in the
    team org — that's the point of converting.

    If new personal org creation fails, the conversion is rolled back.
    """
    org = await prisma.organization.find_unique(where={"id": org_id})
    if org is None:
        raise NotFoundError(f"Organization {org_id} not found")
    if not org.isPersonal:
        raise ValueError("Organization is already a team org")

    # Step 1: Flip isPersonal on the old org
    await prisma.organization.update(
        where={"id": org_id},
        data={"isPersonal": False},
    )

    # Step 2: Create a new personal org for the user
    try:
        slug_base = f"{_sanitize_slug(org.slug)}-personal-1"
        # Fetch user name for display
        user = await prisma.user.find_unique(where={"id": user_id})
        display_name = user.name if user and user.name else org.name

        await _create_personal_org_for_user(
            user_id=user_id,
            slug_base=slug_base,
            display_name=display_name,
        )
    except Exception:
        # Roll back: restore isPersonal on the old org
        logger.exception(
            f"Failed to create new personal org for user {user_id} during "
            f"conversion of org {org_id} — rolling back"
        )
        await prisma.organization.update(
            where={"id": org_id},
            data={"isPersonal": True},
        )
        raise

    return await get_org(org_id)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


async def list_org_members(org_id: str) -> list[OrgMemberResponse]:
    """List all active members of an organization."""
    members = await prisma.orgmember.find_many(
        where={"orgId": org_id, "status": "ACTIVE"},
        include={"User": True},
    )
    return [OrgMemberResponse.from_db(m) for m in members]


async def add_org_member(
    org_id: str,
    user_id: str,
    is_admin: bool = False,
    is_billing_manager: bool = False,
    invited_by: str | None = None,
) -> OrgMemberResponse:
    """Add a member to an organization and its default workspace."""
    member = await prisma.orgmember.create(
        data={
            "orgId": org_id,
            "userId": user_id,
            "isAdmin": is_admin,
            "isBillingManager": is_billing_manager,
            "status": "ACTIVE",
            "invitedByUserId": invited_by,
        },
        include={"User": True},
    )

    default_ws = await prisma.orgworkspace.find_first(
        where={"orgId": org_id, "isDefault": True}
    )
    if default_ws:
        await prisma.orgworkspacemember.create(
            data={
                "workspaceId": default_ws.id,
                "userId": user_id,
                "status": "ACTIVE",
            }
        )

    return OrgMemberResponse.from_db(member)


async def update_org_member(
    org_id: str, user_id: str, is_admin: bool | None, is_billing_manager: bool | None
) -> OrgMemberResponse:
    """Update a member's role flags."""
    member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )
    if member is None:
        raise NotFoundError(f"Member {user_id} not found in org {org_id}")
    if member.isOwner:
        raise ValueError(
            "Cannot change the owner's role flags directly. Use transfer-ownership."
        )

    update_data: dict = {}
    if is_admin is not None:
        update_data["isAdmin"] = is_admin
    if is_billing_manager is not None:
        update_data["isBillingManager"] = is_billing_manager

    if update_data:
        await prisma.orgmember.update(
            where={"orgId_userId": {"orgId": org_id, "userId": user_id}},
            data=update_data,
        )
    members = await list_org_members(org_id)
    match = next((m for m in members if m.user_id == user_id), None)
    if match is None:
        raise NotFoundError(f"Member {user_id} not found in org {org_id} after update")
    return match


async def remove_org_member(org_id: str, user_id: str, requesting_user_id: str) -> None:
    """Remove a member from an organization and all its workspaces.

    Guards:
    - Cannot remove the org owner (transfer ownership first)
    - Cannot remove yourself (use leave flow instead)
    - Cannot remove a user who has active schedules (transfer/cancel first)
    - Cannot remove a user who would become org-less (no other org memberships)
    """
    member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )
    if member is None:
        raise NotFoundError(f"Member {user_id} not found in org {org_id}")
    if member.isOwner:
        raise ValueError("Cannot remove the org owner. Transfer ownership first.")
    if user_id == requesting_user_id:
        raise ValueError(
            "Cannot remove yourself from an organization. "
            "Ask another admin to remove you, or transfer ownership first."
        )

    # Check if user would become org-less
    other_memberships = await prisma.orgmember.count(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "orgId": {"not": org_id},
            "Org": {"deletedAt": None},
        }
    )
    if other_memberships == 0:
        raise ValueError(
            "Cannot remove this member — they have no other organization memberships "
            "and would be locked out. They must join or create another org first."
        )

    # Check for active schedules
    # TODO: Check APScheduler for active schedules owned by this user in this org
    # For now, this is a placeholder for the schedule transfer requirement

    # Remove from all workspaces in this org
    workspaces = await prisma.orgworkspace.find_many(where={"orgId": org_id})
    for ws in workspaces:
        await prisma.orgworkspacemember.delete_many(
            where={"workspaceId": ws.id, "userId": user_id}
        )

    # Remove org membership
    await prisma.orgmember.delete(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )


async def transfer_ownership(
    org_id: str, current_owner_id: str, new_owner_id: str
) -> None:
    """Transfer org ownership atomically. Both updates happen in one statement."""
    if current_owner_id == new_owner_id:
        raise ValueError("Cannot transfer ownership to the same user")

    current = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": current_owner_id}}
    )
    if current is None or not current.isOwner:
        raise ValueError("Current user is not the org owner")

    new = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": new_owner_id}}
    )
    if new is None:
        raise NotFoundError(f"User {new_owner_id} is not a member of org {org_id}")

    await prisma.execute_raw(
        """
        UPDATE "OrgMember"
        SET "isOwner" = CASE
                WHEN "userId" = $1 THEN false
                WHEN "userId" = $2 THEN true
                ELSE "isOwner"
            END,
            "isAdmin" = CASE
                WHEN "userId" = $2 THEN true
                ELSE "isAdmin"
            END,
            "updatedAt" = NOW()
        WHERE "orgId" = $3 AND "userId" IN ($1, $2)
        """,
        current_owner_id,
        new_owner_id,
        org_id,
    )


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


async def list_org_aliases(org_id: str) -> list[OrgAliasResponse]:
    """List all aliases for an organization."""
    aliases = await prisma.organizationalias.find_many(
        where={"organizationId": org_id, "removedAt": None}
    )
    return [OrgAliasResponse.from_db(a) for a in aliases]


async def create_org_alias(
    org_id: str, alias_slug: str, user_id: str
) -> OrgAliasResponse:
    """Create a new alias for an organization."""
    existing_org = await prisma.organization.find_unique(where={"slug": alias_slug})
    if existing_org:
        raise ValueError(f"Slug '{alias_slug}' is already used by an organization")

    existing_alias = await prisma.organizationalias.find_unique(
        where={"aliasSlug": alias_slug}
    )
    if existing_alias:
        raise ValueError(f"Slug '{alias_slug}' is already used as an alias")

    alias = await prisma.organizationalias.create(
        data={
            "organizationId": org_id,
            "aliasSlug": alias_slug,
            "aliasType": "MANUAL",
            "createdByUserId": user_id,
        }
    )
    return OrgAliasResponse.from_db(alias)
