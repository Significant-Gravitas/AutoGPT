"""Database operations for organization management."""

import logging

from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


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
            "Org": {"isPersonal": True},
        },
    )
    if member is None:
        return None, None

    org_id = member.orgId
    workspace = await prisma.orgworkspace.find_first(
        where={"orgId": org_id, "isDefault": True}
    )
    ws_id = workspace.id if workspace else None
    return org_id, ws_id


async def create_org(
    name: str,
    slug: str,
    user_id: str,
    description: str | None = None,
) -> dict:
    """Create an organization and make the user the owner.

    Also creates a default workspace and adds the user to it.

    Raises:
        ValueError: If the slug is already taken by another org or alias.
    """
    # Check slug uniqueness (org slugs + alias slugs)
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

    # Create owner membership
    await prisma.orgmember.create(
        data={
            "orgId": org.id,
            "userId": user_id,
            "isOwner": True,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    # Create default workspace
    workspace = await prisma.orgworkspace.create(
        data={
            "name": "Default",
            "orgId": org.id,
            "isDefault": True,
            "joinPolicy": "OPEN",
            "createdByUserId": user_id,
        }
    )

    # Add user to default workspace
    await prisma.orgworkspacemember.create(
        data={
            "workspaceId": workspace.id,
            "userId": user_id,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    # Create org profile
    await prisma.organizationprofile.create(
        data={
            "organizationId": org.id,
            "username": slug,
            "displayName": name,
        }
    )

    # Create seat assignment
    await prisma.organizationseatassignment.create(
        data={
            "organizationId": org.id,
            "userId": user_id,
            "seatType": "FREE",
            "status": "ACTIVE",
            "assignedByUserId": user_id,
        }
    )

    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "avatarUrl": org.avatarUrl,
        "description": org.description,
        "isPersonal": org.isPersonal,
        "memberCount": 1,
        "createdAt": org.createdAt,
    }


async def list_user_orgs(user_id: str) -> list[dict]:
    """List all organizations the user belongs to."""
    memberships = await prisma.orgmember.find_many(
        where={"userId": user_id, "status": "ACTIVE"},
        include={
            "Org": {
                "include": {
                    "_count": {"select": {"Members": {"where": {"status": "ACTIVE"}}}}
                }
            }
        },
    )
    results = []
    for m in memberships:
        org = m.Org
        if org is None:
            continue
        member_count = (
            getattr(org, "Members_count", 0) if hasattr(org, "Members_count") else 0
        )
        results.append(
            {
                "id": org.id,
                "name": org.name,
                "slug": org.slug,
                "avatarUrl": org.avatarUrl,
                "description": org.description,
                "isPersonal": org.isPersonal,
                "memberCount": member_count,
                "createdAt": org.createdAt,
            }
        )
    return results


async def get_org(org_id: str) -> dict:
    """Get organization details."""
    org = await prisma.organization.find_unique(
        where={"id": org_id},
        include={"_count": {"select": {"Members": {"where": {"status": "ACTIVE"}}}}},
    )
    if org is None:
        raise NotFoundError(f"Organization {org_id} not found")

    member_count = (
        getattr(org, "Members_count", 0) if hasattr(org, "Members_count") else 0
    )
    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "avatarUrl": org.avatarUrl,
        "description": org.description,
        "isPersonal": org.isPersonal,
        "memberCount": member_count,
        "createdAt": org.createdAt,
    }


async def update_org(org_id: str, data: dict) -> dict:
    """Update organization fields. Creates a RENAME alias if slug changes."""
    update_data = {k: v for k, v in data.items() if v is not None}
    if not update_data:
        return await get_org(org_id)

    # If slug is changing, validate uniqueness and create alias for old slug
    new_slug = update_data.get("slug")
    if new_slug:
        existing = await prisma.organization.find_unique(where={"slug": new_slug})
        if existing and existing.id != org_id:
            raise ValueError(f"Slug '{new_slug}' is already in use")
        existing_alias = await prisma.organizationalias.find_unique(
            where={"aliasSlug": new_slug}
        )
        if existing_alias:
            raise ValueError(f"Slug '{new_slug}' is already in use as an alias")

        # Create alias for the old slug so old URLs keep working
        old_org = await prisma.organization.find_unique(where={"id": org_id})
        if old_org and old_org.slug != new_slug:
            await prisma.organizationalias.create(
                data={
                    "organizationId": org_id,
                    "aliasSlug": old_org.slug,
                    "aliasType": "RENAME",
                }
            )

    await prisma.organization.update(where={"id": org_id}, data=update_data)
    return await get_org(org_id)


async def delete_org(org_id: str) -> None:
    """Delete an organization. Cannot delete personal orgs."""
    org = await prisma.organization.find_unique(where={"id": org_id})
    if org is None:
        raise NotFoundError(f"Organization {org_id} not found")
    if org.isPersonal:
        raise ValueError("Cannot delete a personal organization. Convert it first.")

    await prisma.organization.delete(where={"id": org_id})


async def convert_personal_org(org_id: str) -> dict:
    """Convert a personal org to a team org (one-way)."""
    org = await prisma.organization.find_unique(where={"id": org_id})
    if org is None:
        raise NotFoundError(f"Organization {org_id} not found")
    if not org.isPersonal:
        raise ValueError("Organization is already a team org")

    await prisma.organization.update(
        where={"id": org_id},
        data={"isPersonal": False},
    )
    return await get_org(org_id)


async def list_org_members(org_id: str) -> list[dict]:
    """List all active members of an organization."""
    members = await prisma.orgmember.find_many(
        where={"orgId": org_id, "status": "ACTIVE"},
        include={"User": True},
    )
    return [
        {
            "id": m.id,
            "userId": m.userId,
            "email": m.User.email if m.User else "",
            "name": m.User.name if m.User else None,
            "isOwner": m.isOwner,
            "isAdmin": m.isAdmin,
            "isBillingManager": m.isBillingManager,
            "joinedAt": m.joinedAt,
        }
        for m in members
    ]


async def add_org_member(
    org_id: str,
    user_id: str,
    is_admin: bool = False,
    is_billing_manager: bool = False,
    invited_by: str | None = None,
) -> dict:
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

    # Auto-add to default workspace
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

    return {
        "id": member.id,
        "userId": member.userId,
        "email": member.User.email if member.User else "",
        "name": member.User.name if member.User else None,
        "isOwner": member.isOwner,
        "isAdmin": member.isAdmin,
        "isBillingManager": member.isBillingManager,
        "joinedAt": member.joinedAt,
    }


async def update_org_member(
    org_id: str, user_id: str, is_admin: bool | None, is_billing_manager: bool | None
) -> dict:
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
    return next(m for m in members if m["userId"] == user_id)


async def remove_org_member(org_id: str, user_id: str) -> None:
    """Remove a member from an organization and all its workspaces."""
    member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )
    if member is None:
        raise NotFoundError(f"Member {user_id} not found in org {org_id}")
    if member.isOwner:
        raise ValueError("Cannot remove the org owner. Transfer ownership first.")

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
    """Transfer org ownership atomically. Both updates happen in one transaction."""
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

    # Atomic transfer — both updates in one SQL statement to prevent ownerless window
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


async def list_org_aliases(org_id: str) -> list[dict]:
    """List all aliases for an organization."""
    aliases = await prisma.organizationalias.find_many(
        where={"organizationId": org_id, "removedAt": None}
    )
    return [
        {
            "id": a.id,
            "aliasSlug": a.aliasSlug,
            "aliasType": a.aliasType,
            "createdAt": a.createdAt,
        }
        for a in aliases
    ]


async def create_org_alias(org_id: str, alias_slug: str, user_id: str) -> dict:
    """Create a new alias for an organization."""
    # Check if slug is already taken by an org or alias
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
    return {
        "id": alias.id,
        "aliasSlug": alias.aliasSlug,
        "aliasType": alias.aliasType,
        "createdAt": alias.createdAt,
    }
