"""Database operations for organization management."""

import asyncio
import logging
from datetime import datetime, timezone

from prisma.errors import UniqueViolationError

from backend.data.db import prisma, transaction
from backend.data.org_migration import (
    _sanitize_slug,
    _soft_delete_blocking_orphan,
    create_personal_org,
)
from backend.util.exceptions import NotFoundError

from .model import OrgAliasResponse, OrgMemberResponse, OrgResponse, UpdateOrgData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


async def _find_personal_org_member(user_id: str):
    # Ordered oldest-first so this agrees with get_request_context (auth) and
    # _find_owned_personal_org (org_migration) on the canonical personal org
    # when a user briefly has more than one.
    return await prisma.orgmember.find_first(
        where={
            "userId": user_id,
            "isOwner": True,
            "Org": {"isPersonal": True, "deletedAt": None},
        },
        order={"createdAt": "asc"},
    )


async def _bootstrap_personal_org(user_id: str) -> str | None:
    """Create the personal org for a user who has none.

    The startup migration only covers users existing at boot — every NEW
    signup lands here on their first request. Redis-locked (single-key
    SET NX, cluster-safe) because a fresh login fires many API calls in
    parallel and each would otherwise race to create an org; losers wait
    for the winner instead.
    """
    from backend.data.redis_client import get_redis_async

    lock_key = f"personal-org-bootstrap:{user_id}"
    try:
        # Bounded: an unreachable redis must surface as a transient failure
        # (the caller 400s and the client retries), never as a hung socket
        # read stalling every first-touch request.
        redis = await asyncio.wait_for(get_redis_async(), timeout=10)
        acquired = await asyncio.wait_for(
            redis.set(lock_key, "1", nx=True, ex=30), timeout=10
        )
    except Exception:
        logger.error(
            f"Personal-org bootstrap lock unavailable for {user_id} "
            "(redis unreachable?) — failing closed",
            exc_info=True,
        )
        return None
    if not acquired:
        for _ in range(40):
            await asyncio.sleep(0.25)
            member = await _find_personal_org_member(user_id)
            if member is not None:
                return member.orgId
        logger.error(
            f"Timed out waiting for concurrent personal-org bootstrap of {user_id}"
        )
        return None

    try:
        # Re-check under the lock — the row may have appeared between the
        # caller's miss and our acquisition.
        member = await _find_personal_org_member(user_id)
        if member is not None:
            return member.orgId

        user = await prisma.user.find_unique(where={"id": user_id})
        if user is None:
            logger.warning(f"Cannot bootstrap personal org: user {user_id} not found")
            return None

        local_part = user.email.split("@")[0] if user.email else "user"
        slug_base = _sanitize_slug(local_part) or "user"
        display_name = user.name or local_part
        try:
            org = await _create_personal_org_for_user(user_id, slug_base, display_name)
        except UniqueViolationError:
            # Lost a race with the sign-up path (ensure_personal_org doesn't
            # take this Redis lock) — the org already exists; re-read the
            # membership instead of surfacing a 500 to the first request.
            member = await _find_personal_org_member(user_id)
            if member is not None:
                logger.info(
                    f"Personal-org bootstrap for {user_id} lost creation race; "
                    "using existing org"
                )
                return member.orgId
            # No membership: a legacy orphan may be squatting on the user's
            # one-personal-per-user index slot — clear it and retry once so
            # a first-touch request self-heals instead of degrading.
            if await _soft_delete_blocking_orphan(user_id):
                try:
                    org = await _create_personal_org_for_user(
                        user_id, slug_base, display_name
                    )
                    logger.info(
                        f"Bootstrapped personal org {org.id} for user {user_id} "
                        "after clearing a blocking orphan"
                    )
                    return org.id
                except UniqueViolationError:
                    # A concurrent creator can win between the orphan clear
                    # and this retry — reconcile before reporting failure.
                    member = await _find_personal_org_member(user_id)
                    if member is not None:
                        return member.orgId
                    logger.error(
                        f"Personal-org bootstrap for {user_id} still failing "
                        "after orphan cleanup",
                        exc_info=True,
                    )
                    return None
            logger.error(
                f"Personal-org bootstrap for {user_id} hit a unique violation "
                "but no membership exists",
                exc_info=True,
            )
            return None
        logger.info(f"Bootstrapped personal org {org.id} for new user {user_id}")
        return org.id
    finally:
        try:
            await redis.delete(lock_key)
        except Exception:
            pass


async def get_user_default_team(
    user_id: str,
) -> tuple[str | None, str | None]:
    """Get the user's personal org ID and its default workspace ID.

    Self-healing: users created after the startup migration (i.e. every
    new signup) have no personal org yet — one is bootstrapped on first
    touch so their first request doesn't fail with "no org context".

    Returns (organization_id, team_id). Either may be None if the user
    row itself is missing or bootstrap failed.
    """
    member = await _find_personal_org_member(user_id)
    if member is not None:
        org_id = member.orgId
    else:
        org_id = await _bootstrap_personal_org(user_id)
        if org_id is None:
            logger.warning(
                f"User {user_id} has no personal org — "
                "account may be in inconsistent state"
            )
            return None, None

    workspace = await prisma.team.find_first(where={"orgId": org_id, "isDefault": True})
    ws_id = workspace.id if workspace else None
    return org_id, ws_id


async def _create_personal_org_for_user(
    user_id: str,
    slug_base: str,
    display_name: str,
) -> OrgResponse:
    """Create a new personal org with all required records.

    Thin wrapper over the data-layer ``create_personal_org`` so the sign-up
    bootstrap, org conversion, and the backfill all share one record shape.
    Used here by conversion (spawning a new personal org when the old one
    becomes a team org).
    """
    org = await create_personal_org(user_id, slug_base, display_name)
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

    # One transaction: a failure partway must not leave an org without its
    # default workspace, owner membership, profile, seat, or balance row.
    async with transaction() as tx:
        org = await tx.organization.create(
            data={
                "name": name,
                "slug": slug,
                "description": description,
                "isPersonal": False,
                "bootstrapUserId": user_id,
                "settings": "{}",
            }
        )

        await tx.orgmember.create(
            data={
                "orgId": org.id,
                "userId": user_id,
                "isOwner": True,
                "isAdmin": True,
                "status": "ACTIVE",
            }
        )

        workspace = await tx.team.create(
            data={
                "name": "Default",
                "orgId": org.id,
                "isDefault": True,
                "joinPolicy": "OPEN",
                "createdByUserId": user_id,
            }
        )

        await tx.teammember.create(
            data={
                "teamId": workspace.id,
                "userId": user_id,
                "isAdmin": True,
                "status": "ACTIVE",
            }
        )

        await tx.organizationprofile.create(
            data={
                "organizationId": org.id,
                "username": slug,
                "displayName": name,
            }
        )

        await tx.organizationseatassignment.create(
            data={
                "organizationId": org.id,
                "userId": user_id,
                "seatType": "FREE",
                "status": "ACTIVE",
                "assignedByUserId": user_id,
            }
        )

        # Create zero-balance row so credit operations don't need upsert
        await tx.orgbalance.create(data={"orgId": org.id, "balance": 0})

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
        # Only a DIFFERENT org's alias blocks the slug — after a rename the
        # previous slug is stored as this org's own RENAME alias, and
        # renaming back must promote it, not fail with "already in use".
        if existing_alias and existing_alias.organizationId != org_id:
            raise ValueError(f"Slug '{data.slug}' is already in use as an alias")

        old_org = await prisma.organization.find_unique(where={"id": org_id})
        if old_org and old_org.slug != data.slug:
            if existing_alias:
                # Re-claiming our own alias as the primary slug.
                await prisma.organizationalias.delete(where={"aliasSlug": data.slug})
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

    # Sync OrganizationProfile when name or slug changes
    profile_update: dict = {}
    if data.name is not None:
        profile_update["displayName"] = data.name
    if data.slug is not None:
        profile_update["username"] = data.slug
    if profile_update:
        await prisma.organizationprofile.update(
            where={"organizationId": org_id},
            data=profile_update,
        )

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

    default_ws = await prisma.team.find_first(
        where={"orgId": org_id, "isDefault": True}
    )
    if default_ws:
        await prisma.teammember.create(
            data={
                "teamId": default_ws.id,
                "userId": user_id,
                # Org admins get workspace admin on the default workspace,
                # matching what org creation grants the owner.
                "isAdmin": is_admin,
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

    # Keep the default-workspace admin flag in step with the org role —
    # add_org_member grants it at add time, so promotion/demotion via PATCH
    # must sync it too or promoted admins stay denied on team-scoped checks.
    if is_admin is not None:
        default_ws = await prisma.team.find_first(
            where={"orgId": org_id, "isDefault": True}
        )
        if default_ws:
            await prisma.teammember.update_many(
                where={"teamId": default_ws.id, "userId": user_id},
                data={"isAdmin": is_admin},
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
    workspaces = await prisma.team.find_many(where={"orgId": org_id})
    for ws in workspaces:
        await prisma.teammember.delete_many(where={"teamId": ws.id, "userId": user_id})

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
