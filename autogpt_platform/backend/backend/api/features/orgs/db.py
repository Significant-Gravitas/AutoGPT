"""Database operations for organization management."""

import asyncio
import logging
from datetime import datetime, timezone

from autogpt_libs.auth.permissions import OrgAction
from prisma import Json, Prisma
from prisma.errors import UniqueViolationError

from backend.data.db import (
    TRANSACTION_TIMEOUT,
    execute_raw_with_schema,
    prisma,
    transaction,
)
from backend.data.org_migration import (
    _sanitize_slug,
    _soft_delete_blocking_orphan,
    create_personal_org,
)
from backend.data.tenancy import (
    lock_live_org_membership_scope,
    lock_live_org_membership_scopes,
    lock_live_org_permission_scope,
    lock_live_org_scope,
)
from backend.util.exceptions import NotAuthorizedError, NotFoundError

from .memory_model import SharedMemoryOrgAccess, SharedMemoryTeamAccess
from .model import OrgAliasResponse, OrgMemberResponse, OrgResponse, UpdateOrgData

logger = logging.getLogger(__name__)


async def _lock_authorized_org_action(
    client: Prisma,
    org_id: str,
    actor_user_id: str | None,
    action: OrgAction,
    related_user_ids: list[str] | None = None,
) -> None:
    if actor_user_id is None:
        await lock_live_org_scope(client, org_id)
        return
    if (
        await lock_live_org_permission_scope(
            client,
            actor_user_id,
            org_id,
            action,
            related_user_ids,
        )
        is None
    ):
        raise NotAuthorizedError("Organization access was revoked")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _coerce_settings_dict(settings) -> dict:
    """Parse an Organization.settings value into a mutable dict.

    The column can hold a parsed dict or the raw JSON-string form; anything
    unrecognized collapses to an empty dict so a read-modify-write never
    clobbers on malformed input (it just rebuilds from empty).
    """
    if isinstance(settings, str):
        import json

        try:
            parsed = json.loads(settings)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(settings, dict):
        return dict(settings)
    return {}


async def get_shared_memory_org_access(
    org_id: str, user_id: str
) -> SharedMemoryOrgAccess | None:
    """Return active org access without leaking a Prisma model over RPC."""
    member = await prisma.orgmember.find_first(
        where={
            "orgId": org_id,
            "userId": user_id,
            "status": "ACTIVE",
            "Org": {"deletedAt": None},
        }
    )
    if member is None:
        return None
    can_access = bool(member.isOwner or member.isAdmin or not member.isBillingManager)
    return SharedMemoryOrgAccess(
        is_admin=bool(member.isAdmin or member.isOwner),
        can_view=can_access,
        can_write=can_access,
    )


async def list_shared_memory_team_access(
    org_id: str, user_id: str
) -> list[SharedMemoryTeamAccess]:
    """Return the caller's active, non-archived teams for memory routing."""
    memberships = await prisma.teammember.find_many(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "Team": {
                "is": {
                    "orgId": org_id,
                    "archivedAt": None,
                    "Org": {
                        "is": {
                            "deletedAt": None,
                            "Members": {
                                "some": {"userId": user_id, "status": "ACTIVE"}
                            },
                        }
                    },
                }
            },
        },
        include={"Team": True},
    )
    return [
        SharedMemoryTeamAccess(
            team_id=membership.teamId,
            name=membership.Team.name,
            is_admin=membership.isAdmin,
            can_view=bool(membership.isAdmin or not membership.isBillingManager),
            can_write=bool(membership.isAdmin or not membership.isBillingManager),
        )
        for membership in memberships
        if membership.Team is not None
    ]


async def get_shared_memory_hold_buffer(org_id: str) -> bool:
    """Return the review-buffer setting, failing closed when it is unavailable."""
    org = await prisma.organization.find_unique(where={"id": org_id})
    if org is None or org.deletedAt is not None:
        return True
    settings = _coerce_settings_dict(org.settings)
    memory = settings.get("memory")
    if not isinstance(memory, dict):
        return True
    return bool(memory.get("holdBuffer", True))


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


async def resolve_default_tenancy(user_id: str) -> tuple[str | None, str | None]:
    """Best-effort default org/team for tenanting newly created rows.

    Wraps ``get_user_default_team`` so tenancy resolution can never abort the
    operation that needs it (an execution, a library add, a notification): a
    raised lookup — or an unresolvable org — yields ``(None, None)`` and the
    row is created untenanted. Callers stamp the returned pair only when
    non-null.
    """
    try:
        return await get_user_default_team(user_id)
    except Exception:
        logger.warning(
            f"Default org/team lookup failed for user {user_id}; "
            "creating the row untenanted",
            exc_info=True,
        )
        return None, None


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


async def is_org_member(org_id: str, user_id: str) -> bool:
    member = await prisma.orgmember.find_first(
        where={
            "orgId": org_id,
            "userId": user_id,
            "status": "ACTIVE",
            "Org": {"deletedAt": None},
        }
    )
    return member is not None


async def get_org(org_id: str) -> OrgResponse:
    """Get organization details."""
    org = await prisma.organization.find_unique(where={"id": org_id})
    if org is None or org.deletedAt is not None:
        raise NotFoundError(f"Organization {org_id} not found")
    return OrgResponse.from_db(org)


async def update_org(
    org_id: str,
    data: UpdateOrgData,
    actor_user_id: str | None = None,
) -> OrgResponse:
    """Update organization fields. Creates a RENAME alias if slug changes.

    Only accepts the structured UpdateOrgData model — no arbitrary dict keys.
    """
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_authorized_org_action(
            tx, org_id, actor_user_id, OrgAction.RENAME_ORG
        )
        existing = None
        existing_alias = None
        if data.slug is not None:
            existing = await tx.organization.find_unique(where={"slug": data.slug})
            existing_alias = await tx.organizationalias.find_unique(
                where={"aliasSlug": data.slug}
            )

        current = await tx.organization.find_unique(where={"id": org_id})
        if current is None or current.deletedAt is not None:
            raise NotFoundError(f"Organization {org_id} not found")

        update_dict: dict = {}
        if data.name is not None:
            update_dict["name"] = data.name
        if data.description is not None:
            update_dict["description"] = data.description
        if data.avatar_url is not None:
            update_dict["avatarUrl"] = data.avatar_url
        if data.memory_hold_buffer is not None:
            settings = _coerce_settings_dict(current.settings)
            memory = settings.get("memory")
            if not isinstance(memory, dict):
                memory = {}
            memory["holdBuffer"] = data.memory_hold_buffer
            settings["memory"] = memory
            update_dict["settings"] = Json(settings)
        if data.slug is not None:
            if existing and existing.id != org_id:
                raise ValueError(f"Slug '{data.slug}' is already in use")
            if existing_alias and existing_alias.organizationId != org_id:
                raise ValueError(f"Slug '{data.slug}' is already in use as an alias")
            if current.slug != data.slug:
                if existing_alias:
                    await tx.organizationalias.delete(where={"aliasSlug": data.slug})
                await tx.organizationalias.create(
                    data={
                        "organizationId": org_id,
                        "aliasSlug": current.slug,
                        "aliasType": "RENAME",
                    }
                )
            update_dict["slug"] = data.slug

        if update_dict:
            await tx.organization.update(where={"id": org_id}, data=update_dict)

        profile_update: dict = {}
        if data.name is not None:
            profile_update["displayName"] = data.name
        if data.slug is not None:
            profile_update["username"] = data.slug
        if profile_update:
            await tx.organizationprofile.update(
                where={"organizationId": org_id}, data=profile_update
            )

    return await get_org(org_id)


async def delete_org(org_id: str, actor_user_id: str | None = None) -> None:
    """Soft-delete an organization. Cannot delete personal orgs.

    Sets deletedAt instead of hard-deleting to preserve financial records.
    """
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_authorized_org_action(
            tx, org_id, actor_user_id, OrgAction.DELETE_ORG
        )
        await execute_raw_with_schema(
            'UPDATE {schema_prefix}"Organization" '
            'SET "updatedAt" = "updatedAt" WHERE "id" = $1',
            org_id,
            client=tx,
        )
        org = await tx.organization.find_unique(where={"id": org_id})
        if org is None:
            raise NotFoundError(f"Organization {org_id} not found")
        if org.isPersonal:
            raise ValueError("Cannot delete a personal organization. Convert it first.")
        if org.deletedAt is not None:
            raise ValueError("Organization is already deleted")

        await tx.organization.update(
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
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_authorized_org_action(tx, org_id, user_id, OrgAction.DELETE_ORG)
        org = await tx.organization.find_unique(where={"id": org_id})
        if org is None or org.deletedAt is not None:
            raise NotFoundError(f"Organization {org_id} not found")
        if not org.isPersonal:
            raise ValueError("Organization is already a team org")
        await tx.organization.update(
            where={"id": org_id},
            data={"isPersonal": False},
        )

    try:
        slug_base = f"{_sanitize_slug(org.slug)}-personal-1"
        user = await prisma.user.find_unique(where={"id": user_id})
        display_name = user.name if user and user.name else org.name

        await _create_personal_org_for_user(
            user_id=user_id,
            slug_base=slug_base,
            display_name=display_name,
        )
    except Exception:
        logger.exception(
            f"Failed to create new personal org for user {user_id} during "
            f"conversion of org {org_id} — rolling back"
        )
        async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
            await lock_live_org_scope(tx, org_id)
            await tx.organization.update(
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


async def lock_org_membership(client: Prisma, org_id: str, user_id: str) -> None:
    await lock_live_org_membership_scope(client, org_id, user_id)
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"User" SET "updatedAt" = "updatedAt" ' 'WHERE "id" = $1',
        user_id,
        client=client,
    )
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"Organization" '
        'SET "updatedAt" = "updatedAt" WHERE "id" = $1',
        org_id,
        client=client,
    )


async def assert_not_last_team_admin(client: Prisma, org_id: str, user_id: str) -> None:
    memberships = await client.teammember.find_many(
        where={
            "userId": user_id,
            "isAdmin": True,
            "status": "ACTIVE",
            "Team": {"is": {"orgId": org_id, "archivedAt": None}},
        }
    )
    for team_id in sorted(membership.teamId for membership in memberships):
        await execute_raw_with_schema(
            'UPDATE {schema_prefix}"Team" SET "updatedAt" = "updatedAt" '
            'WHERE "id" = $1 AND "orgId" = $2',
            team_id,
            org_id,
            client=client,
        )
        other_admins = await client.teammember.count(
            where={
                "teamId": team_id,
                "userId": {"not": user_id},
                "isAdmin": True,
                "status": "ACTIVE",
            }
        )
        if other_admins == 0:
            raise ValueError(
                "Cannot remove this member while they are the last admin of a "
                "workspace. Promote another workspace member first."
            )


async def add_org_member(
    org_id: str,
    user_id: str,
    is_admin: bool = False,
    is_billing_manager: bool = False,
    invited_by: str | None = None,
) -> OrgMemberResponse:
    """Add a member to an organization and its default workspace."""
    async with transaction() as tx:
        await _lock_authorized_org_action(
            tx,
            org_id,
            invited_by,
            OrgAction.MANAGE_MEMBERS,
            [user_id],
        )
        await lock_org_membership(tx, org_id, user_id)
        member = await tx.orgmember.create(
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
        await _revoke_pending_member_invitations(org_id, user_id, tx)

        default_ws = await tx.team.find_first(
            where={"orgId": org_id, "isDefault": True}
        )
        if default_ws:
            await tx.teammember.create(
                data={
                    "teamId": default_ws.id,
                    "userId": user_id,
                    "isAdmin": is_admin,
                    "isBillingManager": is_billing_manager,
                    "status": "ACTIVE",
                }
            )

    return OrgMemberResponse.from_db(member)


async def update_org_member(
    org_id: str,
    user_id: str,
    is_admin: bool | None,
    is_billing_manager: bool | None,
    requesting_user_id: str | None = None,
) -> OrgMemberResponse:
    """Update a member's role flags."""
    async with transaction() as tx:
        await _lock_authorized_org_action(
            tx,
            org_id,
            requesting_user_id,
            OrgAction.MANAGE_MEMBERS,
            [user_id],
        )
        await lock_org_membership(tx, org_id, user_id)
        member = await tx.orgmember.find_unique(
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
            await tx.orgmember.update(
                where={"orgId_userId": {"orgId": org_id, "userId": user_id}},
                data=update_data,
            )
            await _revoke_pending_member_invitations(org_id, user_id, tx)

        if is_admin is not None or is_billing_manager is not None:
            default_ws = await tx.team.find_first(
                where={"orgId": org_id, "isDefault": True}
            )
            if default_ws:
                team_update_data = {}
                if is_admin is not None:
                    team_update_data["isAdmin"] = is_admin
                if is_billing_manager is not None:
                    team_update_data["isBillingManager"] = is_billing_manager
                await tx.teammember.update_many(
                    where={"teamId": default_ws.id, "userId": user_id},
                    data=team_update_data,
                )

    members = await list_org_members(org_id)
    match = next((m for m in members if m.user_id == user_id), None)
    if match is None:
        raise NotFoundError(f"Member {user_id} not found in org {org_id} after update")
    return match


async def _revoke_pending_member_invitations(
    org_id: str,
    user_id: str,
    client: Prisma | None = None,
) -> None:
    db = client or prisma
    user = await db.user.find_unique(where={"id": user_id})
    filters: list[dict] = [{"targetUserId": user_id}]
    if user is not None and user.email:
        filters.append({"email": {"equals": user.email, "mode": "insensitive"}})
    await db.orginvitation.update_many(
        where={
            "orgId": org_id,
            "acceptedAt": None,
            "revokedAt": None,
            "OR": filters,
        },
        data={"revokedAt": datetime.now(timezone.utc)},
    )


def _resource_where(org_id: str, user_id: str, team_id: str | None) -> dict:
    where = {"organizationId": org_id, "userId": user_id}
    if team_id is not None:
        where["teamId"] = team_id
    return where


async def assert_no_owned_resources(
    client: Prisma, org_id: str, user_id: str, team_id: str | None = None
) -> None:
    where = _resource_where(org_id, user_id, team_id)
    api_key_where = {**where, "status": "ACTIVE"}
    if team_id is not None:
        api_key_where.pop("teamId")
        api_key_where["OR"] = [
            {"teamId": team_id},
            {"teamIdRestriction": team_id},
        ]
    expert_where = {
        "organizationId": org_id,
        "ownerUserId": user_id,
        "isArchived": False,
    }
    if team_id is not None:
        expert_where["teamId"] = team_id
    counts = [
        await client.agentgraph.count(where=where),
        await client.libraryagent.count(where={**where, "isDeleted": False}),
        await client.libraryfolder.count(where={**where, "isDeleted": False}),
        await client.agentpreset.count(where={**where, "isDeleted": False}),
        await client.apikey.count(where=api_key_where),
        await client.integrationwebhook.count(where=where),
        await client.chatsession.count(where=where),
        await client.agentgraphexecution.count(
            where={
                **where,
                "executionStatus": {
                    "in": ["INCOMPLETE", "QUEUED", "RUNNING", "REVIEW"]
                },
            }
        ),
        await client.expert.count(where=expert_where),
    ]
    if any(counts):
        scope = "workspace" if team_id is not None else "organization"
        raise ValueError(
            f"Cannot remove this member while they own active resources in the {scope}. "
            "Transfer or delete those resources first."
        )


async def assert_no_owned_schedules(
    org_id: str, user_id: str, team_ids: list[str], team_id: str | None = None
) -> None:
    from backend.util.clients import get_scheduler_client

    schedules = await get_scheduler_client().get_execution_schedules(
        user_id=user_id,
        organization_id=org_id,
        team_ids=team_ids,
        include_paused=True,
    )
    schedules = [
        schedule
        for schedule in schedules
        if schedule.organization_id == org_id
        and (team_id is None or schedule.team_id == team_id)
    ]
    if schedules:
        scope = "workspace" if team_id is not None else "organization"
        raise ValueError(
            f"Cannot remove this member while they own schedules in the {scope}. "
            "Delete those schedules first."
        )


async def _start_org_member_removal(
    org_id: str, user_id: str, requesting_user_id: str
) -> None:
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_authorized_org_action(
            tx,
            org_id,
            requesting_user_id,
            OrgAction.MANAGE_MEMBERS,
            [user_id],
        )
        await lock_org_membership(tx, org_id, user_id)
        member = await tx.orgmember.find_unique(
            where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
        )
        if member is None or member.status != "ACTIVE":
            raise NotFoundError(f"Member {user_id} not found in org {org_id}")
        if member.isOwner:
            raise ValueError("Cannot remove the org owner. Transfer ownership first.")
        if user_id == requesting_user_id:
            raise ValueError("Cannot remove yourself from an organization.")
        other_memberships = await tx.orgmember.count(
            where={
                "userId": user_id,
                "status": "ACTIVE",
                "orgId": {"not": org_id},
                "Org": {"is": {"deletedAt": None}},
            }
        )
        if other_memberships == 0:
            raise ValueError(
                "Cannot remove this member — they have no other organization "
                "memberships and would be locked out."
            )
        await assert_not_last_team_admin(tx, org_id, user_id)
        await tx.orgmember.update(
            where={"orgId_userId": {"orgId": org_id, "userId": user_id}},
            data={"status": "SUSPENDED"},
        )


async def _finalize_org_member_removal(
    org_id: str,
    user_id: str,
    requesting_user_id: str,
) -> None:
    user = await prisma.user.find_unique(where={"id": user_id})
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_authorized_org_action(
            tx,
            org_id,
            requesting_user_id,
            OrgAction.MANAGE_MEMBERS,
            [user_id],
        )
        await lock_org_membership(tx, org_id, user_id)
        current = await tx.orgmember.find_unique(
            where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
        )
        if current is None or current.status != "SUSPENDED":
            raise NotFoundError(f"Member {user_id} not found in org {org_id}")
        if current.isOwner:
            raise ValueError("Cannot remove the org owner. Transfer ownership first.")
        await assert_not_last_team_admin(tx, org_id, user_id)
        await assert_no_owned_resources(tx, org_id, user_id)
        invitation_identity: list[dict[str, object]] = [{"targetUserId": user_id}]
        if user is not None and user.email:
            invitation_identity.append(
                {"email": {"equals": user.email, "mode": "insensitive"}}
            )
        await tx.orginvitation.update_many(
            where={
                "orgId": org_id,
                "acceptedAt": None,
                "revokedAt": None,
                "OR": invitation_identity,
            },
            data={"revokedAt": datetime.now(timezone.utc)},
        )
        teams = await tx.team.find_many(where={"orgId": org_id})
        await tx.teammember.delete_many(
            where={"teamId": {"in": [team.id for team in teams]}, "userId": user_id}
        )
        await tx.orgmember.delete(
            where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
        )


async def remove_org_member(org_id: str, user_id: str, requesting_user_id: str) -> None:
    await _start_org_member_removal(org_id, user_id, requesting_user_id)
    try:
        await assert_no_owned_resources(prisma, org_id, user_id)
        memberships = await prisma.teammember.find_many(
            where={
                "userId": user_id,
                "status": "ACTIVE",
                "Team": {"is": {"orgId": org_id}},
            }
        )
        await assert_no_owned_schedules(
            org_id, user_id, [membership.teamId for membership in memberships]
        )
        await _finalize_org_member_removal(org_id, user_id, requesting_user_id)
    except Exception:
        async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
            await lock_live_org_membership_scopes(
                tx, org_id, [requesting_user_id, user_id]
            )
            org = await tx.organization.find_first(
                where={"id": org_id, "deletedAt": None}
            )
            if org is not None:
                await tx.orgmember.update_many(
                    where={
                        "orgId": org_id,
                        "userId": user_id,
                        "status": "SUSPENDED",
                    },
                    data={"status": "ACTIVE"},
                )
        raise


async def transfer_ownership(
    org_id: str, current_owner_id: str, new_owner_id: str
) -> None:
    """Transfer org ownership atomically."""
    if current_owner_id == new_owner_id:
        raise ValueError("Cannot transfer ownership to the same user")

    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_authorized_org_action(
            tx,
            org_id,
            current_owner_id,
            OrgAction.DELETE_ORG,
            [new_owner_id],
        )
        await execute_raw_with_schema(
            'UPDATE {schema_prefix}"Organization" '
            'SET "updatedAt" = "updatedAt" WHERE "id" = $1',
            org_id,
            client=tx,
        )
        current = await tx.orgmember.find_unique(
            where={"orgId_userId": {"orgId": org_id, "userId": current_owner_id}}
        )
        if current is None or not current.isOwner or current.status != "ACTIVE":
            raise ValueError("Current user is not the org owner")

        new = await tx.orgmember.find_unique(
            where={"orgId_userId": {"orgId": org_id, "userId": new_owner_id}}
        )
        if new is None or new.status != "ACTIVE":
            raise NotFoundError(f"User {new_owner_id} is not a member of org {org_id}")

        await execute_raw_with_schema(
            """
            UPDATE {schema_prefix}"OrgMember"
            SET "isOwner" = ("userId" = $1),
                "isAdmin" = CASE
                    WHEN "userId" = $1 THEN true
                    ELSE "isAdmin"
                END,
                "updatedAt" = NOW()
            WHERE "orgId" = $2
              AND ("isOwner" = true OR "userId" = $1)
            """,
            new_owner_id,
            org_id,
            client=tx,
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
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await _lock_authorized_org_action(tx, org_id, user_id, OrgAction.RENAME_ORG)
        org = await tx.organization.find_unique(where={"id": org_id})
        if org is None or org.deletedAt is not None:
            raise NotFoundError(f"Organization {org_id} not found")
        existing_org = await tx.organization.find_unique(where={"slug": alias_slug})
        if existing_org:
            raise ValueError(f"Slug '{alias_slug}' is already used by an organization")
        existing_alias = await tx.organizationalias.find_unique(
            where={"aliasSlug": alias_slug}
        )
        if existing_alias:
            raise ValueError(f"Slug '{alias_slug}' is already used as an alias")
        alias = await tx.organizationalias.create(
            data={
                "organizationId": org_id,
                "aliasSlug": alias_slug,
                "aliasType": "MANUAL",
                "createdByUserId": user_id,
            }
        )
    return OrgAliasResponse.from_db(alias)
