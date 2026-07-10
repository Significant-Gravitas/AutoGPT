"""
Data migration: Bootstrap personal organizations for existing users.

Creates one Organization per user, with owner membership, default workspace,
org profile, seat assignment, and org balance. Assigns all tenant-bound
resources to the user's default workspace. Idempotent — safe to run repeatedly.

Run automatically during server startup via rest_api.py lifespan.

This module also owns ``ensure_personal_org`` / ``create_personal_org``, the
single-user bootstrap used at sign-up (``data.user.get_or_create_user``) and by
org conversion (``api.features.orgs.db``) so new accounts get a personal org the
same moment the User row is created — not only when the backfill runs.
"""

import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, LiteralString
from uuid import uuid4

from prisma.errors import UniqueViolationError
from prisma.models import Organization

from backend.data.db import prisma, transaction
from backend.util.json import SafeJson

if TYPE_CHECKING:
    from backend.data.redis_client import AsyncRedisClient

logger = logging.getLogger(__name__)

_MIGRATION_LOCK_RENEW_SCRIPT: LiteralString = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
end
return 0
"""

_MIGRATION_LOCK_RELEASE_SCRIPT: LiteralString = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
end
return 0
"""

_RenewLock = Callable[[], Awaitable[None]]


def _sanitize_slug(raw: str) -> str:
    """Convert a string to a URL-safe slug: lowercase, alphanumeric + hyphens."""
    slug = re.sub(r"[^a-z0-9-]", "-", raw.lower().strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "user"


async def _resolve_unique_slug(desired: str) -> str:
    """Return *desired* if no Organization uses it yet, else append a numeric suffix."""
    existing = await prisma.organization.find_unique(where={"slug": desired})
    if existing is None:
        # Also check aliases
        alias = await prisma.organizationalias.find_unique(where={"aliasSlug": desired})
        if alias is None:
            return desired

    # Collision — find the next available numeric suffix
    for i in range(1, 10_000):
        candidate = f"{desired}-{i}"
        org = await prisma.organization.find_unique(where={"slug": candidate})
        alias = await prisma.organizationalias.find_unique(
            where={"aliasSlug": candidate}
        )
        if org is None and alias is None:
            return candidate

    raise RuntimeError(
        f"Could not resolve a unique slug for '{desired}' after 10000 attempts"
    )


# ---------------------------------------------------------------------------
# Single-user personal-org bootstrap (sign-up + conversion)
# ---------------------------------------------------------------------------


async def create_personal_org(
    user_id: str,
    slug_base: str,
    display_name: str,
) -> Organization:
    """Create a personal Organization for *user_id* with all baseline records.

    Creates, in a single transaction: the Organization (``isPersonal``), an
    owner ``OrgMember`` (owner + admin, ACTIVE), a default ``Team`` with an
    owner ``TeamMember``, an ``OrganizationProfile``, a FREE seat assignment,
    and a zero ``OrgBalance`` row. Shared by the sign-up bootstrap and by org
    conversion so every path produces the exact same record shape the backfill
    (``create_orgs_for_existing_users``) creates.

    Wrapping everything in one transaction is what makes the sign-up race safe:
    two concurrent first-requests for the same user resolve to the same
    deterministic slug and collide on ``Organization.slug``'s unique
    constraint. Because Postgres blocks the loser's insert until the winner
    commits, by the time the loser sees ``UniqueViolationError`` the winner's
    org *and* member (same tx) are already visible.

    Raises:
        UniqueViolationError: if the resolved slug was taken concurrently. The
            caller decides whether that means "already bootstrapped" or "retry".
    """
    slug = await _resolve_unique_slug(slug_base)

    async with transaction() as tx:
        org = await tx.organization.create(
            data={
                "name": display_name,
                "slug": slug,
                "isPersonal": True,
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
                "displayName": display_name,
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

        # Zero-balance row so credit operations don't need an upsert.
        await tx.orgbalance.create(data={"orgId": org.id, "balance": 0})

    return org


async def _derive_personal_org_identity(user_id: str) -> tuple[str, str]:
    """Return ``(slug_base, display_name)`` for *user_id*'s personal org.

    Mirrors the backfill's precedence so app- and migration-created orgs look
    consistent: slug from Profile.username → User.name → email local-part →
    ``user-{id[:8]}``; display name from Profile.name → User.name → local-part.
    """
    user = await prisma.user.find_unique(where={"id": user_id})
    profile = await prisma.profile.find_unique(where={"userId": user_id})

    email = (user.email if user else "") or ""
    user_name = user.name if user else None
    profile_username = profile.username if profile else None
    profile_name = profile.name if profile else None
    local_part = email.split("@")[0] if email else ""

    if profile_username:
        slug_base = _sanitize_slug(profile_username)
    elif user_name:
        slug_base = _sanitize_slug(user_name)
    elif local_part:
        slug_base = _sanitize_slug(local_part)
    else:
        slug_base = f"user-{user_id[:8]}"

    display_name = profile_name or user_name or local_part or "user"
    return slug_base, display_name


async def _find_owned_personal_org(user_id: str):
    """Return *user_id*'s owned, non-deleted personal-org membership, or None."""
    return await prisma.orgmember.find_first(
        where={
            "userId": user_id,
            "isOwner": True,
            "Org": {"isPersonal": True, "deletedAt": None},
        },
    )


async def ensure_personal_org(user_id: str) -> None:
    """Ensure *user_id* owns a personal Organization; create one if missing.

    Idempotent and race-safe. If the user already owns a personal org this is a
    no-op. Otherwise it bootstraps one via ``create_personal_org``.

    On ``UniqueViolationError`` there are two possibilities, distinguished by a
    re-check (same pattern as ``data.user._ensure_user_profile``): a concurrent
    first-request already bootstrapped this user (done), or the resolved slug
    collided with a *different* user's org (loop and retry — ``_resolve_unique_slug``
    now sees the taken slug and picks a fresh suffix).

    Unlike the best-effort marketplace Profile, this must NOT be swallowed: a
    user with no org cannot use any org-scoped endpoint, so a persistent failure
    propagates and fails the request loudly rather than bricking the account.
    """
    if await _find_owned_personal_org(user_id) is not None:
        return

    slug_base, display_name = await _derive_personal_org_identity(user_id)

    for _ in range(3):
        try:
            await create_personal_org(user_id, slug_base, display_name)
            return
        except UniqueViolationError:
            existing = await _find_owned_personal_org(user_id)
            if existing is not None:
                # A concurrent request bootstrapped this user first.
                logger.debug("Personal org for user %s created concurrently", user_id)
                return
            # Slug collided with a *different* user — retry with a fresh suffix.

    raise RuntimeError(
        f"Failed to bootstrap a personal organization for user {user_id} after "
        "retries — the account has no usable organization context."
    )


async def create_orgs_for_existing_users() -> int:
    """Create a personal Organization for every user that lacks one.

    Returns the number of orgs created.
    """
    # Find users who are NOT yet an owner of any personal org
    users_without_org = await prisma.query_raw(
        """
        SELECT u."id", u."email", u."name", u."stripeCustomerId", u."topUpConfig",
               p."username" AS profile_username, p."name" AS profile_name,
               p."description" AS profile_description,
               p."avatarUrl" AS profile_avatar_url,
               p."links" AS profile_links
        FROM "User" u
        LEFT JOIN "Profile" p ON p."userId" = u."id"
        WHERE NOT EXISTS (
            SELECT 1 FROM "OrgMember" om
            JOIN "Organization" o ON o."id" = om."orgId"
            WHERE om."userId" = u."id" AND om."isOwner" = true AND o."isPersonal" = true
        )
        """,
    )

    if not users_without_org:
        logger.info("Org migration: all users already have personal orgs")
        return 0

    logger.info(
        f"Org migration: creating personal orgs for {len(users_without_org)} users"
    )

    created = 0
    for row in users_without_org:
        user_id: str = row["id"]
        email: str = row["email"]
        profile_username: str | None = row.get("profile_username")
        profile_name: str | None = row.get("profile_name")
        user_name: str | None = row.get("name")

        # Determine slug: Profile.username → sanitized User.name → email local part → user-{id[:8]}
        if profile_username:
            desired_slug = _sanitize_slug(profile_username)
        elif user_name:
            desired_slug = _sanitize_slug(user_name)
        else:
            local_part = email.split("@")[0] if email else ""
            desired_slug = (
                _sanitize_slug(local_part) if local_part else f"user-{user_id[:8]}"
            )

        slug = await _resolve_unique_slug(desired_slug)

        display_name = profile_name or user_name or email.split("@")[0]

        # Create Organization — only include optional JSON fields when non-None
        org_data: dict = {
            "name": display_name,
            "slug": slug,
            "isPersonal": True,
            "bootstrapUserId": user_id,
            "settings": "{}",
        }
        if row.get("stripeCustomerId"):
            org_data["stripeCustomerId"] = row["stripeCustomerId"]
        if row.get("topUpConfig"):
            # query_raw returns Json columns as parsed Python objects; prisma
            # create() requires them re-wrapped as Json, not raw dict/list.
            org_data["topUpConfig"] = SafeJson(row["topUpConfig"])

        org = await prisma.organization.create(data=org_data)

        # Create OrgMember (owner)
        await prisma.orgmember.create(
            data={
                "Org": {"connect": {"id": org.id}},
                "User": {"connect": {"id": user_id}},
                "isOwner": True,
                "isAdmin": True,
                "status": "ACTIVE",
            }
        )

        # Create default Team
        workspace = await prisma.team.create(
            data={
                "name": "Default",
                "Org": {"connect": {"id": org.id}},
                "isDefault": True,
                "joinPolicy": "OPEN",
                "createdByUserId": user_id,
            }
        )

        # Create TeamMember
        await prisma.teammember.create(
            data={
                "Team": {"connect": {"id": workspace.id}},
                "User": {"connect": {"id": user_id}},
                "isAdmin": True,
                "status": "ACTIVE",
            }
        )

        # Create OrganizationProfile (from user's Profile if exists)
        profile_data: dict = {
            "Organization": {"connect": {"id": org.id}},
            "username": slug,
            "displayName": display_name,
        }
        if row.get("profile_avatar_url"):
            profile_data["avatarUrl"] = row["profile_avatar_url"]
        if row.get("profile_description"):
            profile_data["bio"] = row["profile_description"]
        if row.get("profile_links"):
            # Profile.links is a Json column; query_raw hands it back as a
            # parsed Python object, so re-wrap for prisma create(). Without
            # this, any user with a populated store profile crashes the
            # whole startup migration.
            profile_data["socialLinks"] = SafeJson(row["profile_links"])

        await prisma.organizationprofile.create(data=profile_data)

        # Create seat assignment (FREE seat for personal org)
        await prisma.organizationseatassignment.create(
            data={
                "organizationId": org.id,
                "userId": user_id,
                "seatType": "FREE",
                "status": "ACTIVE",
                "assignedByUserId": user_id,
            }
        )

        # Log if slug diverged from desired (collision resolution)
        if slug != desired_slug:
            logger.info(
                f"Org migration: slug collision for user {user_id} — "
                f"desired '{desired_slug}', assigned '{slug}'"
            )

        created += 1

    logger.info(f"Org migration: created {created} personal orgs")
    return created


async def migrate_org_balances() -> int:
    """Copy UserBalance rows into OrgBalance for personal orgs that lack one.

    Returns the number of balances migrated.
    """
    result = await prisma.execute_raw(
        """
        INSERT INTO "OrgBalance" ("orgId", "balance", "updatedAt")
        SELECT o."id", ub."balance", ub."updatedAt"
        FROM "UserBalance" ub
        JOIN "OrgMember" om ON om."userId" = ub."userId" AND om."isOwner" = true
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE NOT EXISTS (
            SELECT 1 FROM "OrgBalance" ob WHERE ob."orgId" = o."id"
        )
        ON CONFLICT ("orgId") DO NOTHING
        """
    )
    logger.info(f"Org migration: migrated {result} org balances")
    return result


async def migrate_credit_transactions() -> int:
    """Copy CreditTransaction rows into OrgCreditTransaction for personal orgs.

    Only copies transactions that haven't been migrated yet (by checking for
    matching transactionKey + orgId).

    Returns the number of transactions migrated.
    """
    result = await prisma.execute_raw(
        """
        INSERT INTO "OrgCreditTransaction"
            ("transactionKey", "createdAt", "orgId", "initiatedByUserId",
             "amount", "type", "runningBalance", "isActive", "metadata")
        SELECT
            ct."transactionKey", ct."createdAt", o."id", ct."userId",
            ct."amount", ct."type", ct."runningBalance", ct."isActive", ct."metadata"
        FROM "CreditTransaction" ct
        JOIN "OrgMember" om ON om."userId" = ct."userId" AND om."isOwner" = true
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE NOT EXISTS (
            SELECT 1 FROM "OrgCreditTransaction" oct
            WHERE oct."transactionKey" = ct."transactionKey" AND oct."orgId" = o."id"
        )
        ON CONFLICT ("transactionKey", "orgId") DO NOTHING
        """
    )
    logger.info(f"Org migration: migrated {result} credit transactions")
    return result


async def _assign_team_tenancy(
    table_sql: "LiteralString", renew_lock: _RenewLock | None = None
) -> int:
    """Assign organizationId + teamId on a single table's unassigned rows."""
    updated = await prisma.execute_raw(table_sql)
    if renew_lock is not None:
        await renew_lock()
    return updated


async def _assign_org_tenancy(
    table_sql: "LiteralString", renew_lock: _RenewLock | None = None
) -> int:
    """Assign organizationId on a single table's unassigned rows."""
    updated = await prisma.execute_raw(table_sql)
    if renew_lock is not None:
        await renew_lock()
    return updated


async def _assign_team_tenancy_batched(
    table_name: str,
    batch_sql: "LiteralString",
    renew_lock: _RenewLock | None = None,
) -> int:
    """Run a LIMIT-batched tenancy UPDATE until no assignable rows remain.

    Large tables (executions, chat sessions) cannot be updated in one
    statement: a full-table UPDATE exceeds the platform's statement timeout
    (Supabase kills it at 2 minutes — this is what failed the dev deploy).
    ``batch_sql`` must limit itself via a subquery that only selects rows
    its own JOIN will match, so every selected row updates and the loop is
    guaranteed to terminate — rows whose user has no personal org are never
    selected and can't spin it. Each pass is idempotent
    (``organizationId IS NULL`` filter), so an interrupted run resumes
    cleanly on the next startup.
    """
    total = 0
    while True:
        updated = await prisma.execute_raw(batch_sql)
        if renew_lock is not None:
            await renew_lock()
        if updated == 0:
            return total
        total += updated
        logger.info(
            f"Org migration: assigned {total} {table_name} rows so far (batched)"
        )


async def assign_resources_to_teams(
    renew_lock: _RenewLock | None = None,
) -> dict[str, int]:
    """Set organizationId and teamId on all tenant-bound rows that lack them.

    Uses the user's personal org and its default workspace.

    Returns a dict of table_name -> rows_updated.
    """
    results: dict[str, int] = {}

    # --- Tables needing both organizationId + teamId ---

    results["AgentGraph"] = await _assign_team_tenancy(
        """
        UPDATE "AgentGraph" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["AgentGraphExecution"] = await _assign_team_tenancy_batched(
        "AgentGraphExecution",
        """
        UPDATE "AgentGraphExecution" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true
          AND t."id" IN (
              SELECT t2."id"
              FROM "AgentGraphExecution" t2
              JOIN "OrgMember" om2
                ON om2."userId" = t2."userId" AND om2."isOwner" = true
              JOIN "Organization" o2
                ON o2."id" = om2."orgId" AND o2."isPersonal" = true
              JOIN "Team" w2
                ON w2."orgId" = o2."id" AND w2."isDefault" = true
              WHERE t2."organizationId" IS NULL
              LIMIT 10000
          )
        """,
        renew_lock=renew_lock,
    )

    results["ChatSession"] = await _assign_team_tenancy_batched(
        "ChatSession",
        """
        UPDATE "ChatSession" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true
          AND t."id" IN (
              SELECT t2."id"
              FROM "ChatSession" t2
              JOIN "OrgMember" om2
                ON om2."userId" = t2."userId" AND om2."isOwner" = true
              JOIN "Organization" o2
                ON o2."id" = om2."orgId" AND o2."isPersonal" = true
              JOIN "Team" w2
                ON w2."orgId" = o2."id" AND w2."isDefault" = true
              WHERE t2."organizationId" IS NULL
              LIMIT 10000
          )
        """,
        renew_lock=renew_lock,
    )

    results["AgentPreset"] = await _assign_team_tenancy(
        """
        UPDATE "AgentPreset" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["LibraryAgent"] = await _assign_team_tenancy(
        """
        UPDATE "LibraryAgent" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["LibraryFolder"] = await _assign_team_tenancy(
        """
        UPDATE "LibraryFolder" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["IntegrationWebhook"] = await _assign_team_tenancy(
        """
        UPDATE "IntegrationWebhook" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["APIKey"] = await _assign_team_tenancy(
        """
        UPDATE "APIKey" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["UserNotificationBatch"] = await _assign_team_tenancy(
        """
        UPDATE "UserNotificationBatch" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    # --- Tables needing only organizationId ---

    results["BuilderSearchHistory"] = await _assign_org_tenancy(
        """
        UPDATE "BuilderSearchHistory" t
        SET "organizationId" = o."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["PendingHumanReview"] = await _assign_org_tenancy(
        """
        UPDATE "PendingHumanReview" t
        SET "organizationId" = o."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    results["StoreListingVersion"] = await _assign_org_tenancy(
        """
        UPDATE "StoreListingVersion" slv
        SET "organizationId" = o."id"
        FROM "StoreListingVersion" v
        JOIN "StoreListing" sl ON sl."id" = v."storeListingId"
        JOIN "OrgMember" om ON om."userId" = sl."owningUserId" AND om."isOwner" = true
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE slv."id" = v."id" AND slv."organizationId" IS NULL
        """,
        renew_lock=renew_lock,
    )

    for table_name, count in results.items():
        if count > 0:
            logger.info(f"Org migration: assigned {count} {table_name} rows")

    return results


async def migrate_store_listings() -> int:
    """Set owningOrgId on StoreListings that lack it.

    Returns the number of listings migrated.
    """
    result = await prisma.execute_raw(
        """
        UPDATE "StoreListing" sl
        SET "owningOrgId" = o."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE sl."owningUserId" = om."userId"
          AND om."isOwner" = true
          AND sl."owningOrgId" IS NULL
        """
    )
    if result > 0:
        logger.info(f"Org migration: assigned {result} store listings to orgs")
    return result


async def create_store_listing_aliases() -> int:
    """Create OrganizationAlias entries for published store listings.

    This ensures that marketplace URLs using the org slug continue to work.
    Only creates aliases for listings whose org slug matches the user's Profile
    username (which it should for personal orgs created from Profile.username).

    Returns the number of aliases created.
    """
    # DISTINCT ON dedupes within the statement (a user with several approved
    # listings yields the same alias N times — NOT EXISTS only checks the
    # table, not this statement's own rows), and ON CONFLICT covers the race
    # when several services run the backfill concurrently at startup. Either
    # duplicate previously raised a unique violation that crashed the whole
    # lifespan on any boot after listings existed.
    result = await prisma.execute_raw(
        """
        INSERT INTO "OrganizationAlias"
            ("id", "organizationId", "aliasSlug", "aliasType", "createdByUserId", "isRemovable")
        SELECT
            gen_random_uuid(),
            sub."orgId",
            sub."username",
            'MIGRATION',
            sub."bootstrapUserId",
            false
        FROM (
            SELECT DISTINCT ON (p."username")
                o."id" AS "orgId",
                p."username" AS "username",
                o."bootstrapUserId" AS "bootstrapUserId"
            FROM "StoreListing" sl
            JOIN "Organization" o ON o."id" = sl."owningOrgId"
            JOIN "Profile" p ON p."userId" = sl."owningUserId"
            WHERE sl."owningOrgId" IS NOT NULL
              AND sl."hasApprovedVersion" = true
              AND o."slug" != p."username"
        ) sub
        WHERE NOT EXISTS (
            SELECT 1 FROM "OrganizationAlias" oa
            WHERE oa."aliasSlug" = sub."username"
        )
        ON CONFLICT ("aliasSlug") DO NOTHING
        """
    )
    if result > 0:
        logger.info(f"Org migration: created {result} store listing aliases")
    return result


async def migrate_credentials_to_table() -> int:
    """Copy each user's UserIntegrations blob credentials into
    IntegrationCredential rows (ownerType=USER, personal org).

    Big-bang per launch decision: the table becomes the org-aware source
    of truth; the encrypted blob is left untouched as the rollback
    artifact until prod-verified. Row ids reuse the credential's existing
    UUID so graph ``credentials_id`` references stay valid. Idempotent —
    ids already present in the table are skipped, so re-runs (every
    startup) only pick up blob credentials not yet copied.

    Returns the number of credential rows created.
    """
    from backend.data.model import UserIntegrations
    from backend.util.encryption import JSONCryptor

    cryptor = JSONCryptor()
    created = 0

    users = await prisma.user.find_many(where={"integrations": {"not": ""}})
    for user in users:
        try:
            integrations = UserIntegrations.model_validate(
                cryptor.decrypt(user.integrations)
            )
        except Exception:
            logger.error(
                f"Credential migration: cannot decrypt blob for user {user.id}; "
                "skipping (blob left untouched)",
                exc_info=True,
            )
            continue

        if not integrations.credentials:
            continue

        org_row = await prisma.organization.find_first(
            where={
                "isPersonal": True,
                "Members": {"some": {"userId": user.id, "isOwner": True}},
            }
        )
        if org_row is None:
            # Personal org not bootstrapped yet — the next startup sweep
            # (after create_orgs_for_existing_users) will pick this up.
            logger.warning(
                f"Credential migration: no personal org for user {user.id}; deferring"
            )
            continue

        existing_rows = await prisma.integrationcredential.find_many(
            where={"id": {"in": [c.id for c in integrations.credentials]}}
        )
        existing_ids = {row.id for row in existing_rows}

        for cred in integrations.credentials:
            if cred.id in existing_ids:
                continue
            await prisma.integrationcredential.create(
                data={
                    "id": cred.id,
                    "organizationId": org_row.id,
                    "ownerType": "USER",
                    "ownerId": user.id,
                    "provider": cred.provider,
                    "credentialType": cred.type,
                    "displayName": cred.title or cred.provider,
                    "encryptedPayload": cryptor.encrypt(cred.model_dump()),
                    "createdByUserId": user.id,
                }
            )
            created += 1

    if created:
        logger.info(f"Credential migration: copied {created} blob credentials to table")
    return created


async def _renew_migration_lock(
    redis: "AsyncRedisClient",
    lock_key: str,
    lock_token: str,
    lock_timeout: int,
) -> None:
    renewed = await redis.execute_command(
        "EVAL",
        _MIGRATION_LOCK_RENEW_SCRIPT,
        1,
        lock_key,
        lock_token,
        str(lock_timeout),
    )
    if renewed != 1:
        raise RuntimeError("Org migration lost the distributed bootstrap lock")


async def _release_migration_lock(
    redis: "AsyncRedisClient",
    lock_key: str,
    lock_token: str,
) -> bool:
    released = await redis.execute_command(
        "EVAL",
        _MIGRATION_LOCK_RELEASE_SCRIPT,
        1,
        lock_key,
        lock_token,
    )
    return released == 1


async def run_migration() -> None:
    """Orchestrate the full org bootstrap migration. Idempotent.

    Uses a Redis-based distributed lock so that only one pod runs the
    migration concurrently in multi-pod deployments.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    lock_key = "org-migration-bootstrap-lock"
    lock_token = str(uuid4())
    lock_timeout = 300  # 5 min max

    # SET NX EX — acquire only if no other pod holds it
    acquired = await redis.set(lock_key, lock_token, nx=True, ex=lock_timeout)
    if not acquired:
        logger.info("Org migration: another pod holds the lock, skipping")
        return

    async def renew_lock() -> None:
        await _renew_migration_lock(redis, lock_key, lock_token, lock_timeout)

    try:
        start = time.monotonic()
        logger.info("Org migration: starting personal org bootstrap")

        orgs_created = await create_orgs_for_existing_users()
        await renew_lock()
        await migrate_org_balances()
        await renew_lock()
        await migrate_credit_transactions()
        await renew_lock()
        resource_counts = await assign_resources_to_teams(renew_lock=renew_lock)
        await renew_lock()
        await migrate_store_listings()
        await renew_lock()
        await create_store_listing_aliases()
        await renew_lock()
        await migrate_credentials_to_table()
        await renew_lock()

        total_resources = sum(resource_counts.values())
        elapsed = time.monotonic() - start

        logger.info(
            f"Org migration: complete in {elapsed:.2f}s — "
            f"{orgs_created} orgs created, {total_resources} resources assigned"
        )
    finally:
        try:
            await _release_migration_lock(redis, lock_key, lock_token)
        except Exception:
            pass
