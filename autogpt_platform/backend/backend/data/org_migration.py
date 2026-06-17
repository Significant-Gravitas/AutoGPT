"""
Data migration: Bootstrap personal organizations for existing users.

Creates one Organization per user, with owner membership, default workspace,
org profile, seat assignment, and org balance. Assigns all tenant-bound
resources to the user's default workspace. Idempotent — safe to run repeatedly.

Run automatically during server startup via rest_api.py lifespan.
"""

import logging
import re
import time
from typing import LiteralString

from backend.data.db import prisma

logger = logging.getLogger(__name__)


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
            org_data["topUpConfig"] = row["topUpConfig"]

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
            profile_data["socialLinks"] = row["profile_links"]

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
        """
    )
    logger.info(f"Org migration: migrated {result} credit transactions")
    return result


async def _assign_team_tenancy(table_sql: "LiteralString") -> int:
    """Assign organizationId + teamId on a single table's unassigned rows."""
    return await prisma.execute_raw(table_sql)


async def assign_resources_to_teams() -> dict[str, int]:
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
        """
    )

    results["AgentGraphExecution"] = await _assign_team_tenancy(
        """
        UPDATE "AgentGraphExecution" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["ChatSession"] = await _assign_team_tenancy(
        """
        UPDATE "ChatSession" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["AgentPreset"] = await _assign_team_tenancy(
        """
        UPDATE "AgentPreset" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["LibraryAgent"] = await _assign_team_tenancy(
        """
        UPDATE "LibraryAgent" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["LibraryFolder"] = await _assign_team_tenancy(
        """
        UPDATE "LibraryFolder" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["IntegrationWebhook"] = await _assign_team_tenancy(
        """
        UPDATE "IntegrationWebhook" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["APIKey"] = await _assign_team_tenancy(
        """
        UPDATE "APIKey" t
        SET "organizationId" = o."id", "teamId" = w."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        JOIN "Team" w ON w."orgId" = o."id" AND w."isDefault" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    # --- Tables needing only organizationId ---

    results["BuilderSearchHistory"] = await prisma.execute_raw(
        """
        UPDATE "BuilderSearchHistory" t
        SET "organizationId" = o."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["PendingHumanReview"] = await prisma.execute_raw(
        """
        UPDATE "PendingHumanReview" t
        SET "organizationId" = o."id"
        FROM "OrgMember" om
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE t."userId" = om."userId" AND om."isOwner" = true AND t."organizationId" IS NULL
        """
    )

    results["StoreListingVersion"] = await prisma.execute_raw(
        """
        UPDATE "StoreListingVersion" slv
        SET "organizationId" = o."id"
        FROM "StoreListingVersion" v
        JOIN "StoreListing" sl ON sl."id" = v."storeListingId"
        JOIN "OrgMember" om ON om."userId" = sl."owningUserId" AND om."isOwner" = true
        JOIN "Organization" o ON o."id" = om."orgId" AND o."isPersonal" = true
        WHERE slv."id" = v."id" AND slv."organizationId" IS NULL
        """
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
    result = await prisma.execute_raw(
        """
        INSERT INTO "OrganizationAlias"
            ("id", "organizationId", "aliasSlug", "aliasType", "createdByUserId", "isRemovable")
        SELECT
            gen_random_uuid(),
            o."id",
            p."username",
            'MIGRATION',
            o."bootstrapUserId",
            false
        FROM "StoreListing" sl
        JOIN "Organization" o ON o."id" = sl."owningOrgId"
        JOIN "Profile" p ON p."userId" = sl."owningUserId"
        WHERE sl."owningOrgId" IS NOT NULL
          AND sl."hasApprovedVersion" = true
          AND o."slug" != p."username"
          AND NOT EXISTS (
              SELECT 1 FROM "OrganizationAlias" oa
              WHERE oa."aliasSlug" = p."username"
          )
        """
    )
    if result > 0:
        logger.info(f"Org migration: created {result} store listing aliases")
    return result


async def run_migration() -> None:
    """Orchestrate the full org bootstrap migration. Idempotent.

    Uses a Redis-based distributed lock so that only one pod runs the
    migration concurrently in multi-pod deployments.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    lock_key = "org-migration-bootstrap-lock"
    lock_timeout = 300  # 5 min max

    # SET NX EX — acquire only if no other pod holds it
    acquired = await redis.set(lock_key, "1", nx=True, ex=lock_timeout)
    if not acquired:
        logger.info("Org migration: another pod holds the lock, skipping")
        return

    try:
        start = time.monotonic()
        logger.info("Org migration: starting personal org bootstrap")

        orgs_created = await create_orgs_for_existing_users()
        await migrate_org_balances()
        await migrate_credit_transactions()
        resource_counts = await assign_resources_to_teams()
        await migrate_store_listings()
        await create_store_listing_aliases()

        total_resources = sum(resource_counts.values())
        elapsed = time.monotonic() - start

        logger.info(
            f"Org migration: complete in {elapsed:.2f}s — "
            f"{orgs_created} orgs created, {total_resources} resources assigned"
        )
    finally:
        try:
            await redis.delete(lock_key)
        except Exception:
            pass
