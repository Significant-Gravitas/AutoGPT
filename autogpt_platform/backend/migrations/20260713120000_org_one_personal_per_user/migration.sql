-- Enforce at most one live personal org per user.
--
-- The startup backfill (create_orgs_for_existing_users) held the Redis
-- bootstrap lock across its whole per-user loop with no mid-loop renewal.
-- When a wave exceeded the 300s lock TTL, a second pod acquired the lock and
-- re-ran from a fresh snapshot, creating a *second* personal org for every
-- user the first pod had already processed. Nothing at the storage layer
-- prevented it (Organization only has slug @unique, and _resolve_unique_slug
-- side-stepped that with -N suffixes). This migration closes the gap: it
-- de-dupes any orgs the incident already created, then adds a partial unique
-- index so a future double-run collides cleanly instead of duplicating.
--
-- Both steps are set-based and idempotent: on a clean DB (already one personal
-- org per user) the de-dupe UPDATE matches zero rows and the index builds with
-- nothing to reject. Runs inside Prisma's migration transaction, so plain
-- CREATE INDEX (not CONCURRENTLY); the Organization table is small at migrate
-- time in every environment because the backfill runs post-migrate.

-- Step 0: sweep legacy ORPHAN personal orgs — rows carrying a bootstrapUserId
-- with no owner membership (left by pre-transaction partial creates). They are
-- unusable (no member can ever resolve them) but would occupy the unique index
-- slot while staying invisible to the backfill's membership re-check, wedging
-- the user. Idempotent: matches nothing on healthy DBs.
UPDATE "Organization" o
SET "deletedAt" = NOW()
WHERE o."isPersonal" = true
  AND o."deletedAt" IS NULL
  AND o."bootstrapUserId" IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM "OrgMember" m
      WHERE m."orgId" = o."id" AND m."isOwner" = true
  );

-- Step 1: de-duplicate existing personal orgs.
--
-- Winner per bootstrapUserId group = the org that actually accumulated the
-- user's tenanted resources: most AgentGraph rows, then most ChatSession rows,
-- then oldest, then lowest id (final deterministic tiebreak). Losers are
-- soft-deleted (deletedAt = now()); we deliberately do NOT hard-delete or
-- touch balances/ledgers here — money/slug remediation for the affected orgs
-- is handled operationally, not in-migration.
WITH ranked AS (
    SELECT
        o."id" AS org_id,
        ROW_NUMBER() OVER (
            PARTITION BY o."bootstrapUserId"
            ORDER BY
                (
                    SELECT COUNT(*) FROM "AgentGraph" ag
                    WHERE ag."organizationId" = o."id"
                ) DESC,
                (
                    SELECT COUNT(*) FROM "ChatSession" cs
                    WHERE cs."organizationId" = o."id"
                ) DESC,
                o."createdAt" ASC,
                o."id" ASC
        ) AS rn
    FROM "Organization" o
    WHERE o."isPersonal" = true
      AND o."deletedAt" IS NULL
      AND o."bootstrapUserId" IS NOT NULL
)
UPDATE "Organization" o
SET "deletedAt" = NOW()
FROM ranked
WHERE o."id" = ranked.org_id
  AND ranked.rn > 1;

-- Step 2: partial unique index guaranteeing one live personal org per user.
-- Only live personal orgs are constrained; soft-deleted losers (Step 1) and
-- converted team orgs (isPersonal flipped false) are excluded, so the
-- personal->team conversion path — which flips isPersonal=false on the old org
-- BEFORE creating the replacement personal org — never trips it.
CREATE UNIQUE INDEX "Organization_one_personal_per_user"
ON "Organization" ("bootstrapUserId")
WHERE "isPersonal" = true
  AND "deletedAt" IS NULL
  AND "bootstrapUserId" IS NOT NULL;
