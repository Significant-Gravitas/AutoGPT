-- AlterTable
ALTER TABLE "Expert" ADD COLUMN "templateKey" TEXT;

-- CreateIndex
CREATE UNIQUE INDEX "Expert_templateKey_key" ON "Expert"("templateKey");

-- AlterTable
ALTER TABLE "SkillListingVersion" ADD COLUMN "packageSha256" TEXT,
ADD COLUMN "skillMarkdown" TEXT;

-- CreateIndex
CREATE UNIQUE INDEX "SkillListingVersion_skillListingId_packageSha256_key" ON "SkillListingVersion"("skillListingId", "packageSha256");

-- CreateTable
CREATE TABLE "SkillCatalogRelease" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "repository" TEXT NOT NULL,
    "revision" TEXT NOT NULL,
    "releaseKey" TEXT NOT NULL,
    "manifestSha256" TEXT NOT NULL,
    "summary" JSONB NOT NULL,

    CONSTRAINT "SkillCatalogRelease_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "SkillCatalogRelease_createdAt_idx" ON "SkillCatalogRelease"("createdAt");

-- Backfill: the roster seed resolved templates by display name, so the key a
-- template gets is its lowercased name. Where a name was seeded twice the
-- oldest row is the one the seed kept adopting, so it takes the key and the
-- rest stay unkeyed (and unmanaged) rather than colliding on the unique index.
UPDATE "Expert" AS e
SET "templateKey" = keyed.key
FROM (
    SELECT DISTINCT ON (lower("name")) "id", lower("name") AS key
    FROM "Expert"
    WHERE "isTemplate"
      AND "ownerUserId" IS NULL
      AND "organizationId" IS NULL
      AND "teamId" IS NULL
    ORDER BY lower("name"), "createdAt", "id"
) AS keyed
WHERE e."id" = keyed."id";
