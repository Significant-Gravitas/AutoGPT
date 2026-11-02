-- AlterTable
ALTER TABLE "ChatSession" ADD COLUMN     "expertId" TEXT;

-- CreateTable
CREATE TABLE "Expert" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "ownerUserId" TEXT,
    "name" TEXT NOT NULL,
    "avatarUrl" TEXT,
    "role" TEXT NOT NULL,
    "tagline" TEXT,
    "identity" TEXT NOT NULL,
    "toolProfile" JSONB,
    "isTemplate" BOOLEAN NOT NULL DEFAULT false,
    "sourceTemplateId" TEXT,
    "isArchived" BOOLEAN NOT NULL DEFAULT false,
    "organizationId" TEXT,
    "teamId" TEXT,
    "visibility" "ResourceVisibility" NOT NULL DEFAULT 'PRIVATE',

    CONSTRAINT "Expert_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ExpertWorkflow" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expertId" TEXT NOT NULL,
    "storeListingVersionId" TEXT,
    "libraryAgentId" TEXT,

    CONSTRAINT "ExpertWorkflow_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Expert_ownerUserId_isArchived_idx" ON "Expert"("ownerUserId", "isArchived");

-- CreateIndex
CREATE INDEX "Expert_isTemplate_isArchived_idx" ON "Expert"("isTemplate", "isArchived");

-- CreateIndex
CREATE INDEX "Expert_teamId_idx" ON "Expert"("teamId");

-- CreateIndex
CREATE INDEX "Expert_organizationId_idx" ON "Expert"("organizationId");

-- CreateIndex
CREATE UNIQUE INDEX "Expert_ownerUserId_sourceTemplateId_key" ON "Expert"("ownerUserId", "sourceTemplateId");

-- CreateIndex
CREATE INDEX "ExpertWorkflow_expertId_idx" ON "ExpertWorkflow"("expertId");

-- CreateIndex
CREATE UNIQUE INDEX "ExpertWorkflow_expertId_storeListingVersionId_key" ON "ExpertWorkflow"("expertId", "storeListingVersionId");

-- CreateIndex
-- ChatSession is an existing, write-heavy table. Prisma's `migrate deploy`
-- wraps each migration in a single transaction, and Postgres rejects
-- `CREATE INDEX CONCURRENTLY` inside a transaction block; the plain form
-- briefly holds a ShareLock during the scan. For deployments that can't
-- tolerate that, run the CONCURRENTLY equivalent out-of-band before this
-- migration ships and Postgres will skip the recreate via IF NOT EXISTS
-- (same pattern as 20260516120000_add_creator_search_trgm_indexes).
CREATE INDEX IF NOT EXISTS "ChatSession_expertId_idx" ON "ChatSession"("expertId");

-- AddForeignKey
-- NOT VALID keeps the ACCESS EXCLUSIVE window on ChatSession to a catalog
-- update only — no full-table validation scan under lock. Validation runs
-- in the follow-up migration (separate transaction, SHARE UPDATE EXCLUSIVE
-- only), where it is trivially fast since every existing row has a NULL
-- expertId.
ALTER TABLE "ChatSession" ADD CONSTRAINT "ChatSession_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE NOT VALID;

-- AddForeignKey
ALTER TABLE "Expert" ADD CONSTRAINT "Expert_ownerUserId_fkey" FOREIGN KEY ("ownerUserId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Expert" ADD CONSTRAINT "Expert_sourceTemplateId_fkey" FOREIGN KEY ("sourceTemplateId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Expert" ADD CONSTRAINT "Expert_teamId_fkey" FOREIGN KEY ("teamId") REFERENCES "Team"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExpertWorkflow" ADD CONSTRAINT "ExpertWorkflow_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExpertWorkflow" ADD CONSTRAINT "ExpertWorkflow_storeListingVersionId_fkey" FOREIGN KEY ("storeListingVersionId") REFERENCES "StoreListingVersion"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ExpertWorkflow" ADD CONSTRAINT "ExpertWorkflow_libraryAgentId_fkey" FOREIGN KEY ("libraryAgentId") REFERENCES "LibraryAgent"("id") ON DELETE SET NULL ON UPDATE CASCADE;
