-- Marketplace listings for knowledge skills.
--
-- A skill is instructions and examples, not a runnable graph, so it cannot ride
-- StoreListing: that table's "agentGraphId" is NOT NULL and the StoreAgent view
-- exposes the graph as non-null. These tables mirror its shape — slug, version,
-- submission status, review fields — and hold the SKILL.md instead.
--
-- No view accompanies them. StoreAgent exists to aggregate run counts and
-- review stats from materialized views; a skill has neither, and its install
-- count lives on the listing row.

-- CreateTable
CREATE TABLE "SkillListing" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "hasApprovedVersion" BOOLEAN NOT NULL DEFAULT false,
    "slug" TEXT NOT NULL,
    "activeVersionId" TEXT,
    "installCount" INTEGER NOT NULL DEFAULT 0,
    "owningUserId" TEXT,
    "owningOrgId" TEXT,

    CONSTRAINT "SkillListing_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SkillListingVersion" (
    "id" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "body" TEXT NOT NULL,
    "triggers" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "categories" TEXT[],
    "requiredProviders" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "sourceSkillSlug" TEXT,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,
    "isAvailable" BOOLEAN NOT NULL DEFAULT true,
    "submissionStatus" "SubmissionStatus" NOT NULL DEFAULT 'DRAFT',
    "submittedAt" TIMESTAMP(3),
    "skillListingId" TEXT NOT NULL,
    "changesSummary" TEXT,
    "reviewerId" TEXT,
    "internalComments" TEXT,
    "reviewComments" TEXT,
    "reviewedAt" TIMESTAMP(3),
    "organizationId" TEXT,

    CONSTRAINT "SkillListingVersion_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "SkillListing_activeVersionId_key" ON "SkillListing"("activeVersionId");

-- CreateIndex
CREATE UNIQUE INDEX "SkillListing_slug_key" ON "SkillListing"("slug");

-- CreateIndex
CREATE INDEX "SkillListing_owningUserId_idx" ON "SkillListing"("owningUserId");

-- CreateIndex
CREATE INDEX "SkillListing_isDeleted_hasApprovedVersion_idx" ON "SkillListing"("isDeleted", "hasApprovedVersion");

-- CreateIndex
CREATE INDEX "SkillListing_owningOrgId_idx" ON "SkillListing"("owningOrgId");

-- CreateIndex
CREATE UNIQUE INDEX "SkillListingVersion_skillListingId_version_key" ON "SkillListingVersion"("skillListingId", "version");

-- CreateIndex
CREATE INDEX "SkillListingVersion_skillListingId_submissionStatus_isAvail_idx" ON "SkillListingVersion"("skillListingId", "submissionStatus", "isAvailable");

-- CreateIndex
CREATE INDEX "SkillListingVersion_submissionStatus_idx" ON "SkillListingVersion"("submissionStatus");

-- CreateIndex
CREATE INDEX "SkillListingVersion_reviewerId_idx" ON "SkillListingVersion"("reviewerId");

-- CreateIndex
CREATE INDEX "SkillListingVersion_organizationId_idx" ON "SkillListingVersion"("organizationId");

-- AddForeignKey
ALTER TABLE "SkillListing" ADD CONSTRAINT "SkillListing_activeVersionId_fkey" FOREIGN KEY ("activeVersionId") REFERENCES "SkillListingVersion"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillListing" ADD CONSTRAINT "SkillListing_owningUserId_fkey" FOREIGN KEY ("owningUserId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillListing" ADD CONSTRAINT "SkillListing_owner_Profile_fkey" FOREIGN KEY ("owningUserId") REFERENCES "Profile"("userId") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillListing" ADD CONSTRAINT "SkillListing_owningOrgId_fkey" FOREIGN KEY ("owningOrgId") REFERENCES "Organization"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillListingVersion" ADD CONSTRAINT "SkillListingVersion_skillListingId_fkey" FOREIGN KEY ("skillListingId") REFERENCES "SkillListing"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SkillListingVersion" ADD CONSTRAINT "SkillListingVersion_reviewerId_fkey" FOREIGN KEY ("reviewerId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
