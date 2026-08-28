ALTER TABLE "BuilderSearchHistory"
ADD COLUMN "teamId" TEXT;

CREATE INDEX "BuilderSearchHistory_organizationId_teamId_idx"
ON "BuilderSearchHistory"("organizationId", "teamId");
