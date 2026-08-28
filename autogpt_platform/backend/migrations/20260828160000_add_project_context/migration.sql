CREATE TABLE "ProjectContext" (
  "id" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "ownerUserId" TEXT NOT NULL,
  "managerSessionId" TEXT NOT NULL,
  "title" TEXT NOT NULL,
  "summary" TEXT NOT NULL DEFAULT '',
  "phase" TEXT NOT NULL DEFAULT '',
  "decisions" TEXT[] DEFAULT ARRAY[]::TEXT[],
  "constraints" TEXT[] DEFAULT ARRAY[]::TEXT[],
  "artifacts" JSONB NOT NULL DEFAULT '[]',
  "active" BOOLEAN NOT NULL DEFAULT true,
  CONSTRAINT "ProjectContext_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "ProjectContext_managerSessionId_key"
  ON "ProjectContext"("managerSessionId");
CREATE INDEX "ProjectContext_ownerUserId_active_updatedAt_idx"
  ON "ProjectContext"("ownerUserId", "active", "updatedAt");

ALTER TABLE "ProjectContext"
  ADD CONSTRAINT "ProjectContext_ownerUserId_fkey"
  FOREIGN KEY ("ownerUserId") REFERENCES "User"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ProjectContext"
  ADD CONSTRAINT "ProjectContext_managerSessionId_fkey"
  FOREIGN KEY ("managerSessionId") REFERENCES "ChatSession"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
