-- CreateEnum
CREATE TYPE "SharedVia" AS ENUM ('USER', 'CHAT_LINK');

-- AlterTable: ChatSession sharing columns
ALTER TABLE "ChatSession"
    ADD COLUMN "isShared"   BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN "shareToken" TEXT,
    ADD COLUMN "sharedAt"   TIMESTAMP(3);

-- AlterTable: AgentGraphExecution share provenance
ALTER TABLE "AgentGraphExecution"
    ADD COLUMN "sharedVia" "SharedVia";

-- Backfill provenance for already-shared executions.  Any pre-existing
-- ``isShared=true`` rows were created via the user-only sharing flow,
-- so they are ``USER``.  Leaves new rows with NULL until a share is
-- enabled (the route writes ``sharedVia`` together with ``isShared``).
UPDATE "AgentGraphExecution"
   SET "sharedVia" = 'USER'
 WHERE "isShared" = true
   AND "sharedVia" IS NULL;

-- CreateIndex
CREATE UNIQUE INDEX "ChatSession_shareToken_key" ON "ChatSession"("shareToken");
CREATE INDEX "ChatSession_shareToken_idx" ON "ChatSession"("shareToken");

-- CreateTable
CREATE TABLE "SharedChatFile" (
    "id"          TEXT NOT NULL,
    "createdAt"   TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sessionId"   TEXT NOT NULL,
    "fileId"      TEXT NOT NULL,
    "shareToken"  TEXT NOT NULL,

    CONSTRAINT "SharedChatFile_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "SharedChatFile_shareToken_fileId_key" ON "SharedChatFile"("shareToken", "fileId");
CREATE INDEX "SharedChatFile_shareToken_idx" ON "SharedChatFile"("shareToken");
CREATE INDEX "SharedChatFile_sessionId_idx" ON "SharedChatFile"("sessionId");

ALTER TABLE "SharedChatFile"
    ADD CONSTRAINT "SharedChatFile_sessionId_fkey"
    FOREIGN KEY ("sessionId") REFERENCES "ChatSession"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "SharedChatFile"
    ADD CONSTRAINT "SharedChatFile_fileId_fkey"
    FOREIGN KEY ("fileId") REFERENCES "UserWorkspaceFile"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;

-- CreateTable
CREATE TABLE "ChatLinkedShare" (
    "id"          TEXT NOT NULL,
    "createdAt"   TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sessionId"   TEXT NOT NULL,
    "executionId" TEXT NOT NULL,

    CONSTRAINT "ChatLinkedShare_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "ChatLinkedShare_sessionId_executionId_key" ON "ChatLinkedShare"("sessionId", "executionId");
CREATE INDEX "ChatLinkedShare_sessionId_idx" ON "ChatLinkedShare"("sessionId");
CREATE INDEX "ChatLinkedShare_executionId_idx" ON "ChatLinkedShare"("executionId");

ALTER TABLE "ChatLinkedShare"
    ADD CONSTRAINT "ChatLinkedShare_sessionId_fkey"
    FOREIGN KEY ("sessionId") REFERENCES "ChatSession"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ChatLinkedShare"
    ADD CONSTRAINT "ChatLinkedShare_executionId_fkey"
    FOREIGN KEY ("executionId") REFERENCES "AgentGraphExecution"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
