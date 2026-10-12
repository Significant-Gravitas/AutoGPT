-- Projects: many boards per user. A null ChatSession.boardId is the user's
-- default board (Home), so sessions created outside the board need no write.

-- CreateTable
CREATE TABLE "Board" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "isDefault" BOOLEAN NOT NULL DEFAULT false,
    "position" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "columns" JSONB NOT NULL,
    CONSTRAINT "Board_pkey" PRIMARY KEY ("id")
);
CREATE INDEX "Board_userId_position_idx" ON "Board"("userId", "position");
-- One default board per user; Prisma cannot express a partial unique index.
CREATE UNIQUE INDEX "Board_userId_default_key" ON "Board"("userId") WHERE "isDefault";
ALTER TABLE "Board" ADD CONSTRAINT "Board_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Each user's stored columns become their default board.
INSERT INTO "Board" ("id", "updatedAt", "userId", "name", "isDefault", "position", "columns")
SELECT gen_random_uuid()::text, "updatedAt", "userId", 'Home', true, 0, "columns" FROM "UserBoard";

DROP TABLE "UserBoard";

-- Nullable with no default: a catalog-only change, no table rewrite.
ALTER TABLE "ChatSession" ADD COLUMN "boardId" TEXT;
ALTER TABLE "ChatSession" ADD CONSTRAINT "ChatSession_boardId_fkey" FOREIGN KEY ("boardId") REFERENCES "Board"("id") ON DELETE SET NULL ON UPDATE CASCADE;

DROP INDEX IF EXISTS "ChatSession_userId_boardColumn_boardPosition_idx";
-- Brief ShareLock on ChatSession while it builds, like the earlier board index.
CREATE INDEX IF NOT EXISTS "ChatSession_userId_boardId_boardColumn_boardPosition_idx" ON "ChatSession"("userId", "boardId", "boardColumn", "boardPosition");
