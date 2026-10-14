-- Task board: where each chat sits on the user's board, and the user's columns.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "ChatSession" ADD COLUMN     "boardColumn" TEXT,
ADD COLUMN     "boardPosition" DOUBLE PRECISION;

-- CreateTable
CREATE TABLE "UserBoard" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "columns" JSONB NOT NULL,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "UserBoard_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "UserBoard_userId_key" ON "UserBoard"("userId");

-- CreateIndex
-- Prisma's `migrate deploy` wraps each migration file in a single
-- transaction, and Postgres rejects `CREATE INDEX CONCURRENTLY` inside a
-- transaction block. The plain `CREATE INDEX` form briefly acquires a
-- ShareLock on ChatSession; for environments that can't tolerate that, run
-- the equivalent `CREATE INDEX CONCURRENTLY` out-of-band before this
-- migration ships and Postgres will skip the recreate via `IF NOT EXISTS`.
CREATE INDEX IF NOT EXISTS "ChatSession_userId_boardColumn_boardPosition_idx"
    ON "ChatSession" ("userId", "boardColumn", "boardPosition");

-- AddForeignKey
ALTER TABLE "UserBoard" ADD CONSTRAINT "UserBoard_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
