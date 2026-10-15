-- When the user archived the chat; null = not archived. Archived chats leave
-- the user's session lists until the next turn clears it.

-- Nullable with no default: a catalog-only change, no table rewrite.
-- AlterTable
ALTER TABLE "ChatSession" ADD COLUMN     "archivedAt" TIMESTAMP(3);
