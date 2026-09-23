-- Snapshot the marketplace-published title/description onto the LibraryAgent at
-- download time so a downloaded agent appears in the library as it does in the
-- marketplace, instead of showing the creator's original graph name/description.
-- Nullable: user-created agents keep NULL and fall back to the graph's values.
-- AlterTable
ALTER TABLE "LibraryAgent" ADD COLUMN "name" TEXT;
ALTER TABLE "LibraryAgent" ADD COLUMN "description" TEXT;
