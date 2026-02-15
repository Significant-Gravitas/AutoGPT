-- CreateEnum
CREATE TYPE "WaitlistExternalStatus" AS ENUM ('DONE', 'NOT_STARTED', 'CANCELED', 'WORK_IN_PROGRESS');

-- AlterEnum
ALTER TYPE "NotificationType" ADD VALUE 'WAITLIST_LAUNCH';

-- CreateTable
CREATE TABLE "WaitlistEntry" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "storeListingId" TEXT,
    "owningUserId" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "search" tsvector DEFAULT ''::tsvector,
    "name" TEXT NOT NULL,
    "subHeading" TEXT NOT NULL,
    "videoUrl" TEXT,
    "agentOutputDemoUrl" TEXT,
    "imageUrls" TEXT[],
    "description" TEXT NOT NULL,
    "categories" TEXT[],
    "status" "WaitlistExternalStatus" NOT NULL DEFAULT 'NOT_STARTED',
    "votes" INTEGER NOT NULL DEFAULT 0,
    "unaffiliatedEmailUsers" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "WaitlistEntry_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "_joinedWaitlists" (
    "A" TEXT NOT NULL,
    "B" TEXT NOT NULL
);

-- CreateIndex
CREATE UNIQUE INDEX "_joinedWaitlists_AB_unique" ON "_joinedWaitlists"("A", "B");

-- CreateIndex
CREATE INDEX "_joinedWaitlists_B_index" ON "_joinedWaitlists"("B");

-- AddForeignKey
ALTER TABLE "WaitlistEntry" ADD CONSTRAINT "WaitlistEntry_storeListingId_fkey" FOREIGN KEY ("storeListingId") REFERENCES "StoreListing"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "WaitlistEntry" ADD CONSTRAINT "WaitlistEntry_owningUserId_fkey" FOREIGN KEY ("owningUserId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_joinedWaitlists" ADD CONSTRAINT "_joinedWaitlists_A_fkey" FOREIGN KEY ("A") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "_joinedWaitlists" ADD CONSTRAINT "_joinedWaitlists_B_fkey" FOREIGN KEY ("B") REFERENCES "WaitlistEntry"("id") ON DELETE CASCADE ON UPDATE CASCADE;
