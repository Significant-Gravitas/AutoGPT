-- The ban record kept when a banned account is erased (SECRT-2854): a keyed
-- hash of its address, the ban's category and dates, never the address.

-- CreateEnum
CREATE TYPE "AccountBanCategory" AS ENUM ('ADMIN');

-- CreateTable
CREATE TABLE "AccountBanRecord" (
    "id" TEXT NOT NULL,
    "emailHash" TEXT NOT NULL,
    "category" "AccountBanCategory" NOT NULL,
    "recordedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deleteAfter" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "AccountBanRecord_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "AccountBanRecord_emailHash_key" ON "AccountBanRecord"("emailHash");

-- CreateIndex
CREATE INDEX "AccountBanRecord_deleteAfter_idx" ON "AccountBanRecord"("deleteAfter");
