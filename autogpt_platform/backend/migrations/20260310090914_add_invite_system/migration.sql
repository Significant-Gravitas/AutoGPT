/*
  Warnings:

  - You are about to drop the column `search` on the `StoreListingVersion` table. All the data in the column will be lost.

*/-- CreateEnum
CREATE TYPE "InvitedUserStatus" AS ENUM('INVITED',
'CLAIMED',
'REVOKED');
-- CreateEnum
CREATE TYPE "TallyComputationStatus" AS ENUM('PENDING',
'RUNNING',
'READY',
'FAILED');
-- CreateTable
CREATE TABLE "InvitedUser"(
  "id" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL,
  "email" TEXT NOT NULL,
  "status" "InvitedUserStatus" NOT NULL DEFAULT 'INVITED',
  "authUserId" TEXT,
  "name" TEXT,
  "tallyUnderstanding" JSONB,
  "tallyStatus" "TallyComputationStatus" NOT NULL DEFAULT 'PENDING',
  "tallyComputedAt" TIMESTAMP(3),
  "tallyError" TEXT,
  CONSTRAINT "InvitedUser_pkey" PRIMARY KEY("id")
);
-- CreateIndex
CREATE UNIQUE INDEX "InvitedUser_email_key"
ON "InvitedUser"("email");
-- CreateIndex
CREATE UNIQUE INDEX "InvitedUser_authUserId_key"
ON "InvitedUser"("authUserId");
-- CreateIndex
CREATE INDEX "InvitedUser_status_idx"
ON "InvitedUser"("status");
-- CreateIndex
CREATE INDEX "InvitedUser_tallyStatus_idx"
ON "InvitedUser"("tallyStatus");
-- AddForeignKey
ALTER TABLE "InvitedUser" ADD CONSTRAINT "InvitedUser_authUserId_fkey" FOREIGN KEY("authUserId") REFERENCES "User"("id")
ON DELETE 
SET NULL
ON UPDATE CASCADE;
