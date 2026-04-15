/*
  Warnings:

  - You are about to drop the column `source` on the `UserWorkspaceFile` table. All the data in the column will be lost.
  - You are about to drop the column `sourceExecId` on the `UserWorkspaceFile` table. All the data in the column will be lost.
  - You are about to drop the column `sourceSessionId` on the `UserWorkspaceFile` table. All the data in the column will be lost.

*/

-- AlterTable
ALTER TABLE "UserWorkspaceFile" DROP COLUMN "source",
DROP COLUMN "sourceExecId",
DROP COLUMN "sourceSessionId";

-- DropEnum
DROP TYPE "WorkspaceFileSource";
