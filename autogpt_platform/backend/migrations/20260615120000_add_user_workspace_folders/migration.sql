-- AlterTable
ALTER TABLE "UserWorkspaceFile" ADD COLUMN     "folderId" TEXT;

-- CreateTable
CREATE TABLE "UserWorkspaceFolder" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "workspaceId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "icon" TEXT,
    "parentId" TEXT,
    "isDeleted" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "UserWorkspaceFolder_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "UserWorkspaceFolder_workspaceId_isDeleted_idx" ON "UserWorkspaceFolder"("workspaceId", "isDeleted");

-- CreateIndex
CREATE UNIQUE INDEX "UserWorkspaceFolder_workspaceId_parentId_name_key" ON "UserWorkspaceFolder"("workspaceId", "parentId", "name");

-- CreateIndex
CREATE INDEX "UserWorkspaceFile_folderId_idx" ON "UserWorkspaceFile"("folderId");

-- AddForeignKey
ALTER TABLE "UserWorkspaceFile" ADD CONSTRAINT "UserWorkspaceFile_folderId_fkey" FOREIGN KEY ("folderId") REFERENCES "UserWorkspaceFolder"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserWorkspaceFolder" ADD CONSTRAINT "UserWorkspaceFolder_workspaceId_fkey" FOREIGN KEY ("workspaceId") REFERENCES "UserWorkspace"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserWorkspaceFolder" ADD CONSTRAINT "UserWorkspaceFolder_parentId_fkey" FOREIGN KEY ("parentId") REFERENCES "UserWorkspaceFolder"("id") ON DELETE CASCADE ON UPDATE CASCADE;
