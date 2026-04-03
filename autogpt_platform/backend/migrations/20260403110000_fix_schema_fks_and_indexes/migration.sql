-- DropForeignKey
ALTER TABLE "IntegrationCredential" DROP CONSTRAINT "IntegrationCredential_workspace_fkey";

-- DropIndex
DROP INDEX "Organization_slug_idx";

-- AlterTable
ALTER TABLE "IntegrationCredential" ADD COLUMN     "workspaceId" TEXT;

-- AddForeignKey
ALTER TABLE "WorkspaceInvite" ADD CONSTRAINT "WorkspaceInvite_workspaceId_fkey" FOREIGN KEY ("workspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "IntegrationCredential" ADD CONSTRAINT "IntegrationCredential_workspace_fkey" FOREIGN KEY ("workspaceId") REFERENCES "OrgWorkspace"("id") ON DELETE CASCADE ON UPDATE CASCADE;

