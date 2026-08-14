-- DropIndex
DROP INDEX "ExpertPod_userId_idx";

-- CreateIndex
CREATE UNIQUE INDEX "ExpertPod_userId_name_key" ON "ExpertPod"("userId", "name");
