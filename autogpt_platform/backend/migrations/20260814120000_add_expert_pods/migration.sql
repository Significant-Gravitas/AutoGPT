-- CreateTable
CREATE TABLE "ExpertPod" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,

    CONSTRAINT "ExpertPod_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "ExpertPod_userId_name_key" ON "ExpertPod"("userId", "name");

-- AlterTable
ALTER TABLE "Expert" ADD COLUMN "podId" TEXT;

-- CreateIndex
CREATE INDEX "Expert_podId_idx" ON "Expert"("podId");

-- AddForeignKey
ALTER TABLE "ExpertPod" ADD CONSTRAINT "ExpertPod_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Expert" ADD CONSTRAINT "Expert_podId_fkey" FOREIGN KEY ("podId") REFERENCES "ExpertPod"("id") ON DELETE SET NULL ON UPDATE CASCADE;
