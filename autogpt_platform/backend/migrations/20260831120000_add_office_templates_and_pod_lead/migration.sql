-- CreateEnum
CREATE TYPE "ExpertAutonomyLevel" AS ENUM ('SUGGEST', 'ASK_FIRST', 'AUTONOMOUS');

-- AlterTable
ALTER TABLE "Expert" ADD COLUMN "autonomyLevel" "ExpertAutonomyLevel" NOT NULL DEFAULT 'SUGGEST';

-- AlterTable
ALTER TABLE "ExpertPod" ADD COLUMN "leadExpertId" TEXT;

-- CreateTable
CREATE TABLE "OfficeTemplate" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "config" JSONB NOT NULL,

    CONSTRAINT "OfficeTemplate_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "OfficeTemplate_name_key" ON "OfficeTemplate"("name");

-- AddForeignKey
ALTER TABLE "ExpertPod" ADD CONSTRAINT "ExpertPod_leadExpertId_fkey" FOREIGN KEY ("leadExpertId") REFERENCES "Expert"("id") ON DELETE SET NULL ON UPDATE CASCADE;
