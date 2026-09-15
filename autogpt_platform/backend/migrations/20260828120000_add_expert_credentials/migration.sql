-- CreateTable
CREATE TABLE "ExpertCredential" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expertId" TEXT NOT NULL,
    "credentialId" TEXT NOT NULL,
    "provider" TEXT NOT NULL,

    CONSTRAINT "ExpertCredential_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "ExpertCredential_expertId_credentialId_key" ON "ExpertCredential"("expertId", "credentialId");

-- CreateIndex
CREATE INDEX "ExpertCredential_expertId_idx" ON "ExpertCredential"("expertId");

-- AddForeignKey
ALTER TABLE "ExpertCredential" ADD CONSTRAINT "ExpertCredential_expertId_fkey" FOREIGN KEY ("expertId") REFERENCES "Expert"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AlterTable
-- Left NULL for every existing expert on purpose: the first read of an
-- expert's integrations seeds the allow-list from its installed workflows,
-- so enforcement cannot lock out a roster that predates this table.
ALTER TABLE "Expert" ADD COLUMN "credentialsSeededAt" TIMESTAMP(3);
