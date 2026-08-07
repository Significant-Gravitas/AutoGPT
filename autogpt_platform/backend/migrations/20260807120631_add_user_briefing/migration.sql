-- CreateTable
CREATE TABLE "UserBriefing" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "briefingDate" DATE NOT NULL,
    "content" JSONB NOT NULL,

    CONSTRAINT "UserBriefing_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "UserBriefing_userId_briefingDate_key" ON "UserBriefing"("userId", "briefingDate");

-- CreateIndex
CREATE INDEX "UserBriefing_userId_createdAt_idx" ON "UserBriefing"("userId", "createdAt");

-- AddForeignKey
ALTER TABLE "UserBriefing" ADD CONSTRAINT "UserBriefing_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
