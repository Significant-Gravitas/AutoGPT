-- CreateTable
CREATE TABLE "UserAttribution" (
    "userId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "anonymousId" TEXT,
    "posthogDistinctId" TEXT,
    "datafastVisitorId" TEXT,
    "datafastSessionId" TEXT,
    "landingPath" TEXT,
    "referrer" TEXT,
    "utmSource" TEXT,
    "utmMedium" TEXT,
    "utmCampaign" TEXT,
    "signupMethod" TEXT,

    CONSTRAINT "UserAttribution_pkey" PRIMARY KEY ("userId")
);

-- CreateIndex
CREATE INDEX "UserAttribution_anonymousId_idx" ON "UserAttribution"("anonymousId");

-- CreateIndex
CREATE INDEX "UserAttribution_datafastVisitorId_idx" ON "UserAttribution"("datafastVisitorId");

-- AddForeignKey
ALTER TABLE "UserAttribution" ADD CONSTRAINT "UserAttribution_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
