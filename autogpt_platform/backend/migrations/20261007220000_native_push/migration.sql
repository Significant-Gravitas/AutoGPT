CREATE TABLE "NativePushSubscription" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "provider" TEXT NOT NULL CHECK ("provider" IN ('apns', 'fcm')),
    "token" TEXT NOT NULL,
    "environment" TEXT NOT NULL CHECK ("environment" IN ('sandbox', 'production')),
    "origin" TEXT NOT NULL,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "NativePushSubscription_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX "NativePushSubscription_sessionId_key" ON "NativePushSubscription"("sessionId");
CREATE UNIQUE INDEX "NativePushSubscription_provider_token_environment_key" ON "NativePushSubscription"("provider", "token", "environment");
ALTER TABLE "NativePushSubscription" ADD CONSTRAINT "NativePushSubscription_sessionId_fkey"
    FOREIGN KEY ("sessionId") REFERENCES "UserAuthSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;
