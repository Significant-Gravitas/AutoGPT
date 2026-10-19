-- A browser has one push endpoint, so it can only belong to one account.
-- Rows were unique per (userId, endpoint), which let the same browser be
-- registered to several accounts after switching between them; keep the most
-- recently updated row per endpoint (the account signed in there last).
DELETE FROM "PushSubscription" AS older
USING "PushSubscription" AS newer
WHERE older."endpoint" = newer."endpoint"
  AND (
    older."updatedAt" < newer."updatedAt"
    OR (older."updatedAt" = newer."updatedAt" AND older."id" < newer."id")
  );

-- DropIndex
DROP INDEX "PushSubscription_userId_endpoint_key";

-- CreateIndex
CREATE UNIQUE INDEX "PushSubscription_endpoint_key" ON "PushSubscription"("endpoint");

-- CreateIndex
CREATE INDEX "PushSubscription_userId_idx" ON "PushSubscription"("userId");
