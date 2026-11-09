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

-- CreateIndex
-- "PushSubscription_userId_endpoint_key" stays: backend pods still running
-- the previous release upsert on it during a rolling deploy.
CREATE UNIQUE INDEX "PushSubscription_endpoint_key" ON "PushSubscription"("endpoint");
