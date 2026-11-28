-- The cleanup deletes a self-registered client after 30 days without use.
ALTER TABLE "OAuthApplication" ADD COLUMN "lastUsedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP;
