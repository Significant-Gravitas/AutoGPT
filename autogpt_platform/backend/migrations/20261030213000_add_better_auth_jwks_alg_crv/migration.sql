-- Better Auth 1.7's jwt plugin stores each signing key's algorithm and curve,
-- and its startup schema check refuses every auth request while they are
-- missing. Existing keys keep NULL, which Better Auth reads as the configured
-- algorithm (ES256).

-- AlterTable
ALTER TABLE "UserAuthJwks" ADD COLUMN     "alg" TEXT,
ADD COLUMN     "crv" TEXT;
