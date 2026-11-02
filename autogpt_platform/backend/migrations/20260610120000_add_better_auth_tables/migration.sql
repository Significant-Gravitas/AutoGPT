-- Better Auth tables. Owned by Prisma migrations; read/written at runtime by
-- the Better Auth service embedded in the frontend via its pg adapter.

-- CreateTable
CREATE TABLE "UserAuthIdentity" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "emailVerified" BOOLEAN NOT NULL DEFAULT false,
    "image" TEXT,
    "role" TEXT,
    "banned" BOOLEAN,
    "banReason" TEXT,
    "banExpires" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UserAuthIdentity_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserAuthSession" (
    "id" TEXT NOT NULL,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "token" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "ipAddress" TEXT,
    "userAgent" TEXT,
    "impersonatedBy" TEXT,
    "userId" TEXT NOT NULL,

    CONSTRAINT "UserAuthSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserAuthAccount" (
    "id" TEXT NOT NULL,
    "accountId" TEXT NOT NULL,
    "providerId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "accessToken" TEXT,
    "refreshToken" TEXT,
    "idToken" TEXT,
    "accessTokenExpiresAt" TIMESTAMP(3),
    "refreshTokenExpiresAt" TIMESTAMP(3),
    "scope" TEXT,
    "password" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UserAuthAccount_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserAuthVerification" (
    "id" TEXT NOT NULL,
    "identifier" TEXT NOT NULL,
    "value" TEXT NOT NULL,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "UserAuthVerification_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "UserAuthJwks" (
    "id" TEXT NOT NULL,
    "publicKey" TEXT NOT NULL,
    "privateKey" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3),

    CONSTRAINT "UserAuthJwks_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "UserAuthIdentity_email_key" ON "UserAuthIdentity"("email");

-- CreateIndex
CREATE UNIQUE INDEX "UserAuthSession_token_key" ON "UserAuthSession"("token");

-- CreateIndex
CREATE INDEX "UserAuthSession_userId_idx" ON "UserAuthSession"("userId");

-- CreateIndex
CREATE INDEX "UserAuthAccount_userId_idx" ON "UserAuthAccount"("userId");

-- CreateIndex
CREATE INDEX "UserAuthVerification_identifier_idx" ON "UserAuthVerification"("identifier");

-- AddForeignKey
ALTER TABLE "UserAuthSession" ADD CONSTRAINT "UserAuthSession_userId_fkey" FOREIGN KEY ("userId") REFERENCES "UserAuthIdentity"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "UserAuthAccount" ADD CONSTRAINT "UserAuthAccount_userId_fkey" FOREIGN KEY ("userId") REFERENCES "UserAuthIdentity"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Remove the legacy Supabase GoTrue -> platform sync trigger. Platform User
-- rows are created by get_or_create_user on first authenticated request, and
-- new signups land in the Better Auth "UserAuthIdentity" table instead of auth.users.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'auth'
    AND table_name = 'users'
  ) THEN
    DROP TRIGGER IF EXISTS user_added_to_platform ON auth.users;
  END IF;
  DROP FUNCTION IF EXISTS add_user_and_profile_to_platform();
END $$;
