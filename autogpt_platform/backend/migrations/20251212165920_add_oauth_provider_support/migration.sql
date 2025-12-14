-- CreateTable
CREATE TABLE "OAuthApplication" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "clientId" TEXT NOT NULL,
    "clientSecret" TEXT NOT NULL,
    "clientSecretSalt" TEXT NOT NULL,
    "redirectUris" TEXT[],
    "grantTypes" TEXT[] DEFAULT ARRAY['authorization_code', 'refresh_token']::TEXT[],
    "scopes" "APIKeyPermission"[],
    "ownerId" TEXT NOT NULL,
    "isActive" BOOLEAN NOT NULL DEFAULT true,

    CONSTRAINT "OAuthApplication_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OAuthAuthorizationCode" (
    "id" TEXT NOT NULL,
    "code" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "applicationId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "scopes" "APIKeyPermission"[],
    "redirectUri" TEXT NOT NULL,
    "codeChallenge" TEXT,
    "codeChallengeMethod" TEXT,
    "usedAt" TIMESTAMP(3),

    CONSTRAINT "OAuthAuthorizationCode_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OAuthAccessToken" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "applicationId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "scopes" "APIKeyPermission"[],
    "revokedAt" TIMESTAMP(3),

    CONSTRAINT "OAuthAccessToken_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "OAuthRefreshToken" (
    "id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "applicationId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "scopes" "APIKeyPermission"[],
    "revokedAt" TIMESTAMP(3),

    CONSTRAINT "OAuthRefreshToken_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "OAuthApplication_clientId_key" ON "OAuthApplication"("clientId");

-- CreateIndex
CREATE INDEX "OAuthApplication_clientId_idx" ON "OAuthApplication"("clientId");

-- CreateIndex
CREATE INDEX "OAuthApplication_ownerId_idx" ON "OAuthApplication"("ownerId");

-- CreateIndex
CREATE UNIQUE INDEX "OAuthAuthorizationCode_code_key" ON "OAuthAuthorizationCode"("code");

-- CreateIndex
CREATE INDEX "OAuthAuthorizationCode_code_idx" ON "OAuthAuthorizationCode"("code");

-- CreateIndex
CREATE INDEX "OAuthAuthorizationCode_applicationId_userId_idx" ON "OAuthAuthorizationCode"("applicationId", "userId");

-- CreateIndex
CREATE INDEX "OAuthAuthorizationCode_expiresAt_idx" ON "OAuthAuthorizationCode"("expiresAt");

-- CreateIndex
CREATE UNIQUE INDEX "OAuthAccessToken_token_key" ON "OAuthAccessToken"("token");

-- CreateIndex
CREATE INDEX "OAuthAccessToken_token_idx" ON "OAuthAccessToken"("token");

-- CreateIndex
CREATE INDEX "OAuthAccessToken_userId_applicationId_idx" ON "OAuthAccessToken"("userId", "applicationId");

-- CreateIndex
CREATE INDEX "OAuthAccessToken_expiresAt_idx" ON "OAuthAccessToken"("expiresAt");

-- CreateIndex
CREATE UNIQUE INDEX "OAuthRefreshToken_token_key" ON "OAuthRefreshToken"("token");

-- CreateIndex
CREATE INDEX "OAuthRefreshToken_token_idx" ON "OAuthRefreshToken"("token");

-- CreateIndex
CREATE INDEX "OAuthRefreshToken_userId_applicationId_idx" ON "OAuthRefreshToken"("userId", "applicationId");

-- CreateIndex
CREATE INDEX "OAuthRefreshToken_expiresAt_idx" ON "OAuthRefreshToken"("expiresAt");

-- AddForeignKey
ALTER TABLE "OAuthApplication" ADD CONSTRAINT "OAuthApplication_ownerId_fkey" FOREIGN KEY ("ownerId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OAuthAuthorizationCode" ADD CONSTRAINT "OAuthAuthorizationCode_applicationId_fkey" FOREIGN KEY ("applicationId") REFERENCES "OAuthApplication"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OAuthAuthorizationCode" ADD CONSTRAINT "OAuthAuthorizationCode_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OAuthAccessToken" ADD CONSTRAINT "OAuthAccessToken_applicationId_fkey" FOREIGN KEY ("applicationId") REFERENCES "OAuthApplication"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OAuthAccessToken" ADD CONSTRAINT "OAuthAccessToken_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OAuthRefreshToken" ADD CONSTRAINT "OAuthRefreshToken_applicationId_fkey" FOREIGN KEY ("applicationId") REFERENCES "OAuthApplication"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OAuthRefreshToken" ADD CONSTRAINT "OAuthRefreshToken_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
