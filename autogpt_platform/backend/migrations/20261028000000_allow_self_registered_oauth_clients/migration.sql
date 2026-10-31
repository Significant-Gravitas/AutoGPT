-- A client that registers itself (RFC 7591) is public and has no owner.
ALTER TABLE "OAuthApplication" ALTER COLUMN "clientSecret" DROP NOT NULL,
ALTER COLUMN "clientSecretSalt" DROP NOT NULL,
ALTER COLUMN "ownerId" DROP NOT NULL;
