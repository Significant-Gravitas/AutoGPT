CREATE TABLE "PaidUsageActivation" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL UNIQUE REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    "stripeSubscriptionId" TEXT NOT NULL UNIQUE,
    "stripeInvoiceId" TEXT NOT NULL UNIQUE,
    "readyAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE "PaidUsageActivationPolicy" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "startsAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO "PaidUsageActivationPolicy" ("id", "startsAt")
VALUES ('initial-pro-v1', date_trunc('second', CURRENT_TIMESTAMP));
CREATE TABLE "ProActivationAttempt" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT NOT NULL UNIQUE REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    "stripeSubscriptionId" TEXT NOT NULL UNIQUE,
    "stripeCustomerId" TEXT NOT NULL,
    "terms" JSONB NOT NULL,
    "returnTo" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'quoted',
    "confirmedAt" TIMESTAMP(3),
    "invoiceId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL
);
