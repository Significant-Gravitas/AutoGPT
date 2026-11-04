-- A team organization's plan, set by an AutoGPT admin for a sales-led plan:
-- how many seats it pays for, when it ends, and the admin's note.
ALTER TABLE "OrganizationSubscription" ADD COLUMN "seats" INTEGER NOT NULL DEFAULT 1,
ADD COLUMN "endsAt" TIMESTAMP(3),
ADD COLUMN "note" TEXT;
