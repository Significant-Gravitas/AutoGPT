-- When a plan or trial last dropped to NO_TIER, for the product-mail
-- audience's 7-day win-back window. NULL until a tier first ends.
ALTER TABLE "User"
    ADD COLUMN "subscriptionEndedAt" TIMESTAMP(3);
