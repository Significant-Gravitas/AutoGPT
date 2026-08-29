-- Add MAX between PRO and BUSINESS and rename FREE → BASIC. ADD VALUE BEFORE
-- preserves existing rows on BUSINESS/ENTERPRISE; RENAME VALUE updates the
-- label in place so existing FREE rows (the common case) remain valid.
ALTER TYPE "SubscriptionTier" ADD VALUE IF NOT EXISTS 'MAX' BEFORE 'BUSINESS';
ALTER TYPE "SubscriptionTier" RENAME VALUE 'FREE' TO 'BASIC';
