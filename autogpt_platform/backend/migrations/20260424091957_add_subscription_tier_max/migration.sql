-- Add MAX between PRO and BUSINESS. ADD VALUE ... BEFORE preserves existing rows
-- (nothing currently on BUSINESS/ENTERPRISE via self-service flows).
ALTER TYPE "SubscriptionTier" ADD VALUE IF NOT EXISTS 'MAX' BEFORE 'BUSINESS';
