-- AlterTable
-- Marks presets that detach-on-archive deactivated, so re-hire reactivates
-- only those — a preset the user disabled themselves stays off. A constant
-- default on AgentPreset is a catalog-only change (no table rewrite).
ALTER TABLE "AgentPreset" ADD COLUMN IF NOT EXISTS "deactivatedByExpertArchive" BOOLEAN NOT NULL DEFAULT false;
