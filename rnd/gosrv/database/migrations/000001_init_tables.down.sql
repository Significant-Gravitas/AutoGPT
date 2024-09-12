-- Drop foreign key constraints
ALTER TABLE "AnalyticsTracker" DROP CONSTRAINT IF EXISTS "AnalyticsTracker_agentId_fkey";
ALTER TABLE "InstallTracker" DROP CONSTRAINT IF EXISTS "InstallTracker_marketplaceAgentId_fkey";
ALTER TABLE "FeaturedAgent" DROP CONSTRAINT IF EXISTS "FeaturedAgent_agentId_fkey";

-- Drop indexes
DROP INDEX IF EXISTS "FeaturedAgent_agentId_key";
DROP INDEX IF EXISTS "FeaturedAgent_id_key";
DROP INDEX IF EXISTS "InstallTracker_marketplaceAgentId_installedAgentId_key";
DROP INDEX IF EXISTS "AnalyticsTracker_agentId_key";
DROP INDEX IF EXISTS "AnalyticsTracker_id_key";
DROP INDEX IF EXISTS "Agents_id_key";

-- Drop tables
DROP TABLE IF EXISTS "FeaturedAgent";
DROP TABLE IF EXISTS "InstallTracker";
DROP TABLE IF EXISTS "AnalyticsTracker";
DROP TABLE IF EXISTS "Agents";

-- Drop enums
DROP TYPE IF EXISTS "InstallationLocation";
DROP TYPE IF EXISTS "SubmissionStatus";
