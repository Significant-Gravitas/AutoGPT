-- Database Migration Verification Script
-- Run this on both source (Supabase) and target (GCP) databases to compare

SET search_path TO platform;

-- ============================================
-- TABLE ROW COUNTS
-- ============================================

SELECT '=== TABLE ROW COUNTS ===' as section;

SELECT 'User' as table_name, COUNT(*) as row_count FROM "User"
UNION ALL SELECT 'Profile', COUNT(*) FROM "Profile"
UNION ALL SELECT 'UserOnboarding', COUNT(*) FROM "UserOnboarding"
UNION ALL SELECT 'UserBalance', COUNT(*) FROM "UserBalance"
UNION ALL SELECT 'AgentGraph', COUNT(*) FROM "AgentGraph"
UNION ALL SELECT 'AgentNode', COUNT(*) FROM "AgentNode"
UNION ALL SELECT 'AgentBlock', COUNT(*) FROM "AgentBlock"
UNION ALL SELECT 'AgentNodeLink', COUNT(*) FROM "AgentNodeLink"
UNION ALL SELECT 'AgentGraphExecution', COUNT(*) FROM "AgentGraphExecution"
UNION ALL SELECT 'AgentNodeExecution', COUNT(*) FROM "AgentNodeExecution"
UNION ALL SELECT 'AgentNodeExecutionInputOutput', COUNT(*) FROM "AgentNodeExecutionInputOutput"
UNION ALL SELECT 'AgentNodeExecutionKeyValueData', COUNT(*) FROM "AgentNodeExecutionKeyValueData"
UNION ALL SELECT 'AgentPreset', COUNT(*) FROM "AgentPreset"
UNION ALL SELECT 'LibraryAgent', COUNT(*) FROM "LibraryAgent"
UNION ALL SELECT 'StoreListing', COUNT(*) FROM "StoreListing"
UNION ALL SELECT 'StoreListingVersion', COUNT(*) FROM "StoreListingVersion"
UNION ALL SELECT 'StoreListingReview', COUNT(*) FROM "StoreListingReview"
UNION ALL SELECT 'IntegrationWebhook', COUNT(*) FROM "IntegrationWebhook"
UNION ALL SELECT 'APIKey', COUNT(*) FROM "APIKey"
UNION ALL SELECT 'CreditTransaction', COUNT(*) FROM "CreditTransaction"
UNION ALL SELECT 'CreditRefundRequest', COUNT(*) FROM "CreditRefundRequest"
UNION ALL SELECT 'AnalyticsDetails', COUNT(*) FROM "AnalyticsDetails"
UNION ALL SELECT 'AnalyticsMetrics', COUNT(*) FROM "AnalyticsMetrics"
UNION ALL SELECT 'SearchTerms', COUNT(*) FROM "SearchTerms"
UNION ALL SELECT 'NotificationEvent', COUNT(*) FROM "NotificationEvent"
UNION ALL SELECT 'UserNotificationBatch', COUNT(*) FROM "UserNotificationBatch"
UNION ALL SELECT 'BuilderSearchHistory', COUNT(*) FROM "BuilderSearchHistory"
UNION ALL SELECT 'PendingHumanReview', COUNT(*) FROM "PendingHumanReview"
UNION ALL SELECT 'RefreshToken', COUNT(*) FROM "RefreshToken"
UNION ALL SELECT 'PasswordResetToken', COUNT(*) FROM "PasswordResetToken"
ORDER BY table_name;

-- ============================================
-- AUTH DATA VERIFICATION
-- ============================================

SELECT '=== AUTH DATA VERIFICATION ===' as section;

SELECT
    COUNT(*) as total_users,
    COUNT("passwordHash") as users_with_password,
    COUNT("googleId") as users_with_google,
    COUNT(CASE WHEN "emailVerified" = true THEN 1 END) as verified_emails,
    COUNT(CASE WHEN "passwordHash" IS NULL AND "googleId" IS NULL THEN 1 END) as users_without_auth
FROM "User";

-- ============================================
-- VIEW VERIFICATION
-- ============================================

SELECT '=== VIEW VERIFICATION ===' as section;

SELECT 'StoreAgent' as view_name, COUNT(*) as row_count FROM "StoreAgent"
UNION ALL SELECT 'Creator', COUNT(*) FROM "Creator"
UNION ALL SELECT 'StoreSubmission', COUNT(*) FROM "StoreSubmission";

-- ============================================
-- MATERIALIZED VIEW VERIFICATION
-- ============================================

SELECT '=== MATERIALIZED VIEW VERIFICATION ===' as section;

SELECT 'mv_agent_run_counts' as view_name, COUNT(*) as row_count FROM "mv_agent_run_counts"
UNION ALL SELECT 'mv_review_stats', COUNT(*) FROM "mv_review_stats";

-- ============================================
-- FOREIGN KEY INTEGRITY CHECKS
-- ============================================

SELECT '=== FOREIGN KEY INTEGRITY (should all be 0) ===' as section;

SELECT 'Orphaned Profiles' as check_name,
    COUNT(*) as count
FROM "Profile" p
WHERE p."userId" IS NOT NULL
AND NOT EXISTS (SELECT 1 FROM "User" u WHERE u.id = p."userId");

SELECT 'Orphaned AgentGraphs' as check_name,
    COUNT(*) as count
FROM "AgentGraph" g
WHERE NOT EXISTS (SELECT 1 FROM "User" u WHERE u.id = g."userId");

SELECT 'Orphaned AgentNodes' as check_name,
    COUNT(*) as count
FROM "AgentNode" n
WHERE NOT EXISTS (
    SELECT 1 FROM "AgentGraph" g
    WHERE g.id = n."agentGraphId" AND g.version = n."agentGraphVersion"
);

SELECT 'Orphaned Executions' as check_name,
    COUNT(*) as count
FROM "AgentGraphExecution" e
WHERE NOT EXISTS (SELECT 1 FROM "User" u WHERE u.id = e."userId");

SELECT 'Orphaned LibraryAgents' as check_name,
    COUNT(*) as count
FROM "LibraryAgent" l
WHERE NOT EXISTS (SELECT 1 FROM "User" u WHERE u.id = l."userId");

-- ============================================
-- SAMPLE DATA VERIFICATION
-- ============================================

SELECT '=== SAMPLE USERS (first 5) ===' as section;

SELECT
    id,
    email,
    "emailVerified",
    CASE WHEN "passwordHash" IS NOT NULL THEN 'YES' ELSE 'NO' END as has_password,
    CASE WHEN "googleId" IS NOT NULL THEN 'YES' ELSE 'NO' END as has_google,
    "createdAt"
FROM "User"
ORDER BY "createdAt" DESC
LIMIT 5;

-- ============================================
-- STORE LISTINGS SAMPLE
-- ============================================

SELECT '=== SAMPLE STORE LISTINGS (first 5) ===' as section;

SELECT
    id,
    slug,
    "isDeleted",
    "hasApprovedVersion"
FROM "StoreListing"
LIMIT 5;
