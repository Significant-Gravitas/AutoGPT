-- ================================================
-- PRODUCTION DUPLICATE USER/SLUG PAIRS CHECK
-- ================================================
-- This script checks for duplicate user/slug pairs in the StoreListing table
-- Run this query against your production database to verify data integrity

-- 1. SUMMARY STATISTICS
-- ---------------------
SELECT 
    '=== SUMMARY STATISTICS ===' as section,
    NULL as user_id,
    NULL as slug,
    NULL as count,
    NULL as details;

SELECT 
    'Total Listings' as metric,
    NULL as user_id,
    NULL as slug,
    COUNT(*) as count,
    'Non-deleted StoreListing records' as details
FROM "StoreListing"
WHERE "isDeleted" = false;

SELECT 
    'Unique User/Slug Pairs' as metric,
    NULL as user_id,
    NULL as slug,
    COUNT(DISTINCT CONCAT("owningUserId", '|', "slug")) as count,
    'Distinct combinations' as details
FROM "StoreListing"
WHERE "isDeleted" = false;

-- 2. CONSTRAINT CHECK
-- -------------------
SELECT 
    '=== CONSTRAINT CHECK ===' as section,
    NULL as user_id,
    NULL as slug,
    NULL as count,
    NULL as details;

SELECT 
    'Unique Constraint' as metric,
    NULL as user_id,
    NULL as slug,
    CASE 
        WHEN COUNT(*) > 0 THEN 1 
        ELSE 0 
    END as count,
    CASE 
        WHEN COUNT(*) > 0 THEN 'Constraint exists'
        ELSE 'Constraint missing!'
    END as details
FROM pg_indexes 
WHERE tablename = 'StoreListing' 
  AND indexdef LIKE '%owningUserId%slug%'
  AND indexdef LIKE '%UNIQUE%';

-- 3. DUPLICATE CHECK (MAIN QUERY)
-- --------------------------------
SELECT 
    '=== DUPLICATE PAIRS ===' as section,
    NULL as user_id,
    NULL as slug,
    NULL as count,
    NULL as details;

-- This is the main query - if it returns any rows, you have duplicates
SELECT 
    'DUPLICATE FOUND' as section,
    "owningUserId" as user_id,
    "slug" as slug,
    COUNT(*) as count,
    string_agg("id", ', ') as details
FROM "StoreListing"
WHERE "isDeleted" = false
GROUP BY "owningUserId", "slug"
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC, "owningUserId", "slug";

-- 4. DETAILED DUPLICATE ANALYSIS
-- -------------------------------
-- Uncomment this section if duplicates are found above
-- to get more details about the duplicate records

/*
WITH duplicates AS (
    SELECT "owningUserId", "slug"
    FROM "StoreListing"
    WHERE "isDeleted" = false
    GROUP BY "owningUserId", "slug"
    HAVING COUNT(*) > 1
)
SELECT 
    '=== DUPLICATE DETAILS ===' as analysis,
    sl."id" as listing_id,
    sl."owningUserId" as user_id,
    sl."slug" as slug,
    sl."createdAt",
    sl."updatedAt",
    u."email" as owner_email,
    u."name" as owner_name
FROM "StoreListing" sl
JOIN duplicates d ON sl."owningUserId" = d."owningUserId" AND sl."slug" = d."slug"
LEFT JOIN "User" u ON sl."owningUserId" = u."id"
WHERE sl."isDeleted" = false
ORDER BY sl."owningUserId", sl."slug", sl."createdAt";
*/

-- 5. FINAL RESULT INTERPRETATION
-- ------------------------------
SELECT 
    '=== INTERPRETATION ===' as section,
    NULL as user_id,
    NULL as slug,
    NULL as count,
    'If no rows returned in DUPLICATE PAIRS section above, then NO duplicates exist' as details;

SELECT 
    '=== EXPECTED RESULT ===' as section,
    NULL as user_id,
    NULL as slug,
    NULL as count,
    'Zero rows in DUPLICATE PAIRS = Database constraint is working correctly' as details;