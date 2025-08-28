-- Query to check for duplicate user/slug pairs in the StoreListing table
-- This should return 0 rows if the unique constraint is working properly
-- If any rows are returned, there are duplicate user/slug combinations

-- Method 1: Check for actual duplicates using GROUP BY and HAVING
SELECT 
    "owningUserId",
    "slug",
    COUNT(*) as duplicate_count,
    string_agg("id", ', ') as listing_ids
FROM "StoreListing"
WHERE "isDeleted" = false  -- Only check non-deleted listings
GROUP BY "owningUserId", "slug"
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, "owningUserId", "slug";

-- Method 2: Alternative approach - find all records that have duplicates
-- WITH duplicates AS (
--     SELECT "owningUserId", "slug"
--     FROM "StoreListing"
--     WHERE "isDeleted" = false
--     GROUP BY "owningUserId", "slug"
--     HAVING COUNT(*) > 1
-- )
-- SELECT 
--     sl."id",
--     sl."owningUserId", 
--     sl."slug",
--     sl."createdAt",
--     sl."updatedAt",
--     u."email" as owner_email,
--     u."name" as owner_name
-- FROM "StoreListing" sl
-- JOIN duplicates d ON sl."owningUserId" = d."owningUserId" AND sl."slug" = d."slug"
-- JOIN "User" u ON sl."owningUserId" = u."id"
-- WHERE sl."isDeleted" = false
-- ORDER BY sl."owningUserId", sl."slug", sl."createdAt";

-- Method 3: Summary statistics
-- SELECT 
--     COUNT(*) as total_listings,
--     COUNT(DISTINCT ("owningUserId", "slug")) as unique_user_slug_pairs,
--     (COUNT(*) - COUNT(DISTINCT ("owningUserId", "slug"))) as potential_duplicates
-- FROM "StoreListing"
-- WHERE "isDeleted" = false;