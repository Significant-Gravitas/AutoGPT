# User/Slug Duplicate Check Scripts

This directory contains scripts to check for duplicate user/slug pairs in the `StoreListing` table to verify that the unique constraint `@@unique([owningUserId, slug])` is working properly.

## Files Created

### 1. `production_duplicate_check.sql`
**RECOMMENDED FOR PRODUCTION**

A comprehensive SQL script that can be run directly against your production database. This script provides:
- Summary statistics about the StoreListing table
- Verification that the unique constraint exists
- Detection of any duplicate user/slug pairs
- Detailed analysis (commented out by default)

**Usage:**
```bash
# Connect to your production database and run:
psql -h your-prod-host -U your-user -d your-database -f production_duplicate_check.sql
```

### 2. `check_duplicate_user_slug_pairs.sql`
A simpler SQL query that focuses on finding duplicates using GROUP BY and HAVING clauses.

### 3. `check_user_slug_duplicates.py`
A Python script that uses the application's database connection to check for duplicates. Provides detailed output and appropriate exit codes.

**Usage:**
```bash
# From the workspace root:
python3 check_user_slug_duplicates.py
```

## Expected Results

### If NO duplicates exist (expected):
- The main duplicate check query should return **0 rows**
- This indicates the unique constraint is working properly
- Data integrity is maintained

### If duplicates exist (problem):
- The query will return rows showing:
  - `owningUserId`: The user ID
  - `slug`: The duplicate slug
  - `duplicate_count`: How many duplicates exist
  - `listing_ids`: The IDs of the duplicate listings

## Understanding the Schema

From the Prisma schema (`schema.prisma`):

```prisma
model StoreListing {
  // ... other fields ...
  slug String                    // URL-friendly identifier
  owningUserId String            // User who owns this listing
  OwningUser   User   @relation(fields: [owningUserId], references: [id])
  
  @@unique([owningUserId, slug]) // This constraint prevents duplicates
}
```

The unique constraint ensures that:
- Each user can only have ONE listing with a specific slug
- Different users CAN have listings with the same slug
- The same user CANNOT have multiple listings with the same slug

## Database Migration History

The unique constraint was added in migration `20250318043016_update_store_submissions_format`:
```sql
CREATE UNIQUE INDEX "StoreListing_owningUserId_slug_key" ON "StoreListing"("owningUserId", "slug");
```

## Troubleshooting

If duplicates are found:

1. **Check constraint existence**: Verify the unique index exists in the database
2. **Review migration history**: Ensure the migration was applied successfully
3. **Data cleanup**: Remove duplicate entries (keeping the most recent or appropriate one)
4. **Re-apply constraint**: If missing, recreate the unique constraint

## Related Linear Issue

This addresses Linear issue **SECRT-1197**: "write a query to check that there's now duplicate user/slug pairs on prod"

The query verifies that the database maintains data integrity for the user/slug relationship in the StoreListing table.