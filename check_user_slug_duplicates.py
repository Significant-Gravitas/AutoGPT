#!/usr/bin/env python3
"""
Script to check for duplicate user/slug pairs in the StoreListing table.
This verifies that the unique constraint @@unique([owningUserId, slug]) is working properly.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'autogpt_platform', 'backend'))

from backend.data.db import DatabaseManager


async def check_duplicate_user_slug_pairs() -> Dict[str, Any]:
    """
    Check for duplicate user/slug pairs in the StoreListing table.
    
    Returns:
        Dict containing the results of the duplicate check
    """
    db = DatabaseManager()
    
    print("🔍 Checking for duplicate user/slug pairs in StoreListing table...")
    print("=" * 60)
    
    # Query 1: Find actual duplicates
    duplicate_query = """
        SELECT 
            "owningUserId",
            "slug",
            COUNT(*) as duplicate_count,
            string_agg("id", ', ') as listing_ids
        FROM "StoreListing"
        WHERE "isDeleted" = false
        GROUP BY "owningUserId", "slug"
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC, "owningUserId", "slug";
    """
    
    duplicates = await db.prisma.query_raw(duplicate_query)
    
    # Query 2: Get summary statistics
    stats_query = """
        SELECT 
            COUNT(*) as total_listings,
            COUNT(DISTINCT CONCAT("owningUserId", '|', "slug")) as unique_user_slug_pairs,
            (COUNT(*) - COUNT(DISTINCT CONCAT("owningUserId", '|', "slug"))) as potential_duplicates
        FROM "StoreListing"
        WHERE "isDeleted" = false;
    """
    
    stats = await db.prisma.query_raw(stats_query)
    
    # Query 3: Check the unique constraint exists
    constraint_query = """
        SELECT 
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE tablename = 'StoreListing' 
        AND indexdef LIKE '%owningUserId%slug%'
        AND indexdef LIKE '%UNIQUE%';
    """
    
    constraints = await db.prisma.query_raw(constraint_query)
    
    results = {
        'duplicates': duplicates,
        'stats': stats[0] if stats else {},
        'constraints': constraints,
        'has_duplicates': len(duplicates) > 0
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print the results in a readable format."""
    
    print("📊 SUMMARY STATISTICS")
    print("-" * 30)
    if results['stats']:
        stats = results['stats']
        print(f"Total listings (non-deleted): {stats.get('total_listings', 'N/A')}")
        print(f"Unique user/slug pairs: {stats.get('unique_user_slug_pairs', 'N/A')}")
        print(f"Potential duplicates: {stats.get('potential_duplicates', 'N/A')}")
    else:
        print("No statistics available")
    
    print("\n🔒 UNIQUE CONSTRAINTS")
    print("-" * 30)
    if results['constraints']:
        for constraint in results['constraints']:
            print(f"Index: {constraint.get('indexname')}")
            print(f"Definition: {constraint.get('indexdef')}")
    else:
        print("⚠️  No unique constraint found on (owningUserId, slug)")
    
    print(f"\n🚨 DUPLICATE CHECK RESULTS")
    print("-" * 30)
    if results['has_duplicates']:
        print(f"❌ FOUND {len(results['duplicates'])} DUPLICATE USER/SLUG PAIRS:")
        print()
        for dup in results['duplicates']:
            print(f"User ID: {dup.get('owningUserId')}")
            print(f"Slug: {dup.get('slug')}")
            print(f"Count: {dup.get('duplicate_count')}")
            print(f"Listing IDs: {dup.get('listing_ids')}")
            print("-" * 40)
    else:
        print("✅ NO DUPLICATE USER/SLUG PAIRS FOUND")
        print("The unique constraint is working properly!")


async def main():
    """Main function to run the duplicate check."""
    try:
        results = await check_duplicate_user_slug_pairs()
        print_results(results)
        
        # Return appropriate exit code
        if results['has_duplicates']:
            print("\n⚠️  ATTENTION: Duplicate user/slug pairs found!")
            print("This indicates the unique constraint may not be enforced properly.")
            sys.exit(1)
        else:
            print("\n✅ All checks passed! No duplicate user/slug pairs found.")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Error running duplicate check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())