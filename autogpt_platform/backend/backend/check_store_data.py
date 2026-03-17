#!/usr/bin/env python3
"""Check store-related data in the database."""

import asyncio

from prisma import Prisma

from backend.data.db import query_raw_with_schema


async def check_store_data(db):
    """Check what store data exists in the database."""

    print("============================================================")
    print("Store Data Check")
    print("============================================================")

    # Check store listings
    print("\n1. Store Listings:")
    print("-" * 40)
    listings = await db.storelisting.find_many()
    print(f"Total store listings: {len(listings)}")

    if listings:
        for listing in listings[:5]:
            print(f"\nListing ID: {listing.id}")
            print(f"  Name: {listing.name}")
            print(f"  Status: {listing.status}")
            print(f"  Slug: {listing.slug}")

    # Check store listing versions
    print("\n\n2. Store Listing Versions:")
    print("-" * 40)
    versions = await db.storelistingversion.find_many(include={"StoreListing": True})
    print(f"Total store listing versions: {len(versions)}")

    # Group by submission status
    status_counts = {}
    for version in versions:
        status = version.submissionStatus
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nVersions by status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    # Show approved versions
    approved_versions = [v for v in versions if v.submissionStatus == "APPROVED"]
    print(f"\nApproved versions: {len(approved_versions)}")
    if approved_versions:
        for version in approved_versions[:5]:
            print(f"\n  Version ID: {version.id}")
            print(f"    Listing: {version.StoreListing.name}")
            print(f"    Version: {version.version}")

    # Check store listing reviews
    print("\n\n3. Store Listing Reviews:")
    print("-" * 40)
    reviews = await db.storelistingreview.find_many(
        include={"StoreListingVersion": {"include": {"StoreListing": True}}}
    )
    print(f"Total reviews: {len(reviews)}")

    if reviews:
        # Calculate average rating
        total_score = sum(r.score for r in reviews)
        avg_score = total_score / len(reviews) if reviews else 0
        print(f"Average rating: {avg_score:.2f}")

        # Show sample reviews
        print("\nSample reviews:")
        for review in reviews[:3]:
            print(f"\n  Review for: {review.StoreListingVersion.StoreListing.name}")
            print(f"    Score: {review.score}")
            print(f"    Comments: {review.comments[:100]}...")

    # Check StoreAgent view data
    print("\n\n4. StoreAgent View Data:")
    print("-" * 40)

    # Query the StoreAgent view
    query = """
    SELECT 
        sa.listing_id,
        sa.slug,
        sa.agent_name,
        sa.description,
        sa.featured,
        sa.runs,
        sa.rating,
        sa.creator_username,
        sa.categories,
        sa.updated_at
    FROM {schema_prefix}"StoreAgent" sa
    LIMIT 10;
    """

    store_agents = await query_raw_with_schema(query)
    print(f"Total store agents in view: {len(store_agents)}")

    if store_agents:
        for agent in store_agents[:5]:
            print(f"\nStore Agent: {agent['agent_name']}")
            print(f"  Slug: {agent['slug']}")
            print(f"  Runs: {agent['runs']}")
            print(f"  Rating: {agent['rating']}")
            print(f"  Creator: {agent['creator_username']}")

    # Check the underlying data that should populate StoreAgent
    print("\n\n5. Data that should populate StoreAgent view:")
    print("-" * 40)

    # Check for any APPROVED store listing versions
    query = """
    SELECT COUNT(*) as count
    FROM {schema_prefix}"StoreListingVersion"
    WHERE "submissionStatus" = 'APPROVED'
    """

    result = await query_raw_with_schema(query)
    approved_count = result[0]["count"] if result else 0
    print(f"Approved store listing versions: {approved_count}")

    # Check for store listings with hasApprovedVersion = true
    query = """
    SELECT COUNT(*) as count
    FROM {schema_prefix}"StoreListing"
    WHERE "hasApprovedVersion" = true AND "isDeleted" = false
    """

    result = await query_raw_with_schema(query)
    has_approved_count = result[0]["count"] if result else 0
    print(f"Store listings with approved versions: {has_approved_count}")

    # Check agent graph executions
    query = """
    SELECT COUNT(DISTINCT "agentGraphId") as unique_agents,
           COUNT(*) as total_executions
    FROM {schema_prefix}"AgentGraphExecution"
    """

    result = await query_raw_with_schema(query)
    if result:
        print("\nAgent Graph Executions:")
        print(f"  Unique agents with executions: {result[0]['unique_agents']}")
        print(f"  Total executions: {result[0]['total_executions']}")


async def main():
    """Main function."""
    db = Prisma()
    await db.connect()

    try:
        await check_store_data(db)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
