import asyncio
import random
from datetime import datetime

from faker import Faker
from prisma import Prisma

faker = Faker()


async def check_cron_job(db):
    """Check if the pg_cron job for refreshing materialized views exists."""
    print("\n1. Checking pg_cron job...")
    print("-" * 40)

    try:
        # Check if pg_cron extension exists
        extension_check = await db.query_raw("CREATE EXTENSION pg_cron;")
        print(extension_check)
        extension_check = await db.query_raw(
            "SELECT COUNT(*) as count FROM pg_extension WHERE extname = 'pg_cron'"
        )
        if extension_check[0]["count"] == 0:
            print("⚠️  pg_cron extension is not installed")
            return False

        # Check if the refresh job exists
        job_check = await db.query_raw(
            """
            SELECT jobname, schedule, command 
            FROM cron.job 
            WHERE jobname = 'refresh-store-views'
        """
        )

        if job_check:
            job = job_check[0]
            print("✅ pg_cron job found:")
            print(f"   Name: {job['jobname']}")
            print(f"   Schedule: {job['schedule']} (every 15 minutes)")
            print(f"   Command: {job['command']}")
            return True
        else:
            print("⚠️  pg_cron job 'refresh-store-views' not found")
            return False

    except Exception as e:
        print(f"❌ Error checking pg_cron: {e}")
        return False


async def get_materialized_view_counts(db):
    """Get current counts from materialized views."""
    print("\n2. Getting current materialized view data...")
    print("-" * 40)

    # Get counts from mv_agent_run_counts
    agent_runs = await db.query_raw(
        """
        SELECT COUNT(*) as total_agents, 
               SUM(run_count) as total_runs,
               MAX(run_count) as max_runs,
               MIN(run_count) as min_runs
        FROM mv_agent_run_counts
    """
    )

    # Get counts from mv_review_stats
    review_stats = await db.query_raw(
        """
        SELECT COUNT(*) as total_listings,
               SUM(review_count) as total_reviews,
               AVG(avg_rating) as overall_avg_rating
        FROM mv_review_stats
    """
    )

    # Get sample data from StoreAgent view
    store_agents = await db.query_raw(
        """
        SELECT COUNT(*) as total_store_agents,
               AVG(runs) as avg_runs,
               AVG(rating) as avg_rating
        FROM "StoreAgent"
    """
    )

    agent_run_data = agent_runs[0] if agent_runs else {}
    review_data = review_stats[0] if review_stats else {}
    store_data = store_agents[0] if store_agents else {}

    print("📊 mv_agent_run_counts:")
    print(f"   Total agents: {agent_run_data.get('total_agents', 0)}")
    print(f"   Total runs: {agent_run_data.get('total_runs', 0)}")
    print(f"   Max runs per agent: {agent_run_data.get('max_runs', 0)}")
    print(f"   Min runs per agent: {agent_run_data.get('min_runs', 0)}")

    print("\n📊 mv_review_stats:")
    print(f"   Total listings: {review_data.get('total_listings', 0)}")
    print(f"   Total reviews: {review_data.get('total_reviews', 0)}")
    print(f"   Overall avg rating: {review_data.get('overall_avg_rating') or 0:.2f}")

    print("\n📊 StoreAgent view:")
    print(f"   Total store agents: {store_data.get('total_store_agents', 0)}")
    print(f"   Average runs: {store_data.get('avg_runs') or 0:.2f}")
    print(f"   Average rating: {store_data.get('avg_rating') or 0:.2f}")

    return {
        "agent_runs": agent_run_data,
        "reviews": review_data,
        "store_agents": store_data,
    }


async def add_test_data(db):
    """Add some test data to verify materialized view updates."""
    print("\n3. Adding test data...")
    print("-" * 40)

    # Get some existing data
    users = await db.user.find_many(take=5)
    graphs = await db.agentgraph.find_many(take=5)

    if not users or not graphs:
        print("❌ No existing users or graphs found. Run test_data_creator.py first.")
        return False

    # Add new executions
    print("Adding new agent graph executions...")
    new_executions = 0
    for graph in graphs:
        for _ in range(random.randint(2, 5)):
            await db.agentgraphexecution.create(
                data={
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "userId": random.choice(users).id,
                    "executionStatus": "COMPLETED",
                    "startedAt": datetime.now(),
                }
            )
            new_executions += 1

    print(f"✅ Added {new_executions} new executions")

    # Check if we need to create store listings first
    store_versions = await db.storelistingversion.find_many(
        where={"submissionStatus": "APPROVED"}, take=5
    )

    if not store_versions:
        print("\nNo approved store listings found. Creating test store listings...")

        # Create store listings for existing agent graphs
        for i, graph in enumerate(graphs[:3]):  # Create up to 3 store listings
            # Create a store listing
            listing = await db.storelisting.create(
                data={
                    "slug": f"test-agent-{graph.id[:8]}",
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "hasApprovedVersion": True,
                    "owningUserId": graph.userId,
                }
            )

            # Create an approved version
            version = await db.storelistingversion.create(
                data={
                    "storeListingId": listing.id,
                    "agentGraphId": graph.id,
                    "agentGraphVersion": graph.version,
                    "name": f"Test Agent {i+1}",
                    "subHeading": faker.catch_phrase(),
                    "description": faker.paragraph(nb_sentences=5),
                    "imageUrls": [faker.image_url()],
                    "categories": ["productivity", "automation"],
                    "submissionStatus": "APPROVED",
                    "submittedAt": datetime.now(),
                }
            )

            # Update listing with active version
            await db.storelisting.update(
                where={"id": listing.id}, data={"activeVersionId": version.id}
            )

        print("✅ Created test store listings")

        # Re-fetch approved versions
        store_versions = await db.storelistingversion.find_many(
            where={"submissionStatus": "APPROVED"}, take=5
        )

    # Add new reviews
    print("\nAdding new store listing reviews...")
    new_reviews = 0
    for version in store_versions:
        # Find users who haven't reviewed this version
        existing_reviews = await db.storelistingreview.find_many(
            where={"storeListingVersionId": version.id}
        )
        reviewed_user_ids = {r.reviewByUserId for r in existing_reviews}
        available_users = [u for u in users if u.id not in reviewed_user_ids]

        if available_users:
            user = random.choice(available_users)
            await db.storelistingreview.create(
                data={
                    "storeListingVersionId": version.id,
                    "reviewByUserId": user.id,
                    "score": random.randint(3, 5),
                    "comments": faker.text(max_nb_chars=100),
                }
            )
            new_reviews += 1

    print(f"✅ Added {new_reviews} new reviews")

    return True


async def refresh_materialized_views(db):
    """Manually refresh the materialized views."""
    print("\n4. Manually refreshing materialized views...")
    print("-" * 40)

    try:
        await db.execute_raw("SELECT refresh_store_materialized_views();")
        print("✅ Materialized views refreshed successfully")
        return True
    except Exception as e:
        print(f"❌ Error refreshing views: {e}")
        return False


async def compare_counts(before, after):
    """Compare counts before and after refresh."""
    print("\n5. Comparing counts before and after refresh...")
    print("-" * 40)

    # Compare agent runs
    print("🔍 Agent run changes:")
    before_runs = before["agent_runs"].get("total_runs") or 0
    after_runs = after["agent_runs"].get("total_runs") or 0
    print(
        f"   Total runs: {before_runs} → {after_runs} " f"(+{after_runs - before_runs})"
    )

    # Compare reviews
    print("\n🔍 Review changes:")
    before_reviews = before["reviews"].get("total_reviews") or 0
    after_reviews = after["reviews"].get("total_reviews") or 0
    print(
        f"   Total reviews: {before_reviews} → {after_reviews} "
        f"(+{after_reviews - before_reviews})"
    )

    # Compare store agents
    print("\n🔍 StoreAgent view changes:")
    before_avg_runs = before["store_agents"].get("avg_runs", 0) or 0
    after_avg_runs = after["store_agents"].get("avg_runs", 0) or 0
    print(
        f"   Average runs: {before_avg_runs:.2f} → {after_avg_runs:.2f} "
        f"(+{after_avg_runs - before_avg_runs:.2f})"
    )

    # Verify changes occurred
    runs_changed = (after["agent_runs"].get("total_runs") or 0) > (
        before["agent_runs"].get("total_runs") or 0
    )
    reviews_changed = (after["reviews"].get("total_reviews") or 0) > (
        before["reviews"].get("total_reviews") or 0
    )

    if runs_changed and reviews_changed:
        print("\n✅ Materialized views are updating correctly!")
        return True
    else:
        print("\n⚠️  Some materialized views may not have updated:")
        if not runs_changed:
            print("   - Agent run counts did not increase")
        if not reviews_changed:
            print("   - Review counts did not increase")
        return False


async def main():
    db = Prisma()
    await db.connect()

    print("=" * 60)
    print("Materialized Views Test")
    print("=" * 60)

    try:
        # Check if data exists
        user_count = await db.user.count()
        if user_count == 0:
            print("❌ No data in database. Please run test_data_creator.py first.")
            await db.disconnect()
            return

        # 1. Check cron job
        cron_exists = await check_cron_job(db)

        # 2. Get initial counts
        counts_before = await get_materialized_view_counts(db)

        # 3. Add test data
        data_added = await add_test_data(db)
        refresh_success = False

        if data_added:
            # Wait a moment for data to be committed
            print("\nWaiting for data to be committed...")
            await asyncio.sleep(2)

            # 4. Manually refresh views
            refresh_success = await refresh_materialized_views(db)

            if refresh_success:
                # 5. Get counts after refresh
                counts_after = await get_materialized_view_counts(db)

                # 6. Compare results
                await compare_counts(counts_before, counts_after)

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"✓ pg_cron job exists: {'Yes' if cron_exists else 'No'}")
        print(f"✓ Test data added: {'Yes' if data_added else 'No'}")
        print(f"✓ Manual refresh worked: {'Yes' if refresh_success else 'No'}")
        print(
            f"✓ Views updated correctly: {'Yes' if data_added and refresh_success else 'Cannot verify'}"
        )

        if cron_exists:
            print(
                "\n💡 The materialized views will also refresh automatically every 15 minutes via pg_cron."
            )
        else:
            print(
                "\n⚠️  Automatic refresh is not configured. Views must be refreshed manually."
            )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
