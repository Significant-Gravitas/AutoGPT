#!/usr/bin/env python3
"""
Clean the test database by removing all data while preserving the schema.

Usage:
    poetry run python clean_test_db.py [--yes]
    
Options:
    --yes    Skip confirmation prompt
"""

import asyncio
import sys

from prisma import Prisma


async def main():
    db = Prisma()
    await db.connect()

    print("=" * 60)
    print("Cleaning Test Database")
    print("=" * 60)
    print()

    # Get initial counts
    user_count = await db.user.count()
    agent_count = await db.agentgraph.count()

    print(f"Current data: {user_count} users, {agent_count} agent graphs")

    if user_count == 0 and agent_count == 0:
        print("Database is already clean!")
        await db.disconnect()
        return

    # Check for --yes flag
    skip_confirm = "--yes" in sys.argv

    if not skip_confirm:
        response = input("\nDo you want to clean all data? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            await db.disconnect()
            return

    print("\nCleaning database...")

    # Delete in reverse order of dependencies
    tables = [
        ("UserNotificationBatch", db.usernotificationbatch),
        ("NotificationEvent", db.notificationevent),
        ("CreditRefundRequest", db.creditrefundrequest),
        ("StoreListingReview", db.storelistingreview),
        ("StoreListingVersion", db.storelistingversion),
        ("StoreListing", db.storelisting),
        ("AgentNodeExecutionInputOutput", db.agentnodeexecutioninputoutput),
        ("AgentNodeExecution", db.agentnodeexecution),
        ("AgentGraphExecution", db.agentgraphexecution),
        ("AgentNodeLink", db.agentnodelink),
        ("LibraryAgent", db.libraryagent),
        ("AgentPreset", db.agentpreset),
        ("IntegrationWebhook", db.integrationwebhook),
        ("AgentNode", db.agentnode),
        ("AgentGraph", db.agentgraph),
        ("AgentBlock", db.agentblock),
        ("APIKey", db.apikey),
        ("CreditTransaction", db.credittransaction),
        ("AnalyticsMetrics", db.analyticsmetrics),
        ("AnalyticsDetails", db.analyticsdetails),
        ("Profile", db.profile),
        ("UserOnboarding", db.useronboarding),
        ("User", db.user),
    ]

    for table_name, table in tables:
        try:
            count = await table.count()
            if count > 0:
                await table.delete_many()
                print(f"✓ Deleted {count} records from {table_name}")
        except Exception as e:
            print(f"⚠ Error cleaning {table_name}: {e}")

    # Refresh materialized views (they should be empty now)
    try:
        await db.execute_raw("SELECT refresh_store_materialized_views();")
        print("\n✓ Refreshed materialized views")
    except Exception as e:
        print(f"\n⚠ Could not refresh materialized views: {e}")

    await db.disconnect()

    print("\n" + "=" * 60)
    print("Database cleaned successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
