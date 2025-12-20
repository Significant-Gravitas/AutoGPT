"""
Verification script to check scheduler data integrity after native auth migration.

This script verifies that all scheduled jobs reference valid users in the platform.User table.
It can also clean up orphaned schedules (schedules for users that no longer exist).

Usage:
    cd backend
    poetry run python scripts/verify_scheduler_data.py [options]

Options:
    --dry-run              Preview what would be cleaned up without making changes
    --cleanup              Actually remove orphaned schedules
    --database-url <url>   Database URL (overrides DATABASE_URL env var)

Examples:
    # Check for orphaned schedules (read-only)
    poetry run python scripts/verify_scheduler_data.py

    # Preview cleanup
    poetry run python scripts/verify_scheduler_data.py --dry-run

    # Actually clean up orphaned schedules
    poetry run python scripts/verify_scheduler_data.py --cleanup

Prerequisites:
    - Database must be accessible
    - Scheduler service must be running (for cleanup operations)
"""

import argparse
import asyncio
import logging
import os
import pickle
import sys
from datetime import datetime
from urllib.parse import parse_qs, urlparse, urlunparse, urlencode

from prisma import Prisma
from sqlalchemy import create_engine, text, MetaData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_schema_from_url(database_url: str) -> tuple[str, str]:
    """Extract schema from DATABASE_URL and return (schema, clean_url)."""
    parsed_url = urlparse(database_url)
    query_params = parse_qs(parsed_url.query)
    schema_list = query_params.pop("schema", None)
    schema = schema_list[0] if schema_list else "public"
    new_query = urlencode(query_params, doseq=True)
    new_parsed_url = parsed_url._replace(query=new_query)
    database_url_clean = str(urlunparse(new_parsed_url))
    return schema, database_url_clean


async def get_all_user_ids(db: Prisma) -> set[str]:
    """Get all user IDs from the platform.User table."""
    users = await db.user.find_many(select={"id": True})
    return {user.id for user in users}


def get_scheduler_jobs(db_url: str, schema: str) -> list[dict]:
    """Get all jobs from the apscheduler_jobs table."""
    engine = create_engine(db_url)
    jobs = []

    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(
            text(
                f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = :schema
                    AND table_name = 'apscheduler_jobs'
                )
                """
            ),
            {"schema": schema},
        )
        if not result.scalar():
            logger.warning(
                f"Table {schema}.apscheduler_jobs does not exist. "
                "Scheduler may not have been initialized yet."
            )
            return []

        # Get all jobs
        result = conn.execute(
            text(f'SELECT id, job_state FROM {schema}."apscheduler_jobs"')
        )

        for row in result:
            job_id = row[0]
            job_state = row[1]

            try:
                # APScheduler stores job state as pickled data
                job_data = pickle.loads(job_state)
                kwargs = job_data.get("kwargs", {})

                # Only process graph execution jobs (have user_id)
                if "user_id" in kwargs:
                    jobs.append(
                        {
                            "id": job_id,
                            "user_id": kwargs.get("user_id"),
                            "graph_id": kwargs.get("graph_id"),
                            "graph_version": kwargs.get("graph_version"),
                            "cron": kwargs.get("cron"),
                            "agent_name": kwargs.get("agent_name"),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to parse job {job_id}: {e}")

    return jobs


async def verify_scheduler_data(
    db: Prisma, db_url: str, schema: str
) -> tuple[list[dict], list[dict]]:
    """
    Verify scheduler data integrity.

    Returns:
        Tuple of (valid_jobs, orphaned_jobs)
    """
    logger.info("Fetching all users from platform.User...")
    user_ids = await get_all_user_ids(db)
    logger.info(f"Found {len(user_ids)} users in platform.User")

    logger.info("Fetching scheduled jobs from apscheduler_jobs...")
    jobs = get_scheduler_jobs(db_url, schema)
    logger.info(f"Found {len(jobs)} scheduled graph execution jobs")

    valid_jobs = []
    orphaned_jobs = []

    for job in jobs:
        if job["user_id"] in user_ids:
            valid_jobs.append(job)
        else:
            orphaned_jobs.append(job)

    return valid_jobs, orphaned_jobs


async def cleanup_orphaned_schedules(orphaned_jobs: list[dict], db_url: str, schema: str):
    """Remove orphaned schedules from the database."""
    if not orphaned_jobs:
        logger.info("No orphaned schedules to clean up")
        return

    engine = create_engine(db_url)

    with engine.connect() as conn:
        for job in orphaned_jobs:
            try:
                conn.execute(
                    text(f'DELETE FROM {schema}."apscheduler_jobs" WHERE id = :job_id'),
                    {"job_id": job["id"]},
                )
                logger.info(
                    f"Deleted orphaned schedule {job['id']} "
                    f"(user: {job['user_id']}, graph: {job['graph_id']})"
                )
            except Exception as e:
                logger.error(f"Failed to delete schedule {job['id']}: {e}")

        conn.commit()

    logger.info(f"Cleaned up {len(orphaned_jobs)} orphaned schedules")


async def main(dry_run: bool = False, cleanup: bool = False):
    """Run the verification."""
    logger.info("=" * 60)
    logger.info("Scheduler Data Verification Script")
    if dry_run:
        logger.info(">>> DRY RUN MODE - No changes will be made <<<")
    elif cleanup:
        logger.info(">>> CLEANUP MODE - Orphaned schedules will be removed <<<")
    else:
        logger.info(">>> VERIFY MODE - Read-only check <<<")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    # Get database URL
    db_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL or DIRECT_URL environment variable not set")
        sys.exit(1)

    schema, clean_db_url = _extract_schema_from_url(db_url)
    logger.info(f"Using schema: {schema}")

    db = Prisma()
    await db.connect()

    try:
        valid_jobs, orphaned_jobs = await verify_scheduler_data(db, clean_db_url, schema)

        # Report results
        logger.info("\n--- Verification Results ---")
        logger.info(f"Valid scheduled jobs: {len(valid_jobs)}")
        logger.info(f"Orphaned scheduled jobs: {len(orphaned_jobs)}")

        if orphaned_jobs:
            logger.warning("\n--- Orphaned Schedules (users not in platform.User) ---")
            for job in orphaned_jobs:
                logger.warning(
                    f"  Schedule ID: {job['id']}\n"
                    f"    User ID: {job['user_id']}\n"
                    f"    Graph ID: {job['graph_id']}\n"
                    f"    Cron: {job['cron']}\n"
                    f"    Agent: {job['agent_name'] or 'N/A'}"
                )

            if cleanup and not dry_run:
                logger.info("\n--- Cleaning up orphaned schedules ---")
                await cleanup_orphaned_schedules(orphaned_jobs, clean_db_url, schema)
            elif dry_run:
                logger.info(
                    f"\n[DRY RUN] Would delete {len(orphaned_jobs)} orphaned schedules"
                )
            else:
                logger.info(
                    "\nTo clean up orphaned schedules, run with --cleanup flag"
                )
        else:
            logger.info("\n✅ All scheduled jobs reference valid users!")

        # Summary
        logger.info("\n" + "=" * 60)
        if orphaned_jobs and cleanup and not dry_run:
            logger.info("Cleanup completed successfully!")
        else:
            logger.info("Verification completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise
    finally:
        await db.disconnect()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify scheduler data integrity after native auth migration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be cleaned up without making changes",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Actually remove orphaned schedules",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database URL (overrides DATABASE_URL env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override DATABASE_URL if provided via command line
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        os.environ["DIRECT_URL"] = args.database_url

    asyncio.run(main(dry_run=args.dry_run, cleanup=args.cleanup))
