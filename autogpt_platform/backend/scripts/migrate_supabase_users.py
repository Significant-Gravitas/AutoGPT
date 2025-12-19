"""
Migration script to copy password hashes from Supabase auth.users to platform.User.

This script should be run BEFORE removing Supabase services to preserve user credentials.
It copies bcrypt password hashes from Supabase's auth.users table to the platform.User table,
allowing users to continue using their existing passwords after the migration.

Usage:
    cd backend
    poetry run python scripts/migrate_supabase_users.py [options]

Options:
    --dry-run              Preview what would be migrated without making changes
    --database-url <url>   Database URL (overrides DATABASE_URL env var)

Examples:
    # Using environment variable
    poetry run python scripts/migrate_supabase_users.py --dry-run

    # Using explicit database URL
    poetry run python scripts/migrate_supabase_users.py \
        --database-url "postgresql://user:pass@host:5432/db?schema=platform"

Prerequisites:
    - Supabase services must be running (auth.users table must exist)
    - Database migration 'add_native_auth' must be applied first
    - Either DATABASE_URL env var or --database-url must be provided
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from prisma import Prisma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def migrate_credentials(db: Prisma) -> int:
    """
    Copy bcrypt password hashes from auth.users to platform.User.

    Returns the number of users updated.
    """
    logger.info("Migrating user credentials from auth.users to platform.User...")

    result = await db.execute_raw(
        """
        UPDATE platform."User" u
        SET
            "passwordHash" = a.encrypted_password,
            "emailVerified" = (a.email_confirmed_at IS NOT NULL)
        FROM auth.users a
        WHERE u.id::text = a.id::text
        AND a.encrypted_password IS NOT NULL
        AND u."passwordHash" IS NULL
        """
    )

    logger.info(f"Updated {result} users with credentials")
    return result


async def migrate_google_oauth_users(db: Prisma) -> int:
    """
    Copy Google OAuth user IDs from auth.users to platform.User.

    Returns the number of users updated.
    """
    logger.info("Migrating Google OAuth users from auth.users to platform.User...")

    result = await db.execute_raw(
        """
        UPDATE platform."User" u
        SET "googleId" = (a.raw_app_meta_data->>'provider_id')::text
        FROM auth.users a
        WHERE u.id::text = a.id::text
        AND a.raw_app_meta_data->>'provider' = 'google'
        AND a.raw_app_meta_data->>'provider_id' IS NOT NULL
        AND u."googleId" IS NULL
        """
    )

    logger.info(f"Updated {result} users with Google OAuth IDs")
    return result


async def get_migration_stats(db: Prisma) -> dict:
    """Get statistics about the migration."""
    # Count users in platform.User
    platform_users = await db.user.count()

    # Count users with credentials (not null)
    users_with_credentials = await db.user.count(
        where={"passwordHash": {"not": None}}  # type: ignore
    )

    # Count users with Google OAuth (not null)
    users_with_google = await db.user.count(
        where={"googleId": {"not": None}}  # type: ignore
    )

    # Count users without any auth method
    users_without_auth = await db.user.count(
        where={"passwordHash": None, "googleId": None}
    )

    return {
        "total_platform_users": platform_users,
        "users_with_credentials": users_with_credentials,
        "users_with_google_oauth": users_with_google,
        "users_without_auth": users_without_auth,
    }


async def verify_auth_users_exist(db: Prisma) -> bool:
    """Check if auth.users table exists and has data."""
    try:
        result = await db.query_raw("SELECT COUNT(*) as count FROM auth.users")
        count = result[0]["count"] if result else 0
        logger.info(f"Found {count} users in auth.users table")
        return count > 0
    except Exception as e:
        logger.error(f"Cannot access auth.users table: {e}")
        return False


async def preview_migration(db: Prisma) -> dict:
    """Preview what would be migrated without making changes."""
    logger.info("Previewing migration (dry-run mode)...")

    # Count users that would have credentials migrated
    credentials_preview = await db.query_raw(
        """
        SELECT COUNT(*) as count
        FROM platform."User" u
        JOIN auth.users a ON u.id::text = a.id::text
        WHERE a.encrypted_password IS NOT NULL
        AND u."passwordHash" IS NULL
        """
    )
    credentials_to_migrate = (
        credentials_preview[0]["count"] if credentials_preview else 0
    )

    # Count users that would have Google OAuth migrated
    google_preview = await db.query_raw(
        """
        SELECT COUNT(*) as count
        FROM platform."User" u
        JOIN auth.users a ON u.id::text = a.id::text
        WHERE a.raw_app_meta_data->>'provider' = 'google'
        AND a.raw_app_meta_data->>'provider_id' IS NOT NULL
        AND u."googleId" IS NULL
        """
    )
    google_to_migrate = google_preview[0]["count"] if google_preview else 0

    return {
        "credentials_to_migrate": credentials_to_migrate,
        "google_oauth_to_migrate": google_to_migrate,
    }


async def main(dry_run: bool = False):
    """Run the migration."""
    logger.info("=" * 60)
    logger.info("Supabase User Migration Script")
    if dry_run:
        logger.info(">>> DRY RUN MODE - No changes will be made <<<")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")

    db = Prisma()
    await db.connect()

    try:
        # Check if auth.users exists
        if not await verify_auth_users_exist(db):
            logger.error(
                "Cannot find auth.users table or it's empty. "
                "Make sure Supabase is running and has users."
            )
            sys.exit(1)

        # Get stats before migration
        logger.info("\n--- Current State ---")
        stats_before = await get_migration_stats(db)
        for key, value in stats_before.items():
            logger.info(f"  {key}: {value}")

        if dry_run:
            # Preview mode - show what would be migrated
            logger.info("\n--- Preview (would be migrated) ---")
            preview = await preview_migration(db)
            logger.info(
                f"  Credentials to migrate: {preview['credentials_to_migrate']}"
            )
            logger.info(
                f"  Google OAuth IDs to migrate: {preview['google_oauth_to_migrate']}"
            )
            logger.info("\n" + "=" * 60)
            logger.info("Dry run complete. Run without --dry-run to perform migration.")
            logger.info("=" * 60)
        else:
            # Run actual migrations
            logger.info("\n--- Running Migration ---")
            credentials_migrated = await migrate_credentials(db)
            google_migrated = await migrate_google_oauth_users(db)

            # Get stats after migration
            logger.info("\n--- After Migration ---")
            stats_after = await get_migration_stats(db)
            for key, value in stats_after.items():
                logger.info(f"  {key}: {value}")

            # Summary
            logger.info("\n--- Summary ---")
            logger.info(f"Credentials migrated: {credentials_migrated}")
            logger.info(f"Google OAuth IDs migrated: {google_migrated}")
            logger.info(
                f"Users still without auth: {stats_after['users_without_auth']} "
                "(these may be OAuth users from other providers)"
            )

            logger.info("\n" + "=" * 60)
            logger.info("Migration completed successfully!")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        await db.disconnect()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate user auth data from Supabase to native auth"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be migrated without making changes",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database URL (overrides DATABASE_URL env var)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import os

    args = parse_args()

    # Override DATABASE_URL if provided via command line
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url
        os.environ["DIRECT_URL"] = args.database_url

    asyncio.run(main(dry_run=args.dry_run))
