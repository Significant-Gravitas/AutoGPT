"""
Migration script for moving users from Supabase Auth to native FastAPI auth.

This script handles:
1. Marking existing users as migrated from Supabase
2. Sending password reset emails to migrated users
3. Tracking migration progress
4. Generating reports

Usage:
    # Dry run - see what would happen
    python -m backend.data.auth.migration --dry-run

    # Mark users as migrated (no emails)
    python -m backend.data.auth.migration --mark-migrated

    # Send password reset emails to migrated users
    python -m backend.data.auth.migration --send-emails --batch-size 100

    # Full migration (mark + send emails)
    python -m backend.data.auth.migration --full-migration --batch-size 100
"""

import argparse
import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prisma.models import User

from backend.data.auth.email_service import get_auth_email_service
from backend.data.auth.magic_links import create_password_reset_link
from backend.data.db import connect, disconnect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.total_users = 0
        self.already_migrated = 0
        self.marked_migrated = 0
        self.emails_sent = 0
        self.emails_failed = 0
        self.oauth_users_skipped = 0
        self.errors = []

    def __str__(self):
        return f"""
Migration Statistics:
---------------------
Total users processed: {self.total_users}
Already migrated: {self.already_migrated}
Newly marked as migrated: {self.marked_migrated}
Password reset emails sent: {self.emails_sent}
Email failures: {self.emails_failed}
OAuth users skipped: {self.oauth_users_skipped}
Errors: {len(self.errors)}
"""


async def get_users_to_migrate(
    batch_size: int = 100,
    offset: int = 0,
) -> list[User]:
    """
    Get users that need to be migrated.

    Returns users where:
    - authProvider is "supabase" or NULL
    - migratedFromSupabase is False or NULL
    - passwordHash is NULL (they haven't set a native password)
    """
    users = await User.prisma().find_many(
        where={
            "OR": [
                {"authProvider": "supabase"},
                {"authProvider": None},
            ],
            "migratedFromSupabase": False,
            "passwordHash": None,
        },
        take=batch_size,
        skip=offset,
        order={"createdAt": "asc"},
    )
    return users


async def get_migrated_users_needing_email(
    batch_size: int = 100,
    offset: int = 0,
) -> list[User]:
    """
    Get migrated users who haven't set their password yet.

    These users need a password reset email.
    """
    users = await User.prisma().find_many(
        where={
            "migratedFromSupabase": True,
            "passwordHash": None,
            "authProvider": {"not": "google"},  # Skip OAuth users
        },
        take=batch_size,
        skip=offset,
        order={"createdAt": "asc"},
    )
    return users


async def mark_user_as_migrated(user: User, dry_run: bool = False) -> bool:
    """
    Mark a user as migrated from Supabase.

    Sets migratedFromSupabase=True and authProvider="supabase".
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would mark user {user.id} ({user.email}) as migrated")
        return True

    try:
        await User.prisma().update(
            where={"id": user.id},
            data={
                "migratedFromSupabase": True,
                "authProvider": "supabase",
            },
        )
        logger.info(f"Marked user {user.id} ({user.email}) as migrated")
        return True
    except Exception as e:
        logger.error(f"Failed to mark user {user.id} as migrated: {e}")
        return False


async def send_migration_email(
    user: User,
    email_service,
    dry_run: bool = False,
) -> bool:
    """
    Send a password reset email to a migrated user.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would send migration email to {user.email}")
        return True

    try:
        token = await create_password_reset_link(user.email, user.id)
        success = email_service.send_migrated_user_password_reset(user.email, token)

        if success:
            logger.info(f"Sent migration email to {user.email}")
        else:
            logger.warning(f"Failed to send migration email to {user.email}")

        return success
    except Exception as e:
        logger.error(f"Error sending migration email to {user.email}: {e}")
        return False


async def run_migration(
    mark_migrated: bool = False,
    send_emails: bool = False,
    batch_size: int = 100,
    dry_run: bool = False,
    email_delay: float = 0.5,  # Delay between emails to avoid rate limiting
) -> MigrationStats:
    """
    Run the migration process.

    Args:
        mark_migrated: Mark users as migrated from Supabase
        send_emails: Send password reset emails to migrated users
        batch_size: Number of users to process at a time
        dry_run: If True, don't make any changes
        email_delay: Seconds to wait between sending emails

    Returns:
        MigrationStats with results
    """
    stats = MigrationStats()
    email_service = get_auth_email_service() if send_emails else None

    # Phase 1: Mark users as migrated
    if mark_migrated:
        logger.info("Phase 1: Marking users as migrated...")
        offset = 0

        while True:
            users = await get_users_to_migrate(batch_size, offset)
            if not users:
                break

            for user in users:
                stats.total_users += 1

                # Skip OAuth users
                if user.authProvider == "google":
                    stats.oauth_users_skipped += 1
                    continue

                success = await mark_user_as_migrated(user, dry_run)
                if success:
                    stats.marked_migrated += 1
                else:
                    stats.errors.append(f"Failed to mark {user.email}")

            offset += batch_size
            logger.info(f"Processed {offset} users...")

    # Phase 2: Send password reset emails
    if send_emails:
        logger.info("Phase 2: Sending password reset emails...")
        offset = 0

        while True:
            users = await get_migrated_users_needing_email(batch_size, offset)
            if not users:
                break

            for user in users:
                stats.total_users += 1

                success = await send_migration_email(user, email_service, dry_run)
                if success:
                    stats.emails_sent += 1
                else:
                    stats.emails_failed += 1
                    stats.errors.append(f"Failed to email {user.email}")

                # Rate limiting
                if not dry_run and email_delay > 0:
                    await asyncio.sleep(email_delay)

            offset += batch_size
            logger.info(f"Processed {offset} users for email...")

    return stats


async def generate_migration_report(output_path: Optional[str] = None) -> str:
    """
    Generate a CSV report of all users and their migration status.
    """
    if output_path is None:
        output_path = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    users = await User.prisma().find_many(
        order={"createdAt": "asc"},
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_id",
            "email",
            "auth_provider",
            "migrated_from_supabase",
            "has_password",
            "email_verified",
            "created_at",
            "needs_action",
        ])

        for user in users:
            needs_action = (
                user.migratedFromSupabase
                and user.passwordHash is None
                and user.authProvider != "google"
            )

            writer.writerow([
                user.id,
                user.email,
                user.authProvider or "unknown",
                user.migratedFromSupabase,
                user.passwordHash is not None,
                user.emailVerified,
                user.createdAt.isoformat() if user.createdAt else "",
                "YES" if needs_action else "NO",
            ])

    logger.info(f"Report saved to {output_path}")
    return output_path


async def count_migration_status():
    """
    Get counts of users in different migration states.
    """
    total = await User.prisma().count()

    already_native = await User.prisma().count(
        where={"authProvider": "password", "passwordHash": {"not": None}}
    )

    oauth_users = await User.prisma().count(
        where={"authProvider": "google"}
    )

    migrated_pending = await User.prisma().count(
        where={
            "migratedFromSupabase": True,
            "passwordHash": None,
            "authProvider": {"not": "google"},
        }
    )

    not_migrated = await User.prisma().count(
        where={
            "migratedFromSupabase": False,
            "authProvider": {"in": ["supabase", None]},
        }
    )

    return {
        "total": total,
        "already_native": already_native,
        "oauth_users": oauth_users,
        "migrated_pending_password": migrated_pending,
        "not_yet_migrated": not_migrated,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate users from Supabase Auth to native FastAPI auth"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make any changes, just show what would happen",
    )
    parser.add_argument(
        "--mark-migrated",
        action="store_true",
        help="Mark existing Supabase users as migrated",
    )
    parser.add_argument(
        "--send-emails",
        action="store_true",
        help="Send password reset emails to migrated users",
    )
    parser.add_argument(
        "--full-migration",
        action="store_true",
        help="Run full migration (mark + send emails)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of users to process at a time (default: 100)",
    )
    parser.add_argument(
        "--email-delay",
        type=float,
        default=0.5,
        help="Seconds to wait between emails (default: 0.5)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a CSV report of migration status",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current migration status counts",
    )

    args = parser.parse_args()

    # Connect to database
    await connect()

    try:
        if args.status:
            counts = await count_migration_status()
            print("\nMigration Status:")
            print("-" * 40)
            print(f"Total users: {counts['total']}")
            print(f"Already using native auth: {counts['already_native']}")
            print(f"OAuth users (Google): {counts['oauth_users']}")
            print(f"Migrated, pending password: {counts['migrated_pending_password']}")
            print(f"Not yet migrated: {counts['not_yet_migrated']}")
            return

        if args.report:
            await generate_migration_report()
            return

        if args.full_migration:
            args.mark_migrated = True
            args.send_emails = True

        if not args.mark_migrated and not args.send_emails:
            parser.print_help()
            print("\nError: Must specify --mark-migrated, --send-emails, --full-migration, --report, or --status")
            return

        if args.dry_run:
            logger.info("=" * 50)
            logger.info("DRY RUN MODE - No changes will be made")
            logger.info("=" * 50)

        stats = await run_migration(
            mark_migrated=args.mark_migrated,
            send_emails=args.send_emails,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            email_delay=args.email_delay,
        )

        print(stats)

        if stats.errors:
            print("\nErrors encountered:")
            for error in stats.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(stats.errors) > 10:
                print(f"  ... and {len(stats.errors) - 10} more")

    finally:
        await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
