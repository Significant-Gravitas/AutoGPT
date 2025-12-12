"""
Script to backfill embeddings for existing store listing versions.

This script should be run after the migration to add the embedding column
to populate embeddings for all existing store listing versions.

Usage:
    poetry run python -m backend.server.v2.store.backfill_embeddings
    poetry run python -m backend.server.v2.store.backfill_embeddings --dry-run
    poetry run python -m backend.server.v2.store.backfill_embeddings --batch-size 25
"""

import argparse
import asyncio
import logging
import sys

from backend.data.db import connect, disconnect, query_raw_with_schema
from backend.integrations.embeddings import create_search_text, get_embedding_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default batch size for processing
DEFAULT_BATCH_SIZE = 50

# Delay between batches to avoid rate limits (seconds)
BATCH_DELAY_SECONDS = 1.0


async def backfill_embeddings(
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """
    Backfill embeddings for all store listing versions without embeddings.

    Args:
        dry_run: If True, don't make any changes, just report what would be done.
        batch_size: Number of versions to process in each batch.

    Returns:
        Tuple of (processed_count, error_count)
    """
    await connect()

    try:
        embedding_service = get_embedding_service()

        # Get all versions without embeddings
        versions = await query_raw_with_schema(
            """
            SELECT id, name, "subHeading", description
            FROM {schema_prefix}"StoreListingVersion"
            WHERE embedding IS NULL
            ORDER BY "createdAt" DESC
            """
        )

        total = len(versions)
        logger.info(f"Found {total} versions without embeddings")

        if dry_run:
            logger.info("Dry run mode - no changes will be made")
            return (0, 0)

        if total == 0:
            logger.info("No versions need embeddings")
            return (0, 0)

        processed = 0
        errors = 0

        for i in range(0, total, batch_size):
            batch = versions[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            for version in batch:
                version_id = version["id"]
                try:
                    search_text = create_search_text(
                        version["name"] or "",
                        version["subHeading"] or "",
                        version["description"] or "",
                    )

                    if not search_text:
                        logger.warning(f"Skipping {version_id} - no searchable text")
                        continue

                    embedding = await embedding_service.generate_embedding(search_text)
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                    await query_raw_with_schema(
                        """
                        UPDATE {schema_prefix}"StoreListingVersion"
                        SET embedding = $1::vector
                        WHERE id = $2
                        """,
                        embedding_str,
                        version_id,
                    )

                    processed += 1

                except Exception as e:
                    logger.error(f"Error processing {version_id}: {e}")
                    errors += 1

            logger.info(f"Progress: {processed}/{total} processed, {errors} errors")

            # Rate limit: wait between batches to avoid hitting API limits
            if i + batch_size < total:
                await asyncio.sleep(BATCH_DELAY_SECONDS)

        logger.info(f"Backfill complete: {processed} processed, {errors} errors")
        return (processed, errors)

    finally:
        await disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for store listing versions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't make any changes, just report what would be done",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of versions to process in each batch (default: {DEFAULT_BATCH_SIZE})",
    )

    args = parser.parse_args()

    try:
        processed, errors = asyncio.run(
            backfill_embeddings(dry_run=args.dry_run, batch_size=args.batch_size)
        )

        if errors > 0:
            logger.warning(f"Completed with {errors} errors")
            sys.exit(1)
        else:
            logger.info("Completed successfully")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
