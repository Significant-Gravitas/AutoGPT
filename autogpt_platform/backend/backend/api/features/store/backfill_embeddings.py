#!/usr/bin/env python3
"""
CLI script to backfill embeddings for store agents.

Usage:
    poetry run python -m backend.api.features.store.backfill_embeddings [--batch-size N]
"""

import argparse
import asyncio
import logging
import sys

from backend.api.features.store.embeddings import (
    backfill_missing_embeddings,
    get_embedding_stats,
)
from backend.data import db

logger = logging.getLogger(__name__)


async def main(batch_size: int = 100) -> int:
    """Run the backfill process - processes ALL missing embeddings in batches."""
    await db.connect()

    try:
        stats = await get_embedding_stats()

        # Check for error from get_embedding_stats() first
        if "error" in stats:
            logger.error(f"Failed to get embedding stats: {stats['error']}")
            return 1

        logger.info(
            f"Current coverage: {stats['with_embeddings']}/{stats['total_approved']} "
            f"({stats['coverage_percent']}%)"
        )

        if stats["without_embeddings"] == 0:
            logger.info("All agents have embeddings - nothing to backfill")
            return 0

        logger.info(
            f"Backfilling {stats['without_embeddings']} missing embeddings "
            f"(batch size: {batch_size})"
        )

        total_processed = 0
        total_success = 0
        total_failed = 0

        while True:
            result = await backfill_missing_embeddings(batch_size=batch_size)
            if result["processed"] == 0:
                break

            total_processed += result["processed"]
            total_success += result["success"]
            total_failed += result["failed"]

            logger.info(
                f"Batch complete: {result['success']}/{result['processed']} succeeded"
            )

            await asyncio.sleep(1)

        # Final stats
        stats = await get_embedding_stats()
        logger.info(
            f"Backfill complete: {total_success}/{total_processed} succeeded, "
            f"{total_failed} failed"
        )
        if "error" not in stats:
            logger.info(f"Final coverage: {stats['coverage_percent']}%")
        else:
            logger.warning("Could not retrieve final coverage stats")

        return 0 if total_failed == 0 else 1

    finally:
        await db.disconnect()


if __name__ == "__main__":
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Backfill embeddings for store agents")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of embeddings to generate per batch (default: 100)",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(main(batch_size=args.batch_size)))
