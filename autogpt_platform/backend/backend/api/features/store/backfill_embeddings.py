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

import prisma

from backend.api.features.store.embeddings import (
    backfill_missing_embeddings,
    get_embedding_stats,
)

logger = logging.getLogger(__name__)


async def main(batch_size: int = 100) -> int:
    """Run the backfill process - processes ALL missing embeddings in batches."""
    client = prisma.Prisma()
    await client.connect()
    prisma.register(client)

    try:
        stats = await get_embedding_stats()
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
        logger.info(f"Final coverage: {stats['coverage_percent']}%")

        return 0 if total_failed == 0 else 1

    finally:
        await client.disconnect()


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
