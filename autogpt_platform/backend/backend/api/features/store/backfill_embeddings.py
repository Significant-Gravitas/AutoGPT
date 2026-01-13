#!/usr/bin/env python3
"""
CLI script to backfill embeddings for store agents.

Usage:
    poetry run python -m backend.api.features.store.backfill_embeddings [--batch-size N]
"""

import argparse
import asyncio
import sys

import prisma

from backend.api.features.store.embeddings import (
    backfill_missing_embeddings,
    get_embedding_stats,
)


async def main(batch_size: int = 100) -> int:
    """Run the backfill process."""
    # Initialize Prisma client
    client = prisma.Prisma()
    await client.connect()
    prisma.register(client)

    try:
        # Get current stats
        print("Current embedding stats:")
        stats = await get_embedding_stats()
        print(f"  Total approved: {stats['total_approved']}")
        print(f"  With embeddings: {stats['with_embeddings']}")
        print(f"  Without embeddings: {stats['without_embeddings']}")
        print(f"  Coverage: {stats['coverage_percent']}%")

        if stats["without_embeddings"] == 0:
            print("\nAll agents already have embeddings. Nothing to do.")
            return 0

        # Run backfill
        print(f"\nBackfilling up to {batch_size} embeddings...")
        result = await backfill_missing_embeddings(batch_size=batch_size)
        print(f"  Processed: {result['processed']}")
        print(f"  Success: {result['success']}")
        print(f"  Failed: {result['failed']}")

        # Get final stats
        print("\nFinal embedding stats:")
        stats = await get_embedding_stats()
        print(f"  Total approved: {stats['total_approved']}")
        print(f"  With embeddings: {stats['with_embeddings']}")
        print(f"  Without embeddings: {stats['without_embeddings']}")
        print(f"  Coverage: {stats['coverage_percent']}%")

        return 0 if result["failed"] == 0 else 1

    finally:
        await client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill embeddings for store agents")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of embeddings to generate (default: 100)",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(main(batch_size=args.batch_size)))
