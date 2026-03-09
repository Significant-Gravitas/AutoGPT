#!/usr/bin/env python3
"""Backfill quick prompts for saved business understanding records."""

import asyncio
import json
import logging

import click
from prisma.models import CoPilotUnderstanding

from backend.data import db
from backend.data.understanding import (
    BusinessUnderstanding,
    update_business_understanding_prompts,
)
from backend.data.understanding_prompts import (
    generate_understanding_prompts,
    has_prompt_generation_context,
)

logger = logging.getLogger(__name__)


async def backfill_understanding_prompts(
    batch_size: int = 100,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    summary = {
        "scanned": 0,
        "candidates": 0,
        "eligible": 0,
        "updated": 0,
        "failed": 0,
        "skipped_existing": 0,
        "skipped_no_context": 0,
    }
    offset = 0

    while True:
        records = await CoPilotUnderstanding.prisma().find_many(
            order={"id": "asc"},
            skip=offset,
            take=batch_size,
        )
        if not records:
            break

        offset += len(records)

        for record in records:
            summary["scanned"] += 1
            understanding = BusinessUnderstanding.from_db(record)

            if understanding.prompts:
                summary["skipped_existing"] += 1
                continue

            if limit is not None and summary["candidates"] >= limit:
                logger.info("Reached backfill limit of %s records", limit)
                return summary

            summary["candidates"] += 1

            if not has_prompt_generation_context(understanding):
                summary["skipped_no_context"] += 1
                continue

            summary["eligible"] += 1
            if dry_run:
                continue

            try:
                prompts = await generate_understanding_prompts(understanding)
                updated = await update_business_understanding_prompts(
                    understanding.user_id, prompts
                )
            except Exception:
                summary["failed"] += 1
                logger.exception(
                    "Failed to backfill prompts for user %s", understanding.user_id
                )
                continue

            if updated is None:
                summary["failed"] += 1
                logger.warning(
                    "Skipped backfill for user %s because the record no longer exists",
                    understanding.user_id,
                )
                continue

            summary["updated"] += 1

    logger.info("Understanding prompt backfill summary: %s", json.dumps(summary))
    return summary


async def run_backfill(
    batch_size: int = 100,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    await db.connect()
    try:
        return await backfill_understanding_prompts(
            batch_size=batch_size,
            limit=limit,
            dry_run=dry_run,
        )
    finally:
        await db.disconnect()


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Report candidates only.")
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option(
    "--batch-size", type=click.IntRange(min=1), default=100, show_default=True
)
def main(dry_run: bool, limit: int | None, batch_size: int) -> None:
    logging.basicConfig(level=logging.INFO)
    summary = asyncio.run(
        run_backfill(
            batch_size=batch_size,
            limit=limit,
            dry_run=dry_run,
        )
    )
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
